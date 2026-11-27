"""Train a recovery-aware driving policy from existing Raspbot mode/*.npz files.

Example:
    python train_raspbot_recovery.py --train data/mode/run_000001.npz \
        data/mode/run_000002.npz --val data/mode/run_000003.npz \
        --out models/recovery.onnx

Requires numpy, torch and onnx. --inspect works with numpy alone.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from raspbot_context import FRAME_SHAPE, HISTORY_OFFSETS, SHORT_FRAMES, context_rows, frame_count

MODE_NAMES = ("NORMAL", "RECOVERY")
ACTION_NAMES = {"W": "FORWARD", "A": "LEFT", "D": "RIGHT", "S": "REVERSE"}
PAIRS = ((0, "W"), (0, "A"), (1, "S"), (1, "D"), (1, "W"), (1, "A"))
STATE_SHAPE = (SHORT_FRAMES, *FRAME_SHAPE)
EVENT_NAMES = ("reverse start", "right after reverse", "normal left start")


def name(pair):
    mode, command = pair
    return f"{MODE_NAMES[mode]}/{command}"


@dataclass
class Run:
    images: np.ndarray
    labels: np.ndarray
    events: np.ndarray
    rows: np.ndarray
    past_rows: dict


def read_run(path, required_contexts=()):
    with np.load(path, allow_pickle=False) as data:
        required = {"states", "actions", "action_names", "modes", "mode_names",
                    "segment_ids", "session_id", "timestamps"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: use mode/*.npz; missing {sorted(missing)}")
        if "dry_run" in data and bool(data["dry_run"]):
            raise ValueError(f"{path}: dry-run recording")
        session = str(data["session_id"].item())
        images = data["states"]
        if images.ndim != 5 or images.shape[1:] != STATE_SHAPE or images.dtype != np.uint8:
            raise ValueError(f"{path}: expected uint8 [N,5,3,32,32], found {images.shape}, {images.dtype}")
        actions, modes, segments, times = (data[key] for key in
                                           ("actions", "modes", "segment_ids", "timestamps"))
        for values in (actions, modes, segments):
            if values.shape != (len(images),) or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{path}: invalid actions, modes or segment IDs")
        if times.shape != (len(images),) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError(f"{path}: invalid timestamps")
        past_rows = {}
        if required_contexts:
            capture_segments = data["context_segment_ids"] if "context_segment_ids" in data else segments
            if capture_segments.shape != segments.shape or not np.issubdtype(capture_segments.dtype, np.integer):
                raise ValueError(f"{path}: invalid context_segment_ids")
            past_rows = context_rows(data["frame_timestamps"], capture_segments)
        action_names = data["action_names"].astype(str)
        if action_names.ndim != 1 or np.any(actions < 0) or np.any(actions >= len(action_names)):
            raise ValueError(f"{path}: action index outside action_names")
        if tuple(data["mode_names"].astype(str)) != MODE_NAMES or np.any((modes < 0) | (modes > 1)):
            raise ValueError(f"{path}: invalid mode mapping")

    labels = np.full(len(images), -1, dtype=np.int64)
    action_text = action_names[actions]
    for index, (mode, command) in enumerate(PAIRS):
        labels[(modes == mode) & (action_text == ACTION_NAMES[command])] = index
    # Keep raw row positions and timestamps so discarded labels never join segments.
    events = np.zeros(len(images), dtype=np.int64)
    previous = -1
    previous_time = -np.inf
    previous_segment = None
    active_event = 0
    remaining = 0
    stopped = False
    starts = [0] * (len(EVENT_NAMES) + 1)
    for row, (label, segment, timestamp) in enumerate(zip(labels, segments, times)):
        if segment != previous_segment or timestamp - previous_time > 0.45:
            previous = -1
            active_event = remaining = 0
        if label < 0:
            active_event = remaining = 0
            stopped = True
            if action_text[row] != "STOP":
                previous = -1
            continue
        if label != previous or (stopped and label == 2):
            if label == 2:
                active_event = 1
            elif label == 3 and previous == 2:
                active_event = 2
            elif label == 1 and previous == 0:
                active_event = 3
            else:
                active_event = 0
            remaining = 3 if active_event else 0
            starts[active_event] += 1
        if remaining:
            events[row] = active_event
            remaining -= 1
        previous, previous_time, previous_segment = label, timestamp, segment
        stopped = False

    kept = labels >= 0
    rows = np.flatnonzero(kept)
    if required_contexts:
        for layout in required_contexts:
            rows = rows[past_rows[layout][rows, 0] >= 0]
    print(f"{path}: kept {len(rows)}/{kept.sum()} eligible targets ({len(kept)} total ticks) | "
          f"STOP={np.sum(action_text == 'STOP')} | "
          f"out-of-policy={np.sum(~kept & (action_text != 'STOP'))} | "
          f"reverse starts={starts[1]} | S→D starts={starts[2]}")
    return session, Run(images, labels[rows], events[rows], rows, past_rows)


def load_split(paths, seen, required_contexts=()):
    runs = []
    for path in paths:
        session, run = read_run(path, required_contexts)
        if session in seen:
            raise ValueError(f"{path}: duplicate session_id across input files")
        seen.add(session)
        runs.append(run)
    if not any(len(run.rows) for run in runs):
        raise ValueError("This split has no usable samples. Check context coverage in each recording segment.")
    return runs


def input_frames(run, index, layout):
    row = run.rows[index]
    if layout == "short":
        return run.images[row].copy()
    past = run.images[run.past_rows[layout][row], -1]
    current = run.images[row, -1:] if layout == "long" else run.images[row]
    return np.concatenate((past, current))


def augment(torch, inputs):
    # One set of parameters per clip, shared across the full context.
    clips = inputs.reshape(len(inputs), -1, *FRAME_SHAPE)
    random = torch.rand((len(clips), 5), device=clips.device)
    params = (random * 2 - 1).T[:, :, None, None, None]
    angle = params[0] * 0.8
    saturation = (1 + params[1] * 0.45) * (random[:, 4, None, None, None] >= 0.12)
    red, green, blue = clips.unbind(dim=2)
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    u, v = blue - luma, red - luma
    blue_delta = saturation * (angle.cos() * u - angle.sin() * v)
    red_delta = saturation * (angle.sin() * u + angle.cos() * v)
    green_delta = -(0.299 * red_delta + 0.114 * blue_delta) / 0.587
    colored = luma.unsqueeze(2) + torch.stack((red_delta, green_delta, blue_delta), dim=2)
    contrast = 1 + 0.18 * params[2].unsqueeze(2)
    brightness = 0.08 * params[3].unsqueeze(2)
    return ((colored - 0.5) * contrast + 0.5 + brightness).clamp(0, 1).reshape_as(inputs)


def train_model(args, train, val, labels, layout, out):
    import onnx
    import torch
    from torch import nn
    from torch.utils.data import ConcatDataset, DataLoader, Dataset

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    channels = frame_count(layout) * FRAME_SHAPE[0]
    remap = np.full(len(PAIRS), -1, dtype=int)
    remap[labels] = np.arange(len(labels))
    pairs = [PAIRS[index] for index in labels]
    command_names = list(dict.fromkeys(command for _, command in pairs))
    modes = np.array([mode for mode, _ in pairs])

    class SessionDataset(Dataset):
        def __init__(self, run):
            self.run = run
            self.targets = remap[run.labels]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return input_frames(self.run, index, layout), self.targets[index], self.run.events[index]

    def loader(split, shuffle):
        dataset = ConcatDataset([SessionDataset(run) for run in split if len(run.rows)])
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                          generator=torch.Generator().manual_seed(args.seed))

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(channels, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(48 * 8 * 8, 64), nn.ReLU(),
                nn.Linear(64, len(pairs)),
            )
            self.command_groups = [
                [i for i, (_, value) in enumerate(pairs) if value == command]
                for command in command_names]
            self.mode_groups = [np.flatnonzero(modes == mode).tolist() for mode in range(2)]

        def forward(self, images):
            return self.network(images)

    class ReceiverOutput(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, images):
            joint = self.policy(images)
            # One winning joint class determines both the command and the mode,
            # including exact ties. Keep the receiver's two original outputs.
            winning = joint.argmax(dim=1)
            command_ids = joint.new_tensor([command_names.index(c) for _, c in pairs], dtype=torch.long)
            mode_ids = joint.new_tensor(modes.tolist(), dtype=torch.long)
            action_scores = torch.stack([joint[:, indices].amax(dim=1)
                                         for indices in self.policy.command_groups], dim=1)
            mode_scores = torch.stack([joint[:, indices].amax(dim=1)
                                       for indices in self.policy.mode_groups], dim=1)
            command_mask = command_ids[winning, None] == torch.arange(len(command_names), device=joint.device)
            mode_mask = mode_ids[winning, None] == torch.arange(2, device=joint.device)
            return action_scores.masked_fill(~command_mask, -1e6), mode_scores.masked_fill(~mode_mask, -1e6)

    def prepare(images):
        return images.to(device=device, dtype=torch.float32).reshape(-1, channels, *FRAME_SHAPE[1:]) / 255

    model = Policy().to(device)
    counts = np.bincount(remap[np.concatenate([run.labels for run in train])], minlength=len(labels))
    weights = torch.tensor(np.sqrt(counts.max() / counts).clip(1, 4), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader, val_loader = loader(train, True), loader(val, False)
    best_score, best_state = -1.0, None
    recovery_only = np.array([command in ("S", "D") for _, command in pairs])

    for epoch in range(args.epochs):
        model.train()
        for images, targets, events in train_loader:
            inputs = prepare(images)
            if not args.no_augment:
                inputs = augment(torch, inputs)
            targets, events = targets.to(device), events.to(device)
            sample_weights = weights[targets] * torch.where((events == 1) | (events == 2), args.transition_weight, 1.0)
            losses = nn.functional.cross_entropy(model(inputs), targets, reduction="none")
            loss = (losses * sample_weights).sum() / sample_weights.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        matrix = np.zeros((len(pairs), len(pairs)), dtype=int)
        hits = np.zeros(len(EVENT_NAMES), dtype=int)
        totals = np.zeros_like(hits)
        with torch.no_grad():
            for images, targets, events in val_loader:
                predicted = model(prepare(images)).argmax(dim=1).cpu().numpy()
                actual, event = targets.numpy(), events.numpy()
                np.add.at(matrix, (actual, predicted), 1)
                for index in range(len(EVENT_NAMES)):
                    mask = event == index + 1
                    hits[index] += np.sum((actual == predicted) & mask)
                    totals[index] += np.sum(mask)
        support = matrix.sum(axis=1)
        present = support > 0
        score = (matrix.diagonal()[present] / support[present]).mean()
        normal = modes == 0
        false_recovery = matrix[normal][:, recovery_only].sum() / support[normal].sum()
        results = ", ".join(f"{label}={hit}/{total}" for label, hit, total in zip(EVENT_NAMES, hits, totals))
        print(f"Epoch {epoch + 1}/{args.epochs}: joint macro recall={score:.3f}, "
              f"NORMAL false S/D={false_recovery:.1%}, {results}", flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_matrix = matrix.copy()
            best_metrics = {
                "epoch": epoch + 1,
                "macro_recall": float(score),
                "normal_false_reverse_or_right": float(false_recovery),
                "transition_hits": {label: [int(hit), int(total)]
                                    for label, hit, total in zip(EVENT_NAMES, hits, totals)},
                "labels": [name(pair) for pair in pairs],
                "confusion_matrix": matrix.tolist(),
            }

    model.load_state_dict(best_state)
    model.cpu().eval()
    print("Best validation confusion matrix (rows=actual, columns=predicted)")
    print([name(pair) for pair in pairs])
    print(best_matrix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as directory:
        artifact = Path(directory) / "policy.onnx"
        torch.onnx.export(ReceiverOutput(model).eval(), torch.zeros(1, channels, *FRAME_SHAPE[1:]), str(artifact),
                          input_names=["state"], output_names=["logits", "mode_logits"],
                          opset_version=17, dynamo=False)
        graph = onnx.load(artifact)
        onnx.helper.set_model_props(graph, {
            "task": "drive",
            "frame_context": layout,
            "commands": json.dumps(command_names),
            "mode_names": json.dumps(MODE_NAMES),
            "training_pairs": json.dumps(pairs),
        })
        onnx.checker.check_model(graph)
        onnx.save(graph, artifact)
        artifact.replace(out)
    print(f"Exported {out}")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--val", nargs="+", type=Path, default=[])
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--out", type=Path)
    outputs.add_argument("--compare-dir", type=Path, help="train all three layouts on identical rows")
    parser.add_argument("--context", choices=HISTORY_OFFSETS, default="short",
                        help="input layout for a single model (default: short)")
    parser.add_argument("--inspect", action="store_true", help="report coverage without training")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transition-weight", type=float, default=1.5)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()
    if not args.inspect and (not args.val or (args.out is None and args.compare_dir is None)):
        parser.error("training requires --val with separate sessions and --out or --compare-dir")
    if args.compare_dir is not None and args.context != "short":
        parser.error("--compare-dir trains every layout; omit --context")
    if args.compare_dir is not None:
        required_contexts = ("long", "hybrid")
    else:
        required_contexts = () if args.context == "short" else (args.context,)
    if args.epochs < 1 or args.batch_size < 1 or not (np.isfinite(args.lr) and args.lr > 0):
        parser.error("epochs, batch size and learning rate must be positive")
    if not (np.isfinite(args.transition_weight) and args.transition_weight > 0):
        parser.error("transition weight must be positive")

    try:
        seen = set()
        train = load_split(args.train, seen, required_contexts)
        val = load_split(args.val, seen, required_contexts) if args.val else None
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    split_labels = {}
    for split_name, runs in (("train", train), ("val", val)):
        if runs is None:
            continue
        labels = np.concatenate([run.labels for run in runs])
        events = np.concatenate([run.events for run in runs])
        split_labels[split_name] = labels
        counts = np.bincount(labels, minlength=len(PAIRS))
        transition_counts = np.bincount(events, minlength=len(EVENT_NAMES) + 1)[1:]
        print(f"{split_name}: {dict(zip(map(name, PAIRS), counts.tolist()))}")
        print(f"{split_name} transition frames: {dict(zip(EVENT_NAMES, transition_counts.tolist()))}")
    if args.inspect:
        return

    labels = np.unique(split_labels["train"])
    for split_name, targets in split_labels.items():
        missing = set(range(4)).difference(targets)
        if missing:
            parser.error(f"{split_name} is missing required classes: {[name(PAIRS[i]) for i in sorted(missing)]}")
    unknown = np.setdiff1d(split_labels["val"], labels)
    if len(unknown):
        parser.error(f"validation classes absent from training: {[name(PAIRS[i]) for i in unknown]}")
    if args.compare_dir is None:
        train_model(args, train, val, labels, args.context, args.out)
        return

    comparison = {
        "train_rows": len(split_labels["train"]),
        "validation_rows": len(split_labels["val"]),
        "seed": args.seed,
        "models": {},
    }
    for layout in HISTORY_OFFSETS:
        print(f"\n=== {layout}: {frame_count(layout)} frames ===", flush=True)
        comparison["models"][layout] = train_model(
            args, train, val, labels, layout, args.compare_dir / f"{layout}.onnx"
        )
    report = args.compare_dir / "comparison.json"
    report.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Saved comparison to {report}")


if __name__ == "__main__":
    main()
