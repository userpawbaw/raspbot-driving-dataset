import argparse
import json
from pathlib import Path

import numpy as np

from raspbot_policy import STATE_SHAPE
from raspbot_receiver import ACTION_NAMES
from raspbot_recording import MODE_NAMES


def read_run(path, finish, with_mode=False):
    with np.load(path, allow_pickle=False) as data:
        if 'dry_run' in data and bool(data['dry_run']):
            raise ValueError(f'{path}: dry-run data cannot be used for training')
        x = data['states']
        if x.ndim != 5 or x.shape[1:] != STATE_SHAPE or x.dtype != np.uint8:
            raise ValueError(f'{path}: invalid states shape or dtype')
        if finish:
            goals = data['finish_states'] if 'finish_states' in data else []
            if not len(goals) or 'finish_timestamps' not in data:
                return x[:0], np.empty(0, dtype='U1')
            # Keep only clearly earlier samples. Review near-goal negatives separately.
            earlier = x[data['timestamps'] < float(data['finish_timestamps'][0]) - 1.0]
            return np.concatenate([earlier, goals]), np.array(['NO'] * len(earlier) + ['YES'] * len(goals))
        names = [str(name) for name in data['action_names']]
        by_name = {name: cmd.decode() for cmd, name in ACTION_NAMES.items()}
        ids = data['actions']
        if ids.shape != (len(x),) or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= len(names)):
            raise ValueError(f'{path}: invalid action labels')
        actions = np.array([by_name[names[i]] for i in ids])
        keep = actions != 'STOP'
        if not with_mode:
            return x[keep], actions[keep]
        required = {'schema_version', 'modes', 'mode_names', 'session_id'}
        if not required.issubset(data.files) or int(data['schema_version']) != 2:
            raise ValueError(f'{path}: --with-mode requires a version 2 NPZ from the mode directory')
        modes = data['modes']
        if (tuple(data['mode_names']) != MODE_NAMES or modes.shape != (len(x),)
                or not np.issubdtype(modes.dtype, np.integer)
                or np.any(modes < 0) or np.any(modes >= len(MODE_NAMES))):
            raise ValueError(f'{path}: invalid mode labels')
        return x[keep], actions[keep], modes[keep]


def mean_recall(matrix):
    support = matrix.sum(axis=1)
    present = support > 0
    return float((np.diag(matrix)[present] / support[present]).mean()) if present.any() else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs='+', type=Path, required=True)
    parser.add_argument('--val', nargs='+', type=Path, default=[])
    parser.add_argument('--out', type=Path, required=True)
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument('--finish', action='store_true')
    tasks.add_argument('--with-mode', action='store_true', help='Train an extra normal/recovery output')
    parser.add_argument('--mode-weight', type=float, default=0.5, help='Weight of the auxiliary mode loss')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--balanced', action='store_true', help='Use square-root inverse class weighting')
    args = parser.parse_args()
    if args.epochs < 1 or args.out.suffix != '.onnx':
        parser.error('Provide positive --epochs and an .onnx output')
    if not np.isfinite(args.mode_weight) or args.mode_weight <= 0:
        parser.error('--mode-weight must be finite and positive')
    if len({p.resolve() for p in args.train + args.val}) != len(args.train + args.val):
        parser.error('Each run must appear once, in train or validation')

    import onnx
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)

    if args.with_mode:
        sessions = set()
        for path in args.train + args.val:
            with np.load(path, allow_pickle=False) as data:
                if 'session_id' not in data:
                    parser.error(f'{path}: --with-mode requires a version 2 NPZ from the mode directory')
                session = str(data['session_id'].item())
            if session in sessions:
                parser.error(f'{path}: this recording session is already in train or validation')
            sessions.add(session)

    def load(paths):
        runs = [read_run(path, args.finish, args.with_mode) for path in paths]
        return tuple(np.concatenate(items) for items in zip(*runs))

    x, y, *mode_labels = load(args.train)
    commands = ['NO', 'YES'] if args.finish else [c.decode() for c in ACTION_NAMES if c.decode() in y]
    if len(commands) < 2 or any(not np.any(y == command) for command in commands):
        parser.error('Training data must cover at least two classes (both classes for finish)')
    if args.with_mode and set(mode_labels[0]) != set(range(len(MODE_NAMES))):
        parser.error('Training data must include moving samples in both NORMAL and RECOVERY')
    if not args.val:
        print('No held-out runs supplied. Training an end-to-end test model only.')

    def loader(images, labels, *modes, shuffle):
        ids = [commands.index(label) for label in labels]
        tensors = [torch.from_numpy(images), torch.tensor(ids)]
        if modes:
            tensors.append(torch.from_numpy(modes[0].astype(np.int64)))
        return DataLoader(TensorDataset(*tensors), batch_size=64, shuffle=shuffle)

    train = loader(x, y, *mode_labels, shuffle=True)
    val = None
    if args.val:
        vx, vy, *val_modes = load(args.val)
        if not len(vy) or set(vy) - set(commands):
            parser.error('Validation is empty or contains actions absent from training')
        if val_modes and len(np.unique(val_modes[0])) != len(MODE_NAMES):
            print('Validation has no moving samples for one mode.')
        val = loader(vx, vy, *val_modes, shuffle=False)

    features = nn.Sequential(
        nn.Conv2d(15, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(48 * 8 * 8, 64), nn.ReLU()
    )
    if args.with_mode:
        class ModePolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = features
                self.action_head = nn.Linear(64, len(commands))
                self.mode_head = nn.Linear(64, len(MODE_NAMES))

            def forward(self, state):
                embedding = self.features(state)
                return self.action_head(embedding), self.mode_head(embedding)

        model = ModePolicy()
        mode_counts = np.bincount(mode_labels[0], minlength=len(MODE_NAMES))
        print('Training modes:', dict(zip(MODE_NAMES, mode_counts.tolist())))
        mode_weights = torch.tensor(np.sqrt(mode_counts.max() / mode_counts), dtype=torch.float32)
        mode_loss = nn.CrossEntropyLoss(weight=mode_weights if args.balanced else None)
    else:
        model = nn.Sequential(*features, nn.Linear(64, len(commands)))

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    counts = np.array([(y == c).sum() for c in commands])
    print('Training labels:', dict(zip(commands, counts.tolist())))
    weights = torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights if args.balanced or args.finish else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_score = -1.0
    best = None
    for epoch in range(args.epochs):
        model.train()
        for images, labels, *modes in train:
            inputs = images.reshape(-1, 15, 32, 32).float() / 255.0
            result = model(inputs)
            if args.with_mode:
                action_logits, mode_logits = result
                loss = loss_fn(action_logits, labels) + args.mode_weight * mode_loss(mode_logits, modes[0])
            else:
                loss = loss_fn(result, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if val is not None:
            model.eval()
            matrix = np.zeros((len(commands), len(commands)), dtype=int)
            mode_matrix = np.zeros((len(MODE_NAMES), len(MODE_NAMES)), dtype=int)
            action_by_mode = np.zeros((len(MODE_NAMES), *matrix.shape), dtype=int)
            with torch.no_grad():
                for images, labels, *modes in val:
                    result = model(images.reshape(-1, 15, 32, 32).float() / 255.0)
                    action_logits = result[0] if args.with_mode else result
                    pred = action_logits.argmax(1).numpy()
                    np.add.at(matrix, (labels.numpy(), pred), 1)
                    if args.with_mode:
                        actual_modes = modes[0].numpy()
                        np.add.at(mode_matrix, (actual_modes, result[1].argmax(1).numpy()), 1)
                        np.add.at(action_by_mode, (actual_modes, labels.numpy(), pred), 1)
            score = mean_recall(matrix)
            print(f'Epoch {epoch + 1}: validation action mean recall {score:.3f}\n{matrix}')
            if args.with_mode:
                print(f'Mode mean recall {mean_recall(mode_matrix):.3f}\n{mode_matrix}')
                for name, per_mode in zip(MODE_NAMES, action_by_mode):
                    print(f'{name} action mean recall {mean_recall(per_mode):.3f} (n={per_mode.sum()})')
            if score > best_score:
                best_score = score
                best = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            print(f'Epoch {epoch + 1}: last batch loss {loss.item():.4f}')
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    outputs = ['logits', 'mode_logits'] if args.with_mode else ['logits']
    torch.onnx.export(model, torch.zeros(1, 15, 32, 32), str(args.out),
                      input_names=['state'], output_names=outputs, opset_version=17, dynamo=False)
    graph = onnx.load(args.out)
    metadata = {'task': 'finish' if args.finish else 'drive', 'commands': json.dumps(commands)}
    if args.with_mode:
        metadata.update(mode_names=json.dumps(MODE_NAMES), mode_weight=str(args.mode_weight))
    onnx.helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph)
    onnx.save(graph, args.out)
    print('Exported', args.out, 'actions:', commands, 'outputs:', outputs)


if __name__ == '__main__':
    main()
