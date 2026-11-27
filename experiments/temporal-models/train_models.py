import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime'))
from raspbot_sampling import LAYOUTS, MAX_FRAME_GAP, MAX_TICK_GAP, indices

COMMANDS = ['W', 'A', 'D', 'S']
ACTIONS = ['FORWARD', 'LEFT', 'RIGHT', 'REVERSE']
MODE_NAMES = ['NORMAL', 'RECOVERY']


def load_runs(folder):
    splits = {'train': [], 'val': [], 'test': []}
    audit = []
    for number in range(1, 5):
        stem = f'run_{number:06d}'
        with np.load(folder / f'{stem}.npz', allow_pickle=False) as modes, np.load(
                folder / f'{stem} (1).npz', allow_pickle=False) as legacy:
            frames = legacy['states']
            for key in ('states', 'actions', 'timestamps', 'frame_timestamps', 'action_names'):
                value = frames if key == 'states' else legacy[key]
                if not np.array_equal(value, modes[key]):
                    raise ValueError(f'{stem}: paired recordings differ in {key}')
            if list(modes['mode_names']) != MODE_NAMES:
                raise ValueError(f'{stem}: unexpected mode labels')
            timestamps = legacy['frame_timestamps']
            action_text = legacy['action_names'][legacy['actions']]
            labels = np.full(len(frames), -1, dtype=np.int64)
            for i, action in enumerate(ACTIONS):
                labels[action_text == action] = i
            mode = modes['modes'].astype(np.int64)
            segments = modes['segment_ids']
            capture = modes['context_segment_ids'] if 'context_segment_ids' in modes else segments
            split = np.full(len(frames), 'train' if number in (1, 3) else 'val' if number == 2 else 'test')
            if number == 3:
                split[segments == 8] = 'val'
            audit.append({'file': stem, 'ticks': len(frames), 'span_seconds': float(np.ptp(legacy['timestamps'])),
                          'recorded_speed': int(legacy['speed']),
                          'retained_mode_action_counts_before_context': [
                              [int(np.sum((labels == a) & (mode == m))) for a in range(4)] for m in range(2)]})
        discontinuities = np.flatnonzero((capture[1:] != capture[:-1]) |
                                          (split[1:] != split[:-1]) |
                                          (np.diff(timestamps[:, -1]) > MAX_TICK_GAP)) + 1
        boundaries = np.r_[0, discontinuities, len(frames)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            # Keep frames first observed within this block. At a boundary the
            # oldest frame may have come from a different split or a missing tick.
            block = timestamps[left:right]
            times, source = np.unique(block, return_index=True)
            ends = np.searchsorted(times, block[:, -1])
            minimum = int(ends[0]) if left else 0
            gaps = np.flatnonzero(np.diff(times) > MAX_FRAME_GAP) + 1
            picks = {layout: [] for layout in LAYOUTS}
            targets, mode_targets, target_rows = [], [], []
            for relative in np.flatnonzero(labels[left:right] >= 0):
                end = int(ends[relative])
                gap_index = np.searchsorted(gaps, end, side='right')
                start = max(minimum, int(gaps[gap_index - 1]) if gap_index else 0)
                selected = {layout: indices(times, end, start, layout) for layout in LAYOUTS}
                if any(value is None for value in selected.values()):
                    continue
                for layout, value in selected.items():
                    picks[layout].append(source[list(value)] + left * 5)
                row = left + relative
                targets.append(labels[row])
                mode_targets.append(mode[row])
                target_rows.append(row)
            if targets:
                splits[split[left]].append((number, frames.reshape(-1, 3, 32, 32),
                                            {key: np.array(value) for key, value in picks.items()},
                                            np.array(targets), np.array(mode_targets), np.array(target_rows)))
    return splits, audit


def materialize(split, layout):
    return (torch.from_numpy(np.concatenate([frames[picks[layout]] for _, frames, picks, *_ in split])),
            torch.from_numpy(np.concatenate([a for _, _, _, a, _, _ in split])),
            torch.from_numpy(np.concatenate([m for _, _, _, _, m, _ in split])))


class Network(nn.Module):
    def __init__(self, frames, with_mode, dropout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(frames * 3, 20, 3, stride=2, padding=1), nn.BatchNorm2d(20), nn.ReLU(),
            nn.Conv2d(20, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Flatten(), nn.Linear(48 * 4 * 4, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.action = nn.Linear(64, 4)
        self.mode = nn.Linear(64, 2) if with_mode else None

    def forward(self, x):
        embedding = self.body(x)
        action = self.action(embedding)
        return (action, self.mode(embedding)) if self.mode is not None else action


def preprocess(batch, recipe):
    x = batch.float().div_(255)
    if recipe is not None:
        n, count, channels, height, width = x.shape
        luminance = (x * x.new_tensor([.299, .587, .114]).view(1, 1, 3, 1, 1)).sum(2, keepdim=True)
        sat = torch.empty(n, 1, 1, 1, 1).uniform_(.60, 1.4)
        if recipe == 'robust':
            sat *= torch.rand(n, 1, 1, 1, 1) > .12
        tint = torch.empty(n, 1, channels, 1, 1).uniform_(.85, 1.15)
        contrast = torch.empty(n, 1, 1, 1, 1).uniform_(.82, 1.18)
        brightness = torch.empty(n, 1, 1, 1, 1).uniform_(-.08, .08)
        x = (((luminance + (x - luminance) * sat) * tint - .5) * contrast + .5 + brightness)
        if recipe == 'robust':
            theta = torch.eye(2, 3).repeat(n, 1, 1)
            theta[:, :, 2].uniform_(-.11, .11)
            grid = nn.functional.affine_grid(theta, (n, count * channels, height, width), align_corners=False)
            x = nn.functional.grid_sample(x.flatten(1, 2), grid, padding_mode='border',
                                          align_corners=False).reshape_as(x)
            x += torch.randn(n, 1, channels, height, width) * .018
        x.clamp_(0, 1)
    return x.flatten(1, 2)


@torch.inference_mode()
def evaluate(model, data, with_mode):
    model.eval()
    frames, labels, modes = data
    predictions, mode_predictions = [], []
    for batch in frames.split(256):
        out = model(preprocess(batch, None))
        logits, mode = out if with_mode else (out, None)
        predictions.append(logits.argmax(1))
        if mode is not None:
            mode_predictions.append(mode.argmax(1))
    matrix = torch.bincount(labels * 4 + torch.cat(predictions), minlength=16).reshape(4, 4).numpy()
    den = matrix.sum(1)
    recall = [float(matrix[i, i] / den[i]) if den[i] else None for i in range(4)]
    stats = {'accuracy': float(matrix.trace()/matrix.sum()), 'wa_balanced': (recall[0]+recall[1])/2,
             'rare_pred_rate_on_WA': float(matrix[:2, 2:].sum()/matrix[:2].sum()),
             'recall': dict(zip(COMMANDS, recall)), 'action_confusion': matrix.tolist()}
    if with_mode:
        m = torch.bincount(modes * 2 + torch.cat(mode_predictions), minlength=4).reshape(2, 2).numpy()
        stats['mode_confusion'] = m.tolist()
        stats['mode_balanced'] = float(np.mean(m.diagonal()/m.sum(1)))
    return stats


def fit(train, val, with_mode, recipe, epochs):
    torch.manual_seed(73)
    frames, labels, modes = train
    net = Network(frames.shape[1], with_mode, .12 if recipe == 'basic' else .22)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.0005 if recipe == 'basic' else .002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.00018)
    counts = labels.bincount(minlength=4).float()
    w = (counts.max()/counts.clamp_min(1)).pow(.25).clamp_max(4)
    mode_counts = modes.bincount(minlength=2).float()
    mode_weights = (mode_counts.max()/mode_counts.clamp_min(1)).sqrt().clamp_max(4)
    best_score = -1.
    for epoch in range(epochs):
        net.train()
        for ids in torch.randperm(len(labels)).split(128):
            output = net(preprocess(frames[ids], recipe))
            action, mode = output if with_mode else (output, None)
            loss = nn.functional.cross_entropy(action, labels[ids], weight=w,
                                               label_smoothing=.01 if recipe == 'basic' else .025)
            if with_mode:
                loss += .25 * nn.functional.cross_entropy(mode, modes[ids], weight=mode_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        results = evaluate(net, val, with_mode)
        score = results['wa_balanced'] - .20 * results['rare_pred_rate_on_WA']
        if with_mode:
            score = .85 * score + .15 * results['mode_balanced']
        if score > best_score:
            best_score = score
            best = {key: value.detach().clone() for key, value in net.state_dict().items()}
            checkpoint = {'epoch': epoch + 1, 'recipe': recipe, 'score': score, **results}
        if epoch + 1 - checkpoint['epoch'] >= 4:
            break
    net.load_state_dict(best)
    return net.eval(), checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--layouts', nargs='+', choices=LAYOUTS, default=list(LAYOUTS))
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    splits, audit = load_runs(args.data)
    folder = args.output
    (folder/'models').mkdir(parents=True,exist_ok=True)
    support = {name: {'ticks': sum(len(a) for _,_,_,a,_,_ in runs),
                      'by_mode_and_action': sum(np.bincount(m * 4+a, minlength=8).reshape(2,4)
                                                for _,_,_,a,m,_ in runs).tolist()}
               for name,runs in splits.items()}
    print(json.dumps(support), flush=True)
    report = {'data': audit, 'support': support, 'actions': COMMANDS, 'layouts': LAYOUTS, 'models': {}}
    for layout in args.layouts:
        train, val, test = [materialize(splits[k], layout) for k in ('train', 'val', 'test')]
        count = train[0].shape[1]
        for with_mode in (False, True):
            name = ('with_mode_' if with_mode else 'legacy_') + layout
            choices = []
            for recipe in ('basic', 'robust'):
                model, validation = fit(train, val, with_mode, recipe, args.epochs)
                choices.append((model, validation))
                print(name, recipe, validation['epoch'], round(validation['score'], 4), flush=True)
            model, validation = max(choices, key=lambda pair: pair[1]['score'])
            path = folder/'models'/f'{name}.onnx'
            outputs = ['logits', 'mode_logits'] if with_mode else ['logits']
            torch.onnx.export(model, torch.zeros(1, count*3, 32, 32), str(path), opset_version=17,
                              dynamo=False, input_names=['state'], output_names=outputs)
            exported = onnx.load(path)
            props = {'task':'drive', 'commands':json.dumps(COMMANDS),
                     'frame_context':'short' if layout == 'recent5' else 'timestamp',
                     'sampling_layout':layout}
            if with_mode:
                props['mode_names'] = json.dumps(MODE_NAMES)
            onnx.helper.set_model_props(exported, props)
            onnx.checker.check_model(exported)
            onnx.save(exported, path)
            report['models'][name] = {'layout': layout, 'frames': count, 'mode': with_mode,
                                      'validation': validation,
                                      'trials': [result for _, result in choices],
                                      'test': evaluate(model, test, with_mode)}
        (folder/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (folder/'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        out = csv.writer(f)
        out.writerow(['model','recipe','epoch','input_frames','test_accuracy','test_WA_balanced',
                      'test_W','test_A','test_D','test_S','test_mode_balanced','test_rare_prediction_on_WA'])
        for name, result in report['models'].items():
            test = result['test']
            out.writerow([name,result['validation']['recipe'],result['validation']['epoch'],result['frames'],
                          test['accuracy'],test['wa_balanced'],*[test['recall'][c] for c in COMMANDS],
                          test.get('mode_balanced'),test['rare_pred_rate_on_WA']])


if __name__ == '__main__':
    main()
