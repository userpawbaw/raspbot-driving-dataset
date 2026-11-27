"""Calibrate the binary mode threshold on the internal validation split only."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from train_models import LAYOUTS, load_runs


def mode_margins(session, split, layout):
    margins, truth = [], []
    for _, frames, picks, _, modes, _ in split:
        truth.extend(modes)
        for pick in picks[layout]:
            model_input = frames[pick].astype(np.float32).reshape(1, -1, 32, 32) / 255
            score = session.run(['mode_logits'], {'state': model_input})[0][0]
            margins.append(score[1] - score[0])
    return np.asarray(margins), np.asarray(truth)


def confusion(margins, truth, threshold):
    matrix = np.bincount(2 * truth + (margins > threshold), minlength=4).reshape(2, 2)
    return matrix.tolist(), float(np.mean(matrix.diagonal() / matrix.sum(1)))


def choose_threshold(margins, truth):
    distinct = np.unique(margins.astype(np.float64))
    candidates = np.r_[0., (distinct[:-1] + distinct[1:]) / 2]
    return min(candidates, key=lambda t: (-confusion(margins, truth, t)[1], abs(t)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    metrics = args.output / 'metrics.json'
    report = json.loads(metrics.read_text(encoding='utf-8'))
    splits, _ = load_runs(args.data)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for layout in LAYOUTS:
        name = f'with_mode_{layout}'
        model = report['models'][name]
        if 'mode_calibration' in model:
            continue
        path = args.output / 'models' / f'{name}.onnx'
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        validation, actual_val = mode_margins(session, splits['val'], layout)
        threshold = float(np.float32(choose_threshold(validation, actual_val)))
        test, actual_test = mode_margins(session, splits['test'], layout)
        model['mode_calibration'] = {'margin_threshold': threshold}
        for key, margins, truth in [('validation', validation, actual_val), ('test', test, actual_test)]:
            matrix, balanced = confusion(margins, truth, threshold)
            before, before_balanced = confusion(margins, truth, 0.)
            model['mode_calibration'][key] = {'uncalibrated_confusion': before,
                                             'uncalibrated_balanced': before_balanced}
            if key == 'test':
                model[key]['mode_confusion'], model[key]['mode_balanced'] = matrix, balanced
            else:
                model['mode_calibration'][key].update(confusion=matrix, balanced=balanced)
        if threshold != 0.:
            graph = onnx.load(path)
            for node in graph.graph.node:
                for names in (node.input, node.output):
                    for i, output in enumerate(names):
                        if output == 'mode_logits':
                            names[i] = 'uncalibrated_mode_logits'
            offset = np.asarray([0., -threshold], dtype=np.float32)
            graph.graph.initializer.append(numpy_helper.from_array(offset, name='mode_threshold_offset'))
            graph.graph.node.append(helper.make_node('Add',
                                                     ['uncalibrated_mode_logits', 'mode_threshold_offset'],
                                                     ['mode_logits']))
            onnx.checker.check_model(graph)
            onnx.save(graph, path)
        print(name, 'threshold', round(threshold, 3), 'test calibrated', model['test']['mode_balanced'], flush=True)
    metrics.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    table = args.output / 'results.csv'
    with table.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    for row in rows:
        test = report['models'][row['model']]['test']
        if 'mode_balanced' in test:
            row['test_mode_balanced'] = test['mode_balanced']
    with table.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
