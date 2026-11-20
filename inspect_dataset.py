import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    for path in args.files:
        with np.load(path, allow_pickle=False) as data:
            actions = data['actions']
            names = data['action_names']
            timestamps = data['timestamps']
            finish_count = len(data['finish_states']) if 'finish_states' in data else 0
            print(path, 'total:', len(actions), 'finish:', finish_count)
            print(dict(zip(names, np.bincount(actions, minlength=len(names)).tolist())))
            if 'modes' in data:
                modes = data['modes']
                segments = data['segment_ids']
                print('Session:', data['session_id'].item(), '| Legacy file:', data['legacy_filename'].item())
                for mode_id, mode in enumerate(data['mode_names']):
                    selected = modes == mode_id
                    counts = np.bincount(actions[selected], minlength=len(names))
                    moving = int(counts[[name != 'STOP' for name in names]].sum())
                    print(f'{mode}: {moving} moving samples, {len(np.unique(segments[selected]))} segments')
                    print(dict(zip(names, counts.tolist())))
            if len(timestamps) > 1 and timestamps[-1] > timestamps[0]:
                print(f'Saved samples per elapsed second (includes pauses): {(len(timestamps) - 1) / (timestamps[-1] - timestamps[0]):.2f} Hz')


if __name__ == '__main__':
    main()
