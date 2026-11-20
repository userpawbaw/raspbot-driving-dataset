import json

import numpy as np

from raspbot_recording import MODE_NAMES

STATE_SHAPE = (5, 3, 32, 32)


def state_input(frames):
    return np.asarray(frames, dtype=np.float32).reshape(1, 15, 32, 32) / 255.0


class Policy:
    def __init__(self, path, finish=False):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name
        metadata = self.session.get_modelmeta().custom_metadata_map
        expected = 'finish' if finish else 'drive'
        if metadata.get('task') != expected:
            raise ValueError(f'{path}: expected a {expected} model')
        self.modes = tuple(json.loads(metadata.get('mode_names', '[]')))
        self.outputs = [output.name for output in self.session.get_outputs()]
        self.action_output = 'logits' if 'logits' in self.outputs else self.outputs[0]
        if self.modes and (self.modes != MODE_NAMES or 'mode_logits' not in self.outputs):
            raise ValueError('Invalid ONNX mode mapping')
        self.commands = [s.encode('ascii') for s in json.loads(metadata['commands'])]
        if not finish:
            from raspbot_receiver import COMMANDS
            if not self.commands or any(command not in COMMANDS for command in self.commands):
                raise ValueError('Invalid ONNX command mapping')

    def logits(self, x):
        result = self.session.run([self.action_output], {self.input: x})[0][0]
        if not np.isfinite(result).all():
            raise RuntimeError('Non-finite model output')
        return result

    def action(self, x):
        return self._command(self.logits(x))

    def predict(self, x):
        if not self.modes:
            return self.action(x), None
        action, mode = self.session.run([self.action_output, 'mode_logits'], {self.input: x})
        mode = mode[0]
        if mode.shape != (len(self.modes),) or not np.isfinite(mode).all():
            raise RuntimeError('Invalid mode output')
        return self._command(action[0]), self.modes[int(mode.argmax())]

    def _command(self, logits):
        if logits.shape != (len(self.commands),) or not np.isfinite(logits).all():
            raise RuntimeError('Invalid action output')
        return self.commands[int(logits.argmax())]

    def finish_score(self, x):
        logits = self.logits(x)
        exp = np.exp(logits - logits.max())
        return float(exp[1] / exp.sum())
