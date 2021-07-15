import pdb

from ..base import _MicroPreprocessor


class Sequence(_MicroPreprocessor):
    def __init__(self, processers):
        super().__init__()
        self.processers = processers

    def run(self, inputs):
        outputs = inputs
        for processer in self.processers:
            outputs = processer(outputs, **self.params)
        return outputs

class PDBTrace(_MicroPreprocessor):
    def __init__(self, print=False):
        super().__init__()
        self.print = False

    def run(self, inputs):
        if self.print:
            print(inputs)
        pdb.set_trace()