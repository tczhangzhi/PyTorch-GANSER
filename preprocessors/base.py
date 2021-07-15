import os
import pickle as pkl


class _MicroPreprocessor(object):
    def __init__(self):
        self.params = {}

    def run(self, inputs):
        super().__init__()
        raise NotImplementedError

    def __call__(self, inputs, **kwargs):
        self.register(**kwargs)
        return self.run(inputs)

    def register(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value
        return self


class _DatasetPreprocessor(object):
    def __init__(self, root_path, label_pipline={}, feature_pipline={}):
        self.root_path = root_path
        self.label_pipline = label_pipline
        self.feature_pipline = feature_pipline

    def run(self):
        super().__init__()
        raise NotImplementedError

    def label_run(self, labels, **kwargs):
        label_pipline = self.label_pipline
        return {
            k: sequence(labels, **kwargs)
            for k, sequence in label_pipline.items()
        }

    def feature_run(self, feature, **kwargs):
        feature_pipline = self.feature_pipline
        return {
            k: sequence(feature, **kwargs)
            for k, sequence in feature_pipline.items()
        }

    def __call__(self, pkl_path):
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as file:
                return pkl.load(file)
        outputs = self.run()
        with open(pkl_path, 'wb') as file:
            pkl.dump(outputs, file)
        return outputs