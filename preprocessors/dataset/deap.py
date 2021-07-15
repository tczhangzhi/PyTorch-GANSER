import os
import pickle as pkl
from tqdm import tqdm

from ..base import _DatasetPreprocessor


def read_deap_feature(path):
    data = pkl.load(open(path, 'rb'), encoding='iso-8859-1')
    return data['data']


def read_deap_label(path):
    data = pkl.load(open(path, 'rb'), encoding='iso-8859-1')
    return data['labels']


class DEAPDataset(_DatasetPreprocessor):
    def __init__(self, root_path, label_pipline={}, feature_pipline={}):
        super(DEAPDataset, self).__init__(root_path, label_pipline,
                                                      feature_pipline)

    def run(self):
        outputs = {}
        file_list = os.listdir(self.root_path)
        pbar = tqdm(total=len(file_list))
        pbar.set_description("[PROCESS]")
        for file_name in file_list:
            feature = read_deap_feature(os.path.join(self.root_path, file_name))
            labels = read_deap_label(os.path.join(self.root_path, file_name))
            key = os.path.splitext(file_name)[0]

            output = {}
            output.update(self.label_run(labels))
            output.update(self.feature_run(feature))

            outputs[key] = output
            pbar.update(1)
            pbar.set_postfix(ordered_dict={'file': file_name})
        pbar.close()
        return outputs