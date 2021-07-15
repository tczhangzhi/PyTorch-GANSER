import numpy as np

from ..base import _MicroPreprocessor


def generate_four_classification_label(labels, trail_sample_num=60):
    positive_labels = labels[:, :2] > 5
    positive_labels = positive_labels.astype(np.int)

    onehot2class = np.array([[1], [2]], dtype=np.int)
    positive_labels = positive_labels.dot(onehot2class)[:, 0]

    final_positive_labels = np.empty([0])
    for i in range(len(positive_labels)):
        for _ in range(0, trail_sample_num):
            final_positive_labels = np.append(final_positive_labels,
                                              positive_labels[i])
    return final_positive_labels


class FourClassificationLabel(_MicroPreprocessor):
    def __init__(self, trail_sample_num=60, start_trail_num=0,
                 end_trail_num=40):
        super().__init__()
        self.trail_sample_num = trail_sample_num
        self.start_trail_num = start_trail_num
        self.end_trail_num = end_trail_num

    def run(self, labels):
        labels = labels[self.start_trail_num:self.end_trail_num]
        return generate_four_classification_label(labels,
                                       trail_sample_num=self.trail_sample_num)