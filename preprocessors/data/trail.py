import numpy as np

from ..base import _MicroPreprocessor


def raw_to_T_N_C_F(data,
                   frequency=128,
                   channel_num=32,
                   sample_num=63,
                   start_trail_num=0,
                   end_trail_num=40):
    outputs = []
    # 40
    for trial in range(start_trail_num, end_trail_num):
        trail_samples = np.empty([0, sample_num])
        # 32
        for channel in range(channel_num):
            # n * 128
            trial_signal = data[trial, channel]
            trial_signal = trial_signal[:frequency * sample_num]
            clip_sample = trial_signal.reshape([sample_num,
                                                frequency]).transpose([1, 0])
            # 128 * n
            trail_samples = np.vstack([trail_samples, clip_sample])
        # 32 * 128 * n
        trail_samples = trail_samples.reshape(-1, frequency,
                                              sample_num).transpose([2, 0, 1])
        # n * 32 * 128
        outputs.append(trail_samples)
    # 40 * n * 32 * 128
    return np.asarray(outputs)


class Raw2TNCF(_MicroPreprocessor):
    def __init__(self,
                 frequency=128,
                 channel_num=32,
                 sample_num=63,
                 start_trail_num=0,
                 end_trail_num=40):
        super().__init__()
        self.frequency = frequency
        self.channel_num = channel_num
        self.sample_num = sample_num
        self.start_trail_num = start_trail_num
        self.end_trail_num = end_trail_num

    def run(self, feature):
        return raw_to_T_N_C_F(feature,
                              frequency=self.frequency,
                              channel_num=self.channel_num,
                              sample_num=self.sample_num,
                              start_trail_num=self.start_trail_num,
                              end_trail_num=self.end_trail_num)


def remove_baseline(data, baseline_num=3, trail_num=40):
    outputs = []
    for trial in range(trail_num):
        trial_feature = data[trial]

        trail_base_feature = trial_feature[:baseline_num].mean(axis=0,
                                                               keepdims=True)
        trail_data_feature = trial_feature[baseline_num:]

        trail_data_feature = trail_data_feature - trail_base_feature
        outputs.append(trail_data_feature)
    return np.asarray(outputs)


class RemoveBaseline(_MicroPreprocessor):
    def __init__(self, baseline_num=3, trail_num=40):
        super().__init__()
        self.baseline_num = baseline_num
        self.trail_num = trail_num

    def run(self, feature):
        return remove_baseline(feature,
                               baseline_num=self.baseline_num,
                               trail_num=self.trail_num)


def T_N_C_F_to_N_C_F(data):
    return data.reshape(-1, *data.shape[2:])


class TNCF2NCF(_MicroPreprocessor):
    def __init__(self):
        super().__init__()

    def run(self, feature):
        return T_N_C_F_to_N_C_F(feature)
