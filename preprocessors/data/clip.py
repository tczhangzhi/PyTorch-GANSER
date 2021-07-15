import numpy as np

from ..base import _MicroPreprocessor
from ..constant import CHANNEL_LIST, LOCATION_LIST


def get_channel_location(channel_list, location_list):
    location_list = np.array(location_list)
    output = {}
    for channel in channel_list:
        location = (np.argwhere(location_list == channel)[0] + 1).tolist()
        output[channel] = location
    return output


def channel_to_location(data, norm_fn=None):
    # sample_num*channel*bands
    data = data.swapaxes(0, 1)
    # channel*sample_num*bands
    outputs = np.zeros([9, 9, *data.shape[1:]])
    channel_location = get_channel_location(CHANNEL_LIST, LOCATION_LIST)
    for i, (x, y) in enumerate(channel_location.values()):
        x = x - 1
        y = y - 1
        if norm_fn:
            outputs[x][y] = norm_fn(data[i])
        else:
            outputs[x][y] = data[i]
    # 9*9*sample_num*bands
    outputs = outputs.transpose([2, 3, 0, 1])
    return outputs


class ChannelToLocation(_MicroPreprocessor):
    def __init__(self, norm_fn=None):
        super().__init__()
        # (data - data.mean()) / data.std()
        self.norm_fn = norm_fn

    def run(self, feature):
        return channel_to_location(feature, norm_fn=self.norm_fn)