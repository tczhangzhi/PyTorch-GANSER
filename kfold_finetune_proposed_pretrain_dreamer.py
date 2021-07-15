import os
import torch

from multiprocessing import Process

DEVICE_LIST = [7, 6, 5, 4, 3, 2, 1, 0]  # list(range(torch.cuda.device_count()))
FOLD_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def start_exp(dataset, device):
    os.system(
        'CUDA_VISIBLE_DEVICES={} python finetune_proposed_pretrain_dreamer.py --fold {}'.format(
            device, dataset))


def main():
    device_count = len(DEVICE_LIST)
    for i, dataset in enumerate(FOLD_LIST):
        device = DEVICE_LIST[i % (device_count)]
        p = Process(target=start_exp, args=(dataset, device))
        p.start()


if __name__ == '__main__':
    main()