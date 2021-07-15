import os
import torch

from multiprocessing import Process

DEVICE_LIST = list(range(torch.cuda.device_count())) # [0, 1, 2, 3, 4, 5, 6, 7] for my local setting
FOLD_LIST = [0, 1, 2, 3, 4]


def start_exp(dataset, device):
    os.system(
        'CUDA_VISIBLE_DEVICES={} python finetune_proposed_wossl_pretrain.py --fold {}'.format(
            device, dataset))


def main():
    device_count = len(DEVICE_LIST)
    for i, dataset in enumerate(FOLD_LIST):
        device = DEVICE_LIST[i % (device_count)]
        p = Process(target=start_exp, args=(dataset, device))
        p.start()


if __name__ == '__main__':
    main()