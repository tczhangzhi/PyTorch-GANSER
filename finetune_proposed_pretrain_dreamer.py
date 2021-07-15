import os
import torch
import logging
import argparse
import numpy as np
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from torch import autograd
from sklearn import model_selection
from torch.utils.data import Dataset, Subset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='Fold index.')
args = parser.parse_args()


class CFG:
    NUM_EPOCHS = 300
    NUM_CLASSES = 2
    BATCH_SIZE = 64
    TIMESTEP_NUM = 128
    FOLD = args.fold


# CUDA_VISIBLE_DEVICES=3 python finetune_proposed_pretrain.py
RECEIVED_PARAMS = {
    "c_lr": 0.00001,
    "weight_ssl": 0.5,
}

TRAIL_ID = 'cross_validation_finetune_dreamer' + str(CFG.FOLD)

logger = logging.getLogger(TRAIL_ID)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('logs/{}.log'.format(TRAIL_ID))

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class EEGDataset(Dataset):
    def __init__(self):

        with open('./dataset/dreamer_binary_dataset.pkl', 'rb') as file:
            data = pkl.load(file)

        self.feature_list = data['feature']
        self.label_list = data['label'][:, 0]

    def __getitem__(self, index):
        feature = torch.from_numpy(self.feature_list[index]).float()
        label = torch.tensor(self.label_list[index]).long()
        return feature, label

    def __len__(self):
        return len(self.label_list)


def train_test_split(dataset,
                     kfold_split_index_path='./dataset/kfold_split_index_dreamer.pkl',
                     fold=0,
                     n_splits=10,
                     shuffle=True,
                     seed=520):
    if not os.path.exists(kfold_split_index_path):
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        kfold = model_selection.StratifiedKFold(n_splits=n_splits,
                                                shuffle=shuffle,
                                                random_state=seed)

        index_dict = {}
        for i, (train_index, test_index) in enumerate(
                kfold.split(indices, dataset.label_list)):
            index_dict[i] = {
                'train_index': train_index,
                'test_index': test_index
            }

        with open(kfold_split_index_path, 'wb') as file:
            pkl.dump(index_dict, file)
    else:
        with open(kfold_split_index_path, 'rb') as file:
            index_dict = pkl.load(file)

    index_split = index_dict[fold]
    train_index, test_index = index_split['train_index'], index_split[
        'test_index']

    trian_dataset = Subset(dataset, train_index)
    test_dataset = Subset(dataset, test_index)

    return trian_dataset, test_dataset


class Generator(nn.Module):
    def __init__(self, in_channels=4, out_channels=128):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      128,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU())
        self.delayer1 = nn.Sequential(
            nn.ConvTranspose2d(16 + 32,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True), nn.LeakyReLU())
        self.delayer2 = nn.Sequential(
            nn.ConvTranspose2d(32 + 64,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True), nn.LeakyReLU())
        self.delayer3 = nn.Sequential(
            nn.ConvTranspose2d(64 + 128,
                               128,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True))

    def forward(self, x):
        #         x = channel_to_location(x)
        mask = (x.abs().sum(dim=1, keepdim=True) > 0).float()
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = self.layer4(out3)
        out = self.delayer1(torch.cat([out, out3], dim=1))
        out = self.delayer2(torch.cat([out, out2], dim=1))
        out = self.delayer3(torch.cat([out, out1], dim=1))
        return out * mask


class ResidualConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=bias), nn.SELU(),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=bias))
        self.res = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=bias)

    def forward(self, x):
        return self.conv(x) + self.res(x)


class InceptionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv5x5 = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2,
                                 bias=bias)
        self.conv3x3 = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=bias)
        self.conv1x1 = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=bias)

    def forward(self, x):
        return self.conv5x5(x) + self.conv3x3(x) + self.conv1x1(x)


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True):
        super().__init__()
        self.depth = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               groups=in_channels,
                               bias=bias)
        self.point = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=stride,
                               padding=0,
                               bias=bias)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes, in_channels=4):
        super(Classifier, self).__init__()
        self.layer1 = nn.Conv2d(in_channels,
                                256,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=True)
        self.layer2 = nn.Conv2d(256,
                                128,
                                kernel_size=5,
                                stride=1,
                                padding=2,
                                bias=True)
        self.layer3 = nn.Conv2d(128,
                                64,
                                kernel_size=5,
                                stride=1,
                                padding=2,
                                bias=True)
        self.layer4 = SeparableConv2d(64,
                                      32,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2,
                                      bias=True)
        self.layer5 = InceptionConv2d(32, 16)
        self.drop = nn.Sequential(nn.Dropout(), nn.SELU())
        self.fc1 = nn.Sequential(nn.Linear(9 * 9 * 16, 1024, bias=True),
                                 nn.SELU())
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop(out)
        out = self.layer2(out)
        out = self.drop(out)
        out = self.layer3(out)
        out = self.drop(out)
        out = self.layer4(out)
        out = self.drop(out)
        out = self.layer5(out)
        out = self.drop(out)
        out = out.view(out.size(0), -1)
        feat = self.fc1(out)
        out = self.fc2(feat)
        return out, feat


g_model = Generator(in_channels=CFG.TIMESTEP_NUM, out_channels=CFG.TIMESTEP_NUM)
c_model = Classifier(num_classes=CFG.NUM_CLASSES, in_channels=CFG.TIMESTEP_NUM)


def random_mask(data, min_r=0.5, max_r=0.9):
    # batch_size*channel_num*time_step
    data = data.clone()
    mask = torch.rand(*data.shape[:2],
                      *([1] * (len(data.shape) - 2)),
                      device=data.device)
    # ratio = np.random.beta(1.0, 1.0, size=(data.shape[0], 1, 1, 1))
    # ratio = torch.tensor(ratio, device=mask.device).clamp(max=0.5)
    ratio = torch.rand(size=(data.shape[0], 1, 1, 1),
                       device=mask.device) * (max_r - min_r) + min_r
    mask = mask < ratio
    mask = mask.expand_as(data)
    data[mask] = 0.0
    return data, ratio


def gradient_penalty(model, real, fake):
    device = real.device
    real = real.data
    fake = fake.data
    alpha = torch.rand(real.size(0), *([1] * (len(real.shape) - 1))).to(device)
    inputs = alpha * real + ((1 - alpha) * fake)
    inputs.requires_grad_()
    outputs = model(inputs)

    gradient = autograd.grad(outputs=outputs,
                             inputs=inputs,
                             grad_outputs=torch.ones_like(outputs).to(device),
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]

    gradient = gradient.flatten(1)
    return ((gradient.norm(2, dim=1) - 1)**2).mean()


class Trainer():
    def __init__(self, c_model, g_model, trainer_kwargs={'max_epochs': 10}):
        super().__init__()
        self.c_model = c_model.cuda()
        self.g_model = g_model.cuda()

        self._loss_fn_ce = nn.CrossEntropyLoss()
        self._loss_fn_mse = nn.MSELoss()
        self._optimizer_c_model = torch.optim.Adam(c_model.parameters(),
                                                   lr=RECEIVED_PARAMS['c_lr'],
                                                   weight_decay=0.0005)

        self._trainer_kwargs = trainer_kwargs

        eeg_dataset = EEGDataset()
        train_dataset, val_dataset = train_test_split(eeg_dataset,
                                                      fold=CFG.FOLD)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CFG.BATCH_SIZE,
                                      shuffle=True,
                                      drop_last=False)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=CFG.BATCH_SIZE,
                                    shuffle=False,
                                    drop_last=False)

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def _accuracy(self, input, target):  # pylint: disable=redefined-builtin
        _, predict = torch.max(input.data, 1)
        correct = predict.eq(target.data).cpu().sum().item()
        return correct / input.size(0)

    def training_step_c_model(self, batch, batch_idx):
        for p in self.c_model.parameters():
            p.requires_grad = True

        self._optimizer_c_model.zero_grad()

        x, y = batch
        x, y = x.cuda(), y.cuda()

        y_hat, x_feat = self.c_model(x)
        loss = self._loss_fn_ce(y_hat, y)

        aug_x, ratio = random_mask(x)
        aug_x = self.g_model(aug_x).detach()
        aug_y_hat, aug_x_feat = self.c_model(aug_x)

        loss += RECEIVED_PARAMS['weight_ssl'] * (
            (1 - ratio).squeeze() * F.mse_loss(
                x_feat, aug_x_feat, reduction='none').mean(dim=-1)).mean()

        loss.backward()
        self._optimizer_c_model.step()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.validation_step_before_model(batch, batch_idx)
        y_hat, x_feat = self.c_model(x)
        return (y_hat.detach().cpu(), y.detach().cpu())

    def validation_step_before_model(self, batch, batch_idx):
        x, y = batch
        x, y = x.cuda(), y.cuda()
        return x, y

    def validation_epoch_end(self, outputs):
        # We might need dict metrics in future?
        y_hat, y = zip(*outputs)
        y_hat = torch.cat(y_hat, dim=0)
        y = torch.cat(y, dim=0)
        avg_acc = self._accuracy(y_hat, y)
        logger.info('[VAL] Average ACC at epoch end is {}'.format(avg_acc))
        return {'val_acc': avg_acc}

    def _validate(self, epoch_idx=-1):
        validation_outputs = []
        for i, batch in enumerate(self._val_dataloader):
            validation_outputs.append(self.validation_step(batch, i))
        return self.validation_epoch_end(validation_outputs)

    def _train(self, epoch_idx=-1):
        pbar = tqdm(total=len(self._train_dataloader))
        pbar.set_description("[TRAIN] {}".format(epoch_idx + 1))
        for i, batch in enumerate(self._train_dataloader):
            loss_c_model = self.training_step_c_model(batch, i)
            pbar.update(1)
            pbar.set_postfix(
                ordered_dict={'loss_c_model': '%.3f' % loss_c_model.item()})

    def fit(self) -> None:
        for i in tqdm(range(self._trainer_kwargs['max_epochs'])):
            self._train(i + 1)
            self._validate(i + 1)

        logger.info('[VAL] Final ACC at experiment end is {}'.format(
            self._validate()['val_acc']))

    def save(self, param_path):
        torch.save({
            'c_model': trainer.c_model.state_dict(),
        }, param_path)

    def load(self):
        gan_model_state_dict = torch.load(
            './parameters/cross_validation_proposed_pretrain_dreamer.pth')
        self.g_model.load_state_dict(gan_model_state_dict['g_model'])

        if os.path.exists('./parameters/cross_validation_backbone_dreamer' +
                          str(CFG.FOLD) + '.pth'):
            c_model_state_dict = torch.load(
                './parameters/cross_validation_backbone_dreamer' + str(CFG.FOLD) +
                '.pth')
            self.c_model.load_state_dict(c_model_state_dict['c_model'])


trainer = Trainer(c_model,
                  g_model,
                  trainer_kwargs={'max_epochs': CFG.NUM_EPOCHS})
trainer.load()
trainer.fit()
trainer.save('./parameters/' + TRAIL_ID + '.pth')