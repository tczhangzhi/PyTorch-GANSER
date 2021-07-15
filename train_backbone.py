import os
import torch
import logging
import argparse
import numpy as np
import pickle as pkl
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from sklearn import model_selection
from torch.utils.data import Dataset, Subset, DataLoader

from preprocessors import DEAPDataset, Sequence
from preprocessors import BinaryLabel
from preprocessors import Raw2TNCF, RemoveBaseline, TNCF2NCF, ChannelToLocation

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='Fold index.')
args = parser.parse_args()


class CFG:
    NUM_EPOCHS = 300
    NUM_CLASSES = 2
    BATCH_SIZE = 64
    TIMESTEP_NUM = 128
    FOLD = args.fold


# CUDA_VISIBLE_DEVICES=3 python train_backbone.py
RECEIVED_PARAMS = {"c_lr": 0.0001}
TRAIL_ID = 'cross_validation_backbone' + str(CFG.FOLD)

logger = logging.getLogger(TRAIL_ID)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('logs/{}.log'.format(TRAIL_ID))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

DATASET_BASE_DIR = Path('/home/username/Data/eeg-datasets')
DATASET_FOLD_DIR = DATASET_BASE_DIR / 'DEAP'
PREPROCESSED_EEG_DIR = DATASET_FOLD_DIR / 'data_preprocessed_python'

label_preprocessors = {'label': Sequence([BinaryLabel()])}
feature_preprocessors = {
    'feature':
    Sequence([Raw2TNCF(),
              RemoveBaseline(),
              TNCF2NCF(),
              ChannelToLocation()])
}

preprocessors_results = DEAPDataset(
    PREPROCESSED_EEG_DIR, label_preprocessors,
    feature_preprocessors)('./dataset/deap_binary_valence_dataset.pkl')


class EEGDataset(Dataset):
    def __init__(self,
                 preprocessors_results,
                 feature_key='feature',
                 label_key='label'):
        self.feature_key = feature_key
        self.label_key = label_key

        feature_list = []
        label_list = []

        for trail in preprocessors_results.keys():
            feature = preprocessors_results[trail][feature_key]
            feature_list.append(feature)
            label = preprocessors_results[trail][label_key]
            label_list.append(label)

        feature_list = np.concatenate(feature_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)

        self.feature_list = feature_list
        self.label_list = label_list

    def __getitem__(self, index):
        feature = torch.from_numpy(self.feature_list[index]).float()
        label = torch.tensor(self.label_list[index]).long()
        return feature, label

    def __len__(self):
        return len(self.label_list)


def train_test_split(dataset,
                     kfold_split_index_path='./dataset/kfold_split_index.pkl',
                     fold=0,
                     n_splits=5,
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


c_model = Classifier(num_classes=CFG.NUM_CLASSES, in_channels=CFG.TIMESTEP_NUM)


class Trainer():
    def __init__(self, c_model, trainer_kwargs={'max_epochs': 10}):
        super().__init__()
        self.c_model = c_model.cuda()

        self._loss_fn_ce = nn.CrossEntropyLoss()
        self._optimizer_c_model = torch.optim.Adam(c_model.parameters(),
                                                   lr=RECEIVED_PARAMS['c_lr'],
                                                   weight_decay=0.0005)
        self._trainer_kwargs = trainer_kwargs

        eeg_dataset = EEGDataset(preprocessors_results,
                                 feature_key='feature',
                                 label_key='label')
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
            'c_model': self.c_model.state_dict(),
        }, param_path)


trainer = Trainer(c_model, trainer_kwargs={'max_epochs': CFG.NUM_EPOCHS})
trainer.fit()
trainer.save('./parameters/' + TRAIL_ID + '.pth')