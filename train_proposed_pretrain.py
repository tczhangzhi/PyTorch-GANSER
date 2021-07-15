import torch
import logging
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from torch import autograd
from sklearn import model_selection
from torch.utils.data import Dataset, Subset, DataLoader

from preprocessors import DEAPDataset, Sequence
from preprocessors import BinaryLabel
from preprocessors import Raw2TNCF, RemoveBaseline, TNCF2NCF, ChannelToLocation

# CUDA_VISIBLE_DEVICES=3 python train_proposed_pretrain.py

RECEIVED_PARAMS = {
    "c_lr": 0.00001,
    "g_lr": 0.00001,
    "d_lr": 0.00001,
    "weight_gp": 1.0,
    "weight_decay": 0.0005
}
TRAIL_ID = 'cross_validation_proposed_pretrain'

logger = logging.getLogger(TRAIL_ID)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('logs/{}.log'.format(TRAIL_ID))

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class CFG:
    NUM_EPOCHS = 300
    NUM_CLASSES = 2
    BATCH_SIZE = 64
    TIMESTEP_NUM = 128


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


def train_test_split(dataset, test_size=0.2, random_state=520, shuffle=True):
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    train_index, test_index = model_selection.train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle)

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


class Discriminator(nn.Module):
    def __init__(self, num_classes, in_channels=4):
        super(Discriminator, self).__init__()
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

        self.drop = nn.Sequential(nn.SELU())
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
        out = self.fc1(out)
        out = self.fc2(out)
        return out


g_model = Generator(in_channels=CFG.TIMESTEP_NUM, out_channels=CFG.TIMESTEP_NUM)
d_model = Discriminator(num_classes=1, in_channels=CFG.TIMESTEP_NUM)


def random_mask(data, min_r=0.0, max_r=0.5):
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
    def __init__(self, g_model, d_model, trainer_kwargs={'max_epochs': 10}):
        super().__init__()
        self.g_model = g_model.cuda()
        self.d_model = d_model.cuda()

        self._loss_fn_ce = nn.CrossEntropyLoss()
        self._loss_fn_mse = nn.MSELoss()
        self._optimizer_g_model = torch.optim.Adam(
            g_model.parameters(),
            lr=RECEIVED_PARAMS['g_lr'],
            weight_decay=RECEIVED_PARAMS['weight_decay'])
        self._optimizer_d_model = torch.optim.Adam(
            d_model.parameters(),
            lr=RECEIVED_PARAMS['d_lr'],
            weight_decay=RECEIVED_PARAMS['weight_decay'])

        self._trainer_kwargs = trainer_kwargs

        eeg_dataset = EEGDataset(preprocessors_results,
                                 feature_key='feature',
                                 label_key='label')
        train_dataset, val_dataset = train_test_split(eeg_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CFG.BATCH_SIZE,
                                      shuffle=True,
                                      drop_last=False)

        self._train_dataloader = train_dataloader

    def _accuracy(self, input, target):  # pylint: disable=redefined-builtin
        _, predict = torch.max(input.data, 1)
        correct = predict.eq(target.data).cpu().sum().item()
        return correct / input.size(0)

    def training_step_g_model(self, batch, batch_idx, augment_fn=random_mask):
        self._optimizer_g_model.zero_grad()

        for p in self.d_model.parameters():
            p.requires_grad = False

        x, y = batch
        x, y = x.cuda(), y.cuda()

        aug_x, ratio = random_mask(x)
        pred_x = self.g_model(aug_x)
        loss = -self.d_model(pred_x).mean()

        loss.backward()
        self._optimizer_g_model.step()

        return loss

    def training_step_d_model(self, batch, batch_idx, augment_fn=random_mask):
        self._optimizer_d_model.zero_grad()

        for p in self.d_model.parameters():
            p.requires_grad = True

        x, y = batch
        x, y = x.cuda(), y.cuda()

        aug_x, ratio = random_mask(x)
        pred_x = self.g_model(aug_x).detach()

        loss = self.d_model(pred_x).mean() - self.d_model(x).mean()
        loss += RECEIVED_PARAMS['weight_gp'] * gradient_penalty(
            self.d_model, x, pred_x)

        if batch_idx % 5 == 0:
            loss.backward()
            self._optimizer_d_model.step()

        return loss

    def _train(self, epoch_idx=-1):
        pbar = tqdm(total=len(self._train_dataloader))
        pbar.set_description("[TRAIN] {}".format(epoch_idx + 1))
        for i, batch in enumerate(self._train_dataloader):
            loss_d_model = self.training_step_d_model(batch, i)
            loss_g_model = self.training_step_g_model(batch, i)
            pbar.update(1)
            pbar.set_postfix(
                ordered_dict={
                    'loss_g_model': '%.3f' % loss_g_model.item(),
                    'loss_d_model': '%.3f' % loss_d_model.item()
                })

    def fit(self) -> None:
        for i in tqdm(range(self._trainer_kwargs['max_epochs'])):
            self._train(i + 1)

    def save(self, param_path):
        torch.save(
            {
                'g_model': self.g_model.state_dict(),
                'd_model': self.d_model.state_dict()
            }, param_path)


trainer = Trainer(g_model,
                  d_model,
                  trainer_kwargs={'max_epochs': CFG.NUM_EPOCHS})
trainer.fit()
trainer.save('./parameters/' + TRAIL_ID + '.pth')