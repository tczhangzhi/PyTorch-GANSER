# GANSER - Official PyTorch Implementation
#### [Project Page](https://github.com/tczhangzhi/PyTorch-GANSER) | [Paper](https://arxiv.org/abs/2109.03124) |
This is the official implementation of the paper "GANSER: A Self-supervised Data Augmentation Framework for EEG-based Emotion Recognition". NOTE: For privacy reasons, we have removed pre-processed data.

## Roadmap
In order to help reviewers to reproduce the experimental results of this paper easily, we will open-source our study following this roadmap:

- [x] open-source all the experimental logs and notebooks for reported comparison experiments, ablation studies, and visualization experiments.
- [x] refactor the main training and evaluation scripts, trained  parameter files, and documents for out-of-box evaluation.

## Quick Start

#### Dataset and Pre-processing

Please download the [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) dataset [1], and modify the path to the downloaded DEAP dataset `DATASET_FOLD_DIR` in `train_proposed_pretrain.py` and `finetune_proposed_pretrain.py`. The `preprocessors` would automatically generate the cache to `./dataset`. For the next time when you need to use these pre-processed results, the preprocessors will automatically load them from locally cached files.

[1] S. Koelstra, C. Muhl, M. Soleymani, J.-S. Lee, A. Yazdani, T. Ebrahimi, T. Pun, A. Nijholt, and I. Patras, “DEAP: A database for emotion analysis; using physiological signals,” *IEEE Transactions on Affective Computing*, vol. 3, no. 1, pp. 18–31, 2011.

#### Train the proposed GAN

To obtain our designed model for generating augmented samples, you need to use the following script to train the proposed adversarial augmentation network:

```
CUDA_VISIBLE_DEVICES=0 python train_proposed_pretrain.py
```

#### Train the classifier

Then, you can utilize the following script to train our proposed classifier purely on the real samples. Notably, to speed up the training of five-fold cross-validation, we deploy five-fold training and test at the same time on five GPUs, with kfold_finetune_proposed_pretrain.py.

```
python kfold_train_backbone.py
```

You can manually run the script `CUDA_VISIBLE_DEVICES=0 python kfold_train_backbone.py --fold n` (here, `n` ranges from zero to four) one by one, of course, if you do not have enough devices.

#### Finetune the classifier

Finally, as we described in the paper, we propose a multi-factor training network for fine-tuning the classifier with augmented samples. You can run the following script to deploy five-fold training and test at the same time on five GPUs:

```
python kfold_finetune_proposed_pretrain.py
```

You can manually run the script `CUDA_VISIBLE_DEVICES=0 python finetune_proposed_pretrain.py --fold n` (here, `n` ranges from zero to four) one by one, of course, if you do not have enough devices.

## Contact

If you have any questions, please feel free to open an issue.
