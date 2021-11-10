# GANSER - Official PyTorch Implementation
#### [Project Page](https://github.com/tczhangzhi/PyTorch-GANSER) | [Paper](https://arxiv.org/abs/2109.03124) | [Download](https://pan.baidu.com/s/1olReEiHvWNz-F_wQyYnnWA)
This is the official implementation of the paper "GANSER: A Self-supervised Data Augmentation Framework for EEG-based Emotion Recognition". NOTE: We are refactoring this project to the best practice of engineering.

## Roadmap
In order to help reviewers to reproduce the experimental results of this paper easily, we will open-source our study following this roadmap:

- [x] open-source all the experimental logs and notebooks for reported comparison experiments, ablation studies, and visualization experiments.
- [x] refactor the main training and evaluation scripts, trained  parameter files, and documents for out-of-box evaluation.
- [ ] refactor the temporary process-oriented DREAMER dataset preprocessors and related training codes.

## Quick Start

#### Dataset and Pre-processing

Usually, you need to download the DEAP dataset and run the whole pre-processing codes to start with an EEG-based emotion recognition project. In this project, we provide a packaged tool called preprocessors, allowing you to pre-process the DEAP dataset and cache the results on the disk, and use them without changing a single line of codes.

Here, we highly recommend you download all cached files from [here](https://pan.baidu.com/s/1olReEiHvWNz-F_wQyYnnWA) (password: rqdc) and place them in the `./dataset` fold and uncompress them:

```
cat deap_binary_valence_dataset.tar.bz2.* > deap_binary_valence_dataset.tar.gz
tar -zxvf deap_binary_valence_dataset.tar.gz
```

And in this way, you do not need to download the DEAP dataset yourself.

Otherwise, if you already downloaded the DEAP dataset, you can modify the path of the DEAP dataset `DATASET_FOLD_DIR` in `train_proposed_pretrain.py` and `finetune_proposed_pretrain.py`, and then the `preprocessors` would automatically generate the cache to `./dataset`. For the next time when you need to use these pre-processed results, the preprocessors will automatically load them from cached files.

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
