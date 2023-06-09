import argparse
import collections
import json
import os
import warnings
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from parse_config import ConfigParser
import torch
import torch.nn as nn
import torch.nn.functional as F



def parser_option():
    parser = argparse.ArgumentParser(description='OpenCompatible Template')
    parser.add_argument('--name', default=None, type=str,
                        help='model name')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--use_pos_sampler', action='store_true',
                        help='use the positive sampler or not')
    parser.add_argument('--use_amp', action='store_true')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags default type help target')
    # parser options about model and dataset
    options = [
        CustomArgs(['--train_data_dir'], default=None, type=str, help='training image path',
                   target='dataset;data_dir'),
        CustomArgs(['--train_img_list'], default=None, type=str, help='training image list',
                   target='dataset;img_list'),
        CustomArgs(['--train_img_size'], default=224, type=int, help='training image size',
                   target='dataset;img_size'),
        CustomArgs(['--arch'], default='resnet50', type=str, help='new model arch',
                   target='new_model;arch'),
        CustomArgs(['--old_arch'], default=None, type=str, help='old model arch',
                   target='old_model;arch'),
        CustomArgs(['--emb_dim'], default=512, type=int, help='embedding dimension',
                   target='new_model;emb_dim'),
        CustomArgs(['--old_emb_dim'], default=512, type=int, help='old embedding dimension',
                   target='old_model;emb_dim'),
        CustomArgs(['--pretrained_model_path'], default=None, type=str, help='',
                   target='new_model;pretrained_model_path'),
        CustomArgs(['--model_key_in_ckpt'], default=None, type=str,
                   help='if the pretrained model is ckpt type, model_key_in_ckpt is required.',
                   target='new_model;model_key_in_ckpt'),
        CustomArgs(['-r', '--resume'], default=None, type=str, help='',
                   target='path of the latest checkpoint (default: None)'),
        CustomArgs(['--class_num'], default=93431, type=int, help='',
                   target='dataset;class_num'),
        CustomArgs(['--old_pretrained_model_path'], default=None, type=str, help='',
                   target='old_model;pretrained_model_path'),
    ]

    # parser options about optimizer and lr scheduler
    options.extend([
        CustomArgs(['--lr_scheduler'], default='cosine', type=str, help='Options: cosine or step or face_step',
                   target='lr_scheduler;type'),
        CustomArgs(['--lr_adjust_interval'], default=6, type=int,
                   help='only works when scheduler is STEP type',
                   target='lr_scheduler;lr_adjust_interval'),
        CustomArgs(['--lr'], default=0.1, type=float, help='',
                   target='optimizer;lr'),
        CustomArgs(['--momentum'], default=0.9, type=float, help='',
                   target='optimizer;momentum'),  
        CustomArgs(['--weight_decay'], default=1e-4, type=float, help='',
                   target='optimizer;weight_decay'),           
        CustomArgs(['-e', '--epochs'], default=30, type=int, help='',
                   target='trainer;epochs'),
        CustomArgs(['-bs', '--batch_size'], default=192, type=int, help='',
                   target='trainer;batch_size'),
        CustomArgs(['--workers'], default=8, type=int, help='',
                   target='trainer;num_workers'),
        CustomArgs(['--save_period'], default=1, type=int, help='',
                   target='trainer;save_period'),
        CustomArgs(['--val_period'], default=5, type=int, help='',
                   target='trainer;val_period'),
    ])

    # parser options about loss
    options.extend([
        CustomArgs(['--loss_func'], default='softmax', type=str, help='Options: softmax / arcface / cosface',
                   target='loss;type'),
        CustomArgs(['--arcface_scale'], default=64.0, type=float, help='',
                   target='loss;scale'),
        CustomArgs(['--arcface_margin'], default=0.5, type=float, help='',
                   target='loss;margin'),
    ])

    # parser options about backward compatible loss
    options.extend([
        CustomArgs(['--comp_loss_type'], default='bct', type=str,
                   help='Options: bct / bct_ract / lwf / fd / contra / triplet / l2 / contra_ract / triplet_ract / UiBCT / UiBCT_simple',
                   target='comp_loss;type'),
        CustomArgs(['--comp_loss_temp'], default=0.01, type=float, help='',
                   target='comp_loss;temperature'),
        CustomArgs(['--comp_loss_dist_temp'], default=0.01, type=float, help='',
                   target='comp_loss;distillation_temp'),
        CustomArgs(['--comp_loss_triplet_margin'], default=0.8, type=float, help='',
                   target='comp_loss;triplet_margin'),
        CustomArgs(['--comp_loss_topk'], default=10, type=int, help='',
                   target='comp_loss;topk_neg'),
        CustomArgs(['--comp_loss_focal_beta'], default=5.0, type=float, help='',
                   target='comp_loss;focal_beta'),
        CustomArgs(['--comp_loss_focal_alpha'], default=1.0, type=float, help='',
                   target='comp_loss;focal_alpha'),
        CustomArgs(['--comp_loss_weight'], default=1.0, type=float, help='',
                   target='comp_loss;weight'),

        CustomArgs(['--comp_loss_scale'], default=64.0, type=float, help='',
                   target='comp_loss;scale'),
        CustomArgs(['--comp_loss_margin'], default=0.5, type=float, help='',
                   target='comp_loss;margin'),
        CustomArgs(['--comp_loss_lamda'], default=0.9, type=float, help='',
                   target='comp_loss;lamda'),
        CustomArgs(['--comp_loss_hyper'], default=0.05, type=float, help='',
                   target='comp_loss;hyper'),
        CustomArgs(['--comp_loss_weight_uibct'], default=1.0, type=float, help='',
                   target='comp_loss;weight_uibct'),

        CustomArgs(['--gather_all'], default=True, type=bool, help='',
                   target='gather_all'),
    ])

    # parser options about test dataset
    options.extend([
        CustomArgs(['--test_data_type'], default='landmark', type=str,
                   help='Options: landmark / face',
                   target='test_dataset;type'),
        CustomArgs(['--test_data_name'], default='gldv2', type=str,
                   help='Options: roxford / rparis / ijbc',
                   target='test_dataset;name'),
        CustomArgs(['--test_data_dir'], default=None, type=str, help='test image path',
                   target='test_dataset;data_dir'),
        CustomArgs(['--test_img_size'], default=224, type=int, help='test images size',
                   target='test_dataset;img_size'),
    ])

    config = ConfigParser.from_args(parser, options)
    return config


def cudalize(model, args):
    """Select cuda or cpu mode on different machine"""
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.device is not None:
            torch.cuda.set_device(args.device)
            model.cuda(args.device)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.device],
                                                              find_unused_parameters=False)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    elif args.device is not None:
        torch.cuda.set_device(args.device)
        model = model.cuda(args.device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    return model


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids



class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def tensor_to_float(x):
    x_value = x if type(x) == float else x.item()
    return x_value


def load_pretrained_model(model, pretrained_model_path, model_key_in_ckpt=None, logger=None):
    if os.path.isfile(pretrained_model_path):

        pretrained_model = torch.load(pretrained_model_path)
        if model_key_in_ckpt:
            pretrained_model = pretrained_model[model_key_in_ckpt]

        if 'checkpoint' in pretrained_model_path:
            unfit_keys = model.load_state_dict(pretrained_model["model"], strict=False)
        else:
            # modify key name in checkpoint
            modified_pretrained_model = OrderedDict()
            for k, v in pretrained_model.items():
                if k.startswith("module."):
                    modified_pretrained_model[k[7:]] = v
                # if not k.startswith("fc_") and not k.startswith("backbone."):
                elif not k.startswith(("fc_", "backbone.")) and '.' in k:
                    modified_pretrained_model[".".join(("backbone", k))] = v
                else:
                    modified_pretrained_model[k] = v
            unfit_keys = model.load_state_dict(modified_pretrained_model, strict=False)
        logger.info("=> loading pretrained_model from '{}'".format(pretrained_model_path))
        logger.info('=> these keys in model are not in state dict: {}'.format(unfit_keys.missing_keys))
        logger.info('=> these keys in state dict are not in model: {}'.format(unfit_keys.unexpected_keys))
        logger.info("=> loading done!")
    else:
        logger.info("=> no pretrained_model found at '{}'".format(pretrained_model_path))


def resume_checkpoint(model, optimizer, grad_scaler, args, logger):
    ckpt_path = args.resume
    if os.path.isfile(ckpt_path):
        logger.info("=> resume checkpoint '{}'".format(ckpt_path))
        if args.device is None:
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.device)
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['grad_scaler']:
            grad_scaler.load_state_dict(checkpoint['grad_scaler'])
        logger.info("=> successfully resuming checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        return best_acc1
    else:
        logger.info("=> no checkpoint found at '{}'".format(ckpt_path))
        return 0.


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0.
                    if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)