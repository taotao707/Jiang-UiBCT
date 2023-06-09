import argparse
import os
from datetime import timedelta

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

import data_loader.data_loaders as module_data
from model.build import build_model, build_lr_scheduler
from trainer import LandmarkTrainer, FaceTrainer
from logger.logger import create_logger
from utils.util import parser_option, cudalize, set_random_seed, resume_checkpoint, ArcFace


def main(config):
    args = argparse.Namespace(**config.config)

    # fix random seeds for reproducibility，固定随机种子训练
    if args.seed is not None:
        set_random_seed(args.seed)

    #使用环境变量初始化分布式训练，即进程同步
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    #world_size单机多卡训练时为1，应当为主机数量，ngpus_per_node代表进程数量
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    config._update_config_by_dict({"distributed": args.distributed})
    ngpus_per_node = torch.cuda.device_count()

    #分布式训练world_size乘后为总进程数，每卡一个
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        main_worker(args.device, ngpus_per_node, config)


def main_worker(device, ngpus_per_node, args, config):
    #device为GPU号，标识进行所在GPU
    args.device = device
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes，rank单机训练时为0，加过gpu后等于gpu号
            args.rank = args.rank * ngpus_per_node + device
            config._update_config_by_dict({"rank": args.rank})
            config._update_config_by_dict({"device": args.device})
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank, timeout=timedelta(seconds=45))
        # dist.barrier()
    cudnn.benchmark = True

    logger = create_logger(output_dir=config.log_dir, dist_rank=args.rank, name='train')

    #处理MS1M-v3和IJB-C数据集
    logger.info("Setup dataset...")
    if args.dataset["type"] == 'landmark':
        # load training set
        data_loader = module_data.GLDv2TrainDataset(args)
        train_loader, class_num, args.dataset["class_num"] = data_loader.train_loader, data_loader.class_num, data_loader.class_num

        # load evaluation set
        data_loader = module_data.GLDv2EvalDataset(args)
        eval_query_loader, eval_gallery_loader, eval_query_gts = data_loader.query_loader, \
                                                                 data_loader.gallery_loader,\
                                                                 data_loader.query_gts
        # load test set
        data_loader = module_data.GLDv2TestDataset(args)
        test_query_loader, test_gallery_loader, test_query_gts = data_loader.query_loader, \
                                                                 data_loader.gallery_loader,\
                                                                 data_loader.query_gts

    elif args.dataset["type"] == 'face':
        #load training set
        data_loader = module_data.MS1Mv3TrainDataLoader(args)
        train_loader, class_num, args.dataset["class_num"] = data_loader.train_loader, data_loader.class_num, data_loader.class_num

        #load evaluation set 
        data_loader = module_data.IJBCEvalDataLoader(args)
        eval_query_loader, eval_gallery_loader, eval_query_gts = data_loader.query_loader, \
                                                                 data_loader.gallery_loader,\
                                                                 data_loader.query_gts

        # load test set
        data_loader = module_data.IJBCTestDataLoader(args)
        test_query_loader, test_gallery_loader, test_query_gts = data_loader.query_loader, \
                                                                 data_loader.gallery_loader,\
                                                                 data_loader.query_gts
        
    else:
        raise NotImplementedError


    # build model architecture
    model = build_model(args, logger)

    # build solver
    optimizer = torch.optim.SGD(model.parameters(), lr=args.optimizer["lr"],
                                momentum=args.optimizer["momentum"],
                                weight_decay=args.optimizer["weight_decay"])
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_acc1 = 0.

    if args.new_model["resume"] is not None:
        best_acc1 = resume_checkpoint(model, optimizer, grad_scaler, args, logger)
    config._update_config_by_dict({"best_acc1": best_acc1})

    model = cudalize(model, args)

    #学习率调整策略
    steps_per_epoch = len(train_loader)
    lr_scheduler = build_lr_scheduler(args, optimizer, steps_per_epoch, sche_type=args.lr_scheduler["type"])

    # build loss
    criterion = {}
    criterion['base'] = nn.CrossEntropyLoss().cuda()
    if args.dataset["type"] == 'landmark':
        validation_loader_list = [eval_query_loader, eval_gallery_loader, eval_query_gts]
        test_loader_list = [test_query_loader, test_gallery_loader, test_query_gts]
        trainer = LandmarkTrainer(model,
                                  comp_training=None,
                                  train_loader=train_loader,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  grad_scaler=grad_scaler,
                                  args=args,
                                  config=config,
                                  logger=logger,
                                  validation_loader_list=validation_loader_list,
                                  test_loader_list=test_loader_list,
                                  lr_scheduler=lr_scheduler)
    elif args.dataset["type"] == 'face':
        validation_loader_list = [eval_query_loader, eval_gallery_loader, eval_query_gts]
        test_loader_list = [test_query_loader, test_gallery_loader, test_query_gts]
        trainer = FaceTrainer(model,
                              comp_training=None,
                              train_loader=train_loader,
                              criterion=criterion,
                              optimizer=optimizer,
                              grad_scaler=grad_scaler,
                              args=args,
                              config=config,
                              logger=logger,
                              validation_loader_list=validation_loader_list,
                              test_loader_list=test_loader_list,
                              lr_scheduler=lr_scheduler)
    else:
        raise NotImplementedError

    trainer.train()


if __name__ == '__main__':
    config = parser_option()
    main(config)