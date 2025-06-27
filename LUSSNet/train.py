import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.LUSS import LUSS
from nets.LUSSNet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    parser.add_argument('--Cuda', default=True,type=bool, help='Cuda')
    parser.add_argument('--seed', default=1,type=int, help='seed')
    parser.add_argument('--distributed', default=False, type=bool, help='distributed training')
    parser.add_argument('--sync_bn', default=False, type=bool,help='sync_bn')
    parser.add_argument('--fp16', default=False, type=bool, help='fp16')
    parser.add_argument('--num_classes', default=8, type=int, help='num_classes')
    parser.add_argument('--pretrained', type=bool, default=False, help='pretrained')
    parser.add_argument('--input_shape', type=int, default=[640,640],help='base image size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=10, help='input batch size for training (default: 8)')
    parser.add_argument('--Init_lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--save_period', type=int, default=100, help='save_period')
    parser.add_argument('--VOCdevkit_path', type=str, default= 'SUIM',help='data path')
    parser.add_argument('--save_dir', type=str, default='logs/', help='log')
    parser.add_argument('--eval_flag', default=False, type=bool, help='eval')
    parser.add_argument('--eval_period', type=int, default=5, help='eval_period')
    parser.add_argument('--WDD_loss', default=True, type=bool, help='WDD_loss')
    parser.add_argument('--focal_loss', default=True, type=bool, help='focal_loss')
    parser.add_argument('--num_workers',type=int, default=4,help='dataloader threads')
    parser.add_argument('--optimizer_type', type=str, default='adam', help='sgd and adam')
    parser.add_argument('--lr_decay_type', type=str, default='cos', help='lr_decay_type')
    parser.add_argument('--weight_decay', type=int, default=0, help='weight_decay')
    args=parser.parse_args()
    return args





if __name__ == "__main__":
    args = parse_args()
    Min_lr =args.Init_lr * 0.01
    cls_weights = np.ones([args.num_classes], np.float32)
    seed_everything(args.seed)
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0
    model = LUSS(num_classes=args.num_classes).train()
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(args.save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=args.input_shape)
    else:
        loss_history = None
    if args.fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if args.sync_bn and ngpus_per_node > 1 and args.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif args.sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if args.Cuda:
        if args.distributed:

            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    with open(os.path.join(args.VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=args.num_classes,  input_shape=args.input_shape,  UnFreeze_Epoch=args.epochs,
            Unfreeze_batch_size=args.batch_size, \
            Init_lr=args.Init_lr, Min_lr=Min_lr, optimizer_type=args.optimizer_type, momentum=args.momentum,
            lr_decay_type=args.lr_decay_type, \
            save_period=args.save_period, save_dir=args.save_dir, num_workers=args.num_workers, num_train=num_train, num_val=num_val
        )

    if True:
        UnFreeze_flag = False
        batch_size = args.batch_size

        nbs = 16
        lr_limit_max = 1e-4 if args.optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if args.optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * args.Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(args.momentum, 0.999), weight_decay=args.weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=args.momentum, nesterov=True,
                             weight_decay=args.weight_decay)
        }[args.optimizer_type]

        lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.epochs)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("dataset small")

        train_dataset = UnetDataset(train_lines, args.input_shape, args.num_classes, True, args.VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, args.input_shape, args.num_classes, False, args.VOCdevkit_path)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=args.num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))

        if local_rank == 0:
            eval_callback = EvalCallback(model, args.input_shape, args.num_classes, val_lines, args.VOCdevkit_path, log_dir, args.Cuda, \
                                         eval_flag=args.eval_flag, period=args.eval_period)

        else:
            eval_callback = None

        for epoch in range(0, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, args.epochs, args.Cuda, args.WDD_loss, args.focal_loss,
                          cls_weights, args.num_classes, args.fp16, scaler, args.save_period, args.save_dir, log_dir,local_rank,eval_period=args.eval_period)

            if args.distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
