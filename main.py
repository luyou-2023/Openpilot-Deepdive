import os
import sys
import time
import random
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# 根据 PyTorch 版本选择导入不同的 SummaryWriter
if torch.__version__ == 'parrots':
    from pavi import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

# 导入自定义的数据集和模型
from data import PlanningDataset, SequencePlanningDataset, Comma2k19SequenceDataset
from model import PlaningNetwork, MultipleTrajectoryPredictionLoss, SequencePlanningNetwork
from utils import draw_trajectory_on_ax, get_val_metric, get_val_metric_keys


# 获取命令行参数并设置默认值
def get_hyperparameters(parser: ArgumentParser):
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)  # 学习率
    parser.add_argument('--n_workers', type=int, default=4)  # 数据加载工作线程数
    parser.add_argument('--epochs', type=int, default=100)  # 训练的总轮次
    parser.add_argument('--log_per_n_step', type=int, default=20)  # 每多少步记录一次日志
    parser.add_argument('--val_per_n_epoch', type=int, default=1)  # 每多少轮进行一次验证

    parser.add_argument('--resume', type=str, default='')  # 是否从某个检查点恢复训练

    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--num_pts', type=int, default=33)
    parser.add_argument('--mtp_alpha', type=float, default=1.0)  # 用于损失计算的权重
    parser.add_argument('--optimizer', type=str, default='sgd')  # 优化器类型
    parser.add_argument('--sync_bn', type=bool, default=True)  # 是否使用同步批量归一化
    parser.add_argument('--tqdm', type=bool, default=False)  # 是否显示训练进度条
    parser.add_argument('--optimize_per_n_step', type=int, default=40)  # 每多少步进行一次优化

    # 获取环境变量中的实验名称，如果没有则使用当前时间戳
    try:
        exp_name = os.environ["SLURM_JOB_ID"]
    except KeyError:
        exp_name = str(time.time())
    parser.add_argument('--exp_name', type=str, default=exp_name)

    return parser


# 初始化分布式训练环境
def setup(rank, world_size):
    torch.cuda.set_device(rank)  # 设置当前GPU
    dist.init_process_group('nccl', init_method='tcp://localhost:%s' % os.environ['PORT'], rank=rank,
                            world_size=world_size)
    print('[%.2f]' % time.time(), 'DDP Initialized at %s:%s' % ('localhost', os.environ['PORT']), rank, 'of',
          world_size, flush=True)


# 获取训练和验证数据加载器
def get_dataloader(rank, world_size, batch_size, pin_memory=False, num_workers=0):
    train = Comma2k19SequenceDataset('data/comma2k19_train_non_overlap.txt', 'data/comma2k19/', 'train',
                                     use_memcache=False)
    val = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt', 'data/comma2k19/', 'demo', use_memcache=False)

    # 设置分布式训练的采样器
    dist_sampler_params = dict(num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_sampler = DistributedSampler(train, **dist_sampler_params)
    val_sampler = DistributedSampler(val, **dist_sampler_params)

    loader_args = dict(num_workers=num_workers, persistent_workers=True if num_workers > 0 else False,
                       prefetch_factor=2, pin_memory=pin_memory)
    train_loader = DataLoader(train, batch_size, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val, batch_size=1, sampler=val_sampler, **loader_args)

    return train_loader, val_loader


# 清理分布式环境
def cleanup():
    dist.destroy_process_group()


# 定义 SequenceBaselineV1 模型
class SequenceBaselineV1(nn.Module):
    def __init__(self, M, num_pts, mtp_alpha, lr, optimizer, optimize_per_n_step=40) -> None:
        super().__init__()
        self.M = M
        self.num_pts = num_pts
        self.mtp_alpha = mtp_alpha
        self.lr = lr
        self.optimizer = optimizer

        # 创建 SequencePlanningNetwork 模型
        self.net = SequencePlanningNetwork(M, num_pts)

        self.optimize_per_n_step = optimize_per_n_step  # 控制 GRU 模块的优化频率

    @staticmethod
    def configure_optimizers(args, model):
        # 根据参数选择优化器
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError
        # 学习率调度器
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.9)

        return optimizer, lr_scheduler

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros((2, x.size(0), 512)).to(self.device)
        return self.net(x, hidden)


def main(rank, world_size, args):
    # 如果是 rank 0（主节点），则创建 TensorBoard 的 SummaryWriter 用于记录训练过程中的日志
    if rank == 0:
        writer = SummaryWriter()

    # 获取训练和验证数据加载器
    train_dataloader, val_dataloader = get_dataloader(rank, world_size, args.batch_size, False, args.n_workers)

    # 创建模型实例
    model = SequenceBaselineV1(args.M, args.num_pts, args.mtp_alpha, args.lr, args.optimizer, args.optimize_per_n_step)

    # 是否使用同步批归一化（SyncBN）
    use_sync_bn = args.sync_bn
    if use_sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 将模型移动到 GPU
    model = model.cuda()

    # 配置优化器和学习率调度器
    optimizer, lr_scheduler = model.configure_optimizers(args, model)

    # 如果提供了恢复的模型路径，则加载模型的权重
    model: SequenceBaselineV1
    if args.resume and rank == 0:
        print('Loading weights from', args.resume)
        model.load_state_dict(torch.load(args.resume), strict=True)

    # 同步所有进程，确保模型加载完成
    dist.barrier()

    # 使用 DistributedDataParallel 对模型进行封装
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True,
                                                broadcast_buffers=False)

    # 定义损失函数
    loss = MultipleTrajectoryPredictionLoss(args.mtp_alpha, args.M, args.num_pts, distance_type='angle')

    num_steps = 0
    # 禁用或启用 tqdm 进度条
    disable_tqdm = (not args.tqdm) or (rank != 0)

    # 训练阶段
    for epoch in tqdm(range(args.epochs), disable=disable_tqdm, position=0):
        # 设置训练数据的 epoch，用于不同进程的数据分配
        train_dataloader.sampler.set_epoch(epoch)

        # 遍历训练数据
        for batch_idx, data in enumerate(tqdm(train_dataloader, leave=False, disable=disable_tqdm, position=1)):
            # 获取输入图像和标签
            seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()
            bs = seq_labels.size(0)  # batch size
            seq_length = seq_labels.size(1)  # 序列长度

            # 初始化隐藏状态
            hidden = torch.zeros((2, bs, 512)).cuda()
            total_loss = 0

            # 遍历序列中的每个时间步
            for t in tqdm(range(seq_length), leave=False, disable=disable_tqdm, position=2):
                num_steps += 1
                # 获取当前时间步的输入和标签
                inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]

                # 模型进行前向传播，获取预测结果
                pred_cls, pred_trajectory, hidden = model(inputs, hidden)

                # 计算分类损失和回归损失
                cls_loss, reg_loss = loss(pred_cls, pred_trajectory, labels)
                total_loss += (cls_loss + args.mtp_alpha * reg_loss.mean()) / model.module.optimize_per_n_step

                # 每经过一定步数记录日志
                if rank == 0 and (num_steps + 1) % args.log_per_n_step == 0:
                    writer.add_scalar('train/epoch', epoch, num_steps)
                    writer.add_scalar('loss/cls', cls_loss, num_steps)
                    writer.add_scalar('loss/reg', reg_loss.mean(), num_steps)
                    writer.add_scalar('loss/reg_x', reg_loss[0], num_steps)
                    writer.add_scalar('loss/reg_y', reg_loss[1], num_steps)
                    writer.add_scalar('loss/reg_z', reg_loss[2], num_steps)
                    writer.add_scalar('param/lr', optimizer.param_groups[0]['lr'], num_steps)

                # 每经过一定的步骤进行一次优化
                if (t + 1) % model.module.optimize_per_n_step == 0:
                    hidden = hidden.clone().detach()  # 防止梯度回传到之前的隐藏状态
                    optimizer.zero_grad()  # 清空梯度
                    total_loss.backward()  # 反向传播计算梯度
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
                    optimizer.step()  # 更新参数
                    if rank == 0:
                        writer.add_scalar('loss/total', total_loss, num_steps)
                    total_loss = 0  # 重置损失

            # 如果总损失不是整数，继续进行优化
            if not isinstance(total_loss, int):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                optimizer.step()
                if rank == 0:
                    writer.add_scalar('loss/total', total_loss, num_steps)

        # 更新学习率
        lr_scheduler.step()

        # 每经过一定的周期进行验证
        if (epoch + 1) % args.val_per_n_epoch == 0:
            if rank == 0:
                # 保存模型检查点
                ckpt_path = os.path.join(writer.log_dir, 'epoch_%d.pth' % epoch)
                torch.save(model.module.state_dict(), ckpt_path)
                print('[Epoch %d] checkpoint saved at %s' % (epoch, ckpt_path))

            # 切换到评估模式
            model.eval()
            with torch.no_grad():
                # 初始化度量结果
                saved_metric_epoch = get_val_metric_keys()
                # 遍历验证数据
                for batch_idx, data in enumerate(tqdm(val_dataloader, leave=False, disable=disable_tqdm, position=1)):
                    seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()

                    bs = seq_labels.size(0)
                    seq_length = seq_labels.size(1)

                    # 初始化隐藏状态
                    hidden = torch.zeros((2, bs, 512), device=seq_inputs.device)
                    for t in tqdm(range(seq_length), leave=False, disable=True, position=2):
                        inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
                        pred_cls, pred_trajectory, hidden = model(inputs, hidden)

                        # 计算验证指标
                        metrics = get_val_metric(pred_cls, pred_trajectory.view(-1, args.M, args.num_pts, 3), labels)

                        # 记录每个指标的值
                        for k, v in metrics.items():
                            saved_metric_epoch[k].append(v.float().mean().item())

                dist.barrier()  # 等待所有进程
                # 同步计算所有进程的结果
                metric_single = torch.zeros((len(saved_metric_epoch),), dtype=torch.float32, device='cuda')
                counter_single = torch.zeros((len(saved_metric_epoch),), dtype=torch.int32, device='cuda')
                # 从字典中按照键的顺序取值
                for i, k in enumerate(sorted(saved_metric_epoch.keys())):
                    metric_single[i] = np.mean(saved_metric_epoch[k])
                    counter_single[i] = len(saved_metric_epoch[k])

                # 汇总各个进程的数据
                metric_gather = [torch.zeros((len(saved_metric_epoch),), dtype=torch.float32, device='cuda')[None] for _
                                 in range(world_size)]
                counter_gather = [torch.zeros((len(saved_metric_epoch),), dtype=torch.int32, device='cuda')[None] for _
                                  in range(world_size)]
                dist.all_gather(metric_gather, metric_single[None])
                dist.all_gather(counter_gather, counter_single[None])

                # 在主进程计算加权平均指标
                if rank == 0:
                    metric_gather = torch.cat(metric_gather, dim=0)  # [world_size, num_metric_keys]
                    counter_gather = torch.cat(counter_gather, dim=0)  # [world_size, num_metric_keys]
                    metric_gather_weighted_mean = (metric_gather * counter_gather).sum(0) / counter_gather.sum(0)
                    # 将结果记录到 TensorBoard
                    for i, k in enumerate(sorted(saved_metric_epoch.keys())):
                        writer.add_scalar(k, metric_gather_weighted_mean[i], num_steps)
                dist.barrier()

            # 切换回训练模式
            model.train()

    # 清理工作
    cleanup()


if __name__ == "__main__":
    print('[%.2f]' % time.time(), 'starting job...', os.environ['SLURM_PROCID'], 'of', os.environ['SLURM_NTASKS'],
          flush=True)

    # 解析命令行参数
    parser = ArgumentParser()
    parser = get_hyperparameters(parser)
    args = parser.parse_args()

    # 初始化分布式训练环境
    setup(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']))

    # 启动训练主函数
    main(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']), args=args)
