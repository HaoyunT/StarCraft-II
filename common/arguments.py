"""
训练参数配置模块
"""

import argparse
import torch

def get_common_args():
    """获取基础训练参数"""
    parser = argparse.ArgumentParser(description='MAPPO训练参数配置')
    
    # 环境设置
    parser.add_argument('--map', type=str, default='3m', help='SMAC地图名称')
    parser.add_argument('--difficulty', type=str, default='4', help='游戏难度等级（降低到最简单）')
    parser.add_argument('--game_version', type=str, default='latest', help='星际争霸2版本')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    parser.add_argument('--step_mul', type=int, default=8, help='游戏步数倍率')
    parser.add_argument('--replay_dir', type=str, default='', help='回放文件目录')
    parser.add_argument('--save_replay', action='store_true', help='评估时是否保存回放(replay)')
    parser.set_defaults(save_replay=False)
    # 是否使用集中式价值函数（MAPPO通常使用集中式V）
    parser.add_argument('--use_centralized_V', action='store_true', help='使用集中式价值函数（MAPPO）')
    parser.add_argument('--no-use_centralized_V', dest='use_centralized_V', action='store_false',
                        help='不使用集中式价值函数（使用独立式价值函数）')
    parser.set_defaults(use_centralized_V=True)

    # 算法配置
    parser.add_argument('--alg', type=str, default='mappo', help='算法类型')
    parser.add_argument('--last_action', action='store_true', help='使用历史动作信息')
    parser.add_argument('--no-last_action', dest='last_action', action='store_false', help='不使用历史动作信息')
    parser.set_defaults(last_action=True)
    parser.add_argument('--reuse_network', action='store_true', help='智能体网络参数共享')
    parser.add_argument('--no-reuse_network', dest='reuse_network', action='store_false', help='不共享网络')
    parser.set_defaults(reuse_network=True)

    # 训练设置
    parser.add_argument('--n_steps', type=int, default=50000, help='总训练步数（增加训练时间）')
    parser.add_argument('--evaluate_cycle', type=int, default=100, help='评估间隔步数')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='评估回合数')
    parser.add_argument('--episode_limit', type=int, default=None, help='单回合最大步数')
    
    # 路径设置
    parser.add_argument('--model_dir', type=str, default='./models', help='模型保存路径')
    parser.add_argument('--result_dir', type=str, default='./results', help='结果保存路径')
    parser.add_argument('--learn', action='store_true', help='是否执行训练')
    parser.add_argument('--no-learn', dest='learn', action='store_false', help='只评估，不训练')
    parser.set_defaults(learn=True)

    return parser.parse_args()


def get_mixer_args(args):
    """配置训练参数"""
    
    # GPU检测与配置
    args.use_gpu = torch.cuda.is_available()
    
    if args.use_gpu:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb > 20:
            args.rnn_hidden_dim = 256
            args.n_episodes = 32
        else:
            args.rnn_hidden_dim = 128
            args.n_episodes = 24
    else:
        args.rnn_hidden_dim = 128
        args.n_episodes = 16

    # PPO参数 - 优化后的超参数
    args.ppo_n_epochs = 5  # 增加训练轮数
    args.lr = 5e-4  # 适中的学习率
    args.gamma = 0.99  # 折扣因子
    args.lamda = 0.95  # GAE参数
    args.clip_param = 0.2  # PPO裁剪参数
    args.entropy_coeff = 0.01  # 熵系数 - 起始值
    args.entropy_min = 0.001  # 熵系数最小值
    args.value_loss_coef = 1.0  # 价值函数损失权重 - 增加以更好学习价值
    args.grad_norm_clip = 10.0  # 梯度裁剪

    # KL散度自适应参数
    args.kl_target = 0.01  # 目标KL散度
    args.kl_stop_multiplier = 1.5  # KL过大时停止训练的阈值
    args.kl_adapt = True  # 启用KL自适应

    # 优化器设置
    args.optimizer = "Adam"
    args.lr_actor = args.lr * 1.0  # Actor学习率
    args.lr_critic = args.lr * 1.0  # Critic学习率 - 保持一致

    # 保存设置
    args.save_cycle = 5000
    args.train_steps = 1
    
    return args