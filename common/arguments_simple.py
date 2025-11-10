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
    parser.add_argument('--difficulty', type=str, default='4', help='游戏难度等级')
    parser.add_argument('--game_version', type=str, default='latest', help='星际争霸2版本')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    parser.add_argument('--step_mul', type=int, default=8, help='游戏步数倍率')
    parser.add_argument('--replay_dir', type=str, default='', help='回放文件目录')

    # 算法配置
    parser.add_argument('--alg', type=str, default='mappo', help='算法类型')
    parser.add_argument('--last_action', action='store_true', help='使用历史动作信息')
    parser.set_defaults(last_action=True)
    parser.add_argument('--reuse_network', action='store_true', help='智能体网络参数共享')
    parser.set_defaults(reuse_network=True)

    # 训练设置
    parser.add_argument('--n_steps', type=int, default=15000, help='总训练步数')
    parser.add_argument('--evaluate_cycle', type=int, default=100, help='评估间隔步数')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='评估回合数')
    parser.add_argument('--episode_limit', type=int, default=None, help='单回合最大步数')
    
    # 路径设置
    parser.add_argument('--model_dir', type=str, default='./models', help='模型保存路径')
    parser.add_argument('--result_dir', type=str, default='./results', help='结果保存路径')
    parser.add_argument('--learn', action='store_true', help='是否执行训练')
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

    # PPO参数
    args.ppo_n_epochs = 4
    args.lr = 1e-3
    args.gamma = 0.99
    args.lamda = 0.95
    args.clip_param = 0.3
    args.entropy_coeff = 0.02
    args.value_loss_coef = 0.5
    args.grad_norm_clip = 10.0

    # 优化器设置
    args.optimizer = "Adam"
    args.lr_actor = args.lr
    args.lr_critic = args.lr

    # 保存设置
    args.save_cycle = 5000
    args.train_steps = 1
    
    return args