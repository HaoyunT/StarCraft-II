"""
训练参数配置模块

负责解析命令行参数并根据硬件环境自动优化配置
"""

import argparse
import torch

def get_common_args():
    """获取基础训练参数"""
    parser = argparse.ArgumentParser(description='MAPPO训练参数配置')
    
    # 环境设置
    parser.add_argument('--map', type=str, default='3m', help='SMAC地图名称')
    parser.add_argument('--difficulty', type=str, default='4', help='游戏难度等级 (1-9, 推荐3-5)')
    parser.add_argument('--game_version', type=str, default='latest', help='星际争霸2版本')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    parser.add_argument('--step_mul', type=int, default=8, help='游戏步数倍率')
    parser.add_argument('--replay_dir', type=str, default='', help='回放文件目录')

    # 算法配置
    parser.add_argument('--alg', type=str, default='mappo', choices=['mappo', 'ippo'], 
                       help='算法类型: mappo(集中式价值函数) 或 ippo(独立式价值函数)')
    parser.add_argument('--last_action', action='store_true', help='使用历史动作信息')
    parser.add_argument('--no-last-action', dest='last_action', action='store_false')
    parser.set_defaults(last_action=True)
    parser.add_argument('--reuse_network', action='store_true', help='智能体网络参数共享')
    parser.add_argument('--no-reuse_network', dest='reuse_network', action='store_false')
    parser.set_defaults(reuse_network=True)

    # 训练设置
    parser.add_argument('--n_steps', type=int, default=15000, help='总训练步数')
    parser.add_argument('--evaluate_cycle', type=int, default=100, help='评估间隔步数（每100步输出一次统计）')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='评估回合数（20次实验平均成功率）')
    parser.add_argument('--episode_limit', type=int, default=None, help='单回合最大步数')
    
    # 路径设置
    parser.add_argument('--model_dir', type=str, default='./models', help='模型保存路径')
    parser.add_argument('--result_dir', type=str, default='./results', help='结果保存路径')
    parser.add_argument('--learn', action='store_true', help='是否执行训练')
    parser.add_argument('--no-learn', dest='learn', action='store_false')
    parser.set_defaults(learn=True)
    
    # 回放设置
    parser.add_argument('--save_replay', action='store_true', help='保存游戏回放')
    parser.add_argument('--no-save_replay', dest='save_replay', action='store_false')
    parser.set_defaults(save_replay=False)
    
    # 加载模型
    parser.add_argument('--load_model', action='store_true', help='加载已保存的模型')
    parser.add_argument('--no-load_model', dest='load_model', action='store_false')
    parser.set_defaults(load_model=False)

    return parser.parse_args()


def get_mixer_args(args):
    """根据硬件环境自动配置训练参数"""
    
    # GPU检测与配置
    args.use_gpu = torch.cuda.is_available()
    
    if args.use_gpu:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # 大显存配置（云服务器）
        if gpu_memory_gb > 20:
            args.rnn_hidden_dim = 256
            args.n_episodes = 32
            args.ppo_n_epochs = 4
            args.lr = 1e-3
        # 小显存配置（本地GPU）
        else:
            args.rnn_hidden_dim = 128
            args.n_episodes = 24
            args.ppo_n_epochs = 4
            args.lr = 1e-3
    # CPU配置
    else:
        args.rnn_hidden_dim = 128
        args.n_episodes = 16
        args.ppo_n_epochs = 3
        args.lr = 1e-3

    # 算法超参数
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
    
    # 网络层配置
    args.actor_hidden_size = 64
    args.critic_hidden_size = 64
    
    # MAPPO/IPPO 特定设置
    if args.alg == 'ippo':
        # IPPO使用独立价值函数，不需要全局状态
        args.share_policy = True
        args.use_centralized_V = False
        print("使用IPPO算法 - 独立智能体训练")
    else:  # mappo
        # MAPPO使用集中式价值函数
        args.share_policy = True
        args.use_centralized_V = True
        print("使用MAPPO算法 - 集中式价值函数")
    
    return args