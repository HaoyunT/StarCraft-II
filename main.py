"""
Multi-Agent PPO on SMAC

基于SMAC环境的多智能体强化学习训练。
支持MAPPO和IPPO算法，适配本地和云服务器环境。

使用方法:
    python main.py --map=3m --alg=mappo  # 使用MAPPO算法
    python main.py --map=3m --alg=ippo   # 使用IPPO算法
"""

from runner import Runner
from smac.env import StarCraft2Env  # type: ignore
from common.arguments import get_mixer_args, get_common_args
import torch
import os


if __name__ == '__main__':
    # ========== 环境自适配配置 ==========
    # 自动检测本地/远程环境
    import sys

    # 基础环境配置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['SDL_VIDEODRIVER'] = 'dummy'  # 无头模式，节省资源
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['SC2_VERBOSE'] = '0'

    # StarCraft II 路径自适配
    sc2_paths = [
        os.path.expanduser("~/StarCraft_II"),           # 远程服务器路径
        "/root/StarCraft_II",                           # root用户路径
        "G:\\StarCraft II",                             # 本地Windows路径
        os.path.join(os.path.expanduser("~"), "StarCraft_II"),  # 用户主目录
    ]

    sc2_path = None
    for path in sc2_paths:
        if os.path.exists(path):
            sc2_path = path
            break

    if sc2_path:
        os.environ["SC2PATH"] = sc2_path
        is_remote = "/root/" in sc2_path or "~" in sc2_path
    else:
        # 如果找不到，使用默认本地路径（开发时）
        os.environ["SC2PATH"] = "G:\\StarCraft II"
        is_remote = False
        print("⚠️  警告: StarCraft II未找到，使用默认路径")

    # 离线模式配置
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # 本地GPU显存较小，启用离线模式加快启动
        if gpu_memory <= 20:
            os.environ['SC2_OFFLINE'] = '1'

    print("=" * 50)
    print("🚀 Multi-Agent PPO 星际争霸2 智能体训练")
    print("=" * 50)
    
    # 显示环境信息
    print(f"运行环境:")
    print(f"  Python: {sys.executable}")
    print(f"  工作目录: {os.getcwd()}")
    print(f"  SC2路径: {os.environ.get('SC2PATH', '未配置')}")
    print(f"  运行环境: {'远程服务器' if is_remote else '本地'}")

    # 加载配置参数
    args = get_common_args()
    args = get_mixer_args(args)
    
    # 显示训练配置
    print(f"训练配置:")
    print(f"  地图: {args.map}")
    print(f"  算法: {args.alg.upper()}")
    print(f"  训练步数: {args.n_steps}")
    print(f"  评估间隔: {args.evaluate_cycle}")
    print(f"  网络共享: {'是' if args.reuse_network else '否'}")
    print(f"  历史动作: {'是' if args.last_action else '否'}")
    
    if args.use_gpu and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        mode = "云服务器" if gpu_memory > 20 else "本地"
        print(f"  模式: {mode}环境优化")
    else:
        print(f"  设备: CPU")
    print("-" * 50)
    
    try:
        # 初始化SMAC环境
        print("初始化SMAC环境...")
        
        # 设置回放目录
        replay_dir_path = ""
        if args.save_replay:
            replay_dir_path = args.replay_dir if args.replay_dir else f"./replays/{args.map}_{args.alg}"
            os.makedirs(replay_dir_path, exist_ok=True)
            print(f"回放将保存到: {replay_dir_path}")
        
        env = StarCraft2Env(map_name=args.map,
                           step_mul=args.step_mul,
                           difficulty=args.difficulty,
                           game_version=args.game_version,
                           replay_dir=replay_dir_path,
                           debug=False)

        # 获取环境参数
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        
        if args.episode_limit is None:
            args.episode_limit = env_info["episode_limit"]

        print(f"环境信息:")
        print(f"  智能体数: {args.n_agents} | 动作数: {args.n_actions}")
        print(f"  状态维度: {args.state_shape} | 观察维度: {args.obs_shape}")
        print(f"  回合长度: {args.episode_limit}")
        print("-" * 50)
        
        # 开始训练
        runner = Runner(env, args)
        
        if args.learn:
            runner.run()

            # 训练总结
            if len(runner.win_rates) > 0:
                print(f"\n 训练结果:")
                print(f"  胜率: {runner.win_rates[0]:.1%} → {runner.win_rates[-1]:.1%} (提升{runner.win_rates[-1] - runner.win_rates[0]:+.1%})")
                print(f"  奖励: {runner.episode_rewards[0]:.1f} → {runner.episode_rewards[-1]:.1f}")
                print(f"  训练步数: {runner.train_steps_record[-1] if len(runner.train_steps_record) > 0 else 0}")
        else:
            win_rate, avg_reward = runner.evaluate()
            print(f'评估结果 - 胜率: {win_rate:.2%}, 平均奖励: {avg_reward:.2f}')
        
        env.close()
        
        print(f"\n✅ 结果已保存至: {runner.save_path}")
        print("   包含训练曲线图和详细数据")

    except KeyboardInterrupt:
        print("\n⏹️ 训练已中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {str(e)}")
        raise
    finally:
        if 'env' in locals():
            env.close()
