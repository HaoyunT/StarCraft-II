import numpy as np
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt# type: ignore
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
from smac.env import StarCraft2Env # type: ignore

# 导入配置
try:
    from training_config import USE_REPLAY_BUFFER
except:
    USE_REPLAY_BUFFER = False  # 默认禁用经验回放，确保稳定性


class Runner:
    """训练和评估管理器（优化版）"""

    def __init__(self, env, args):
        self.env = env
        self.args = args
        
        # 创建智能体和环境交互器
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        
        # 经验回放缓冲区（可选，提高样本效率）
        self.use_replay_buffer = USE_REPLAY_BUFFER
        if self.use_replay_buffer:
            buffer_size = max(args.n_episodes * 5, 100)
            self.replay_buffer = ReplayBuffer(args, buffer_size=buffer_size)
            print(f"启用经验回放 - Buffer大小: {buffer_size}")
        else:
            self.replay_buffer = None
            print("禁用经验回放 - 使用标准on-policy训练")

        # 探索参数（epsilon-greedy）
        self.epsilon_start = 0.2  # 起始探索率
        self.epsilon_end = 0.01  # 最终探索率
        self.epsilon_decay_steps = args.n_steps * 0.7  # 衰减步数（70%训练时间）

        # 训练记录
        self.win_rates = []
        self.episode_rewards = []
        self.train_steps_record = []

        # 结果保存路径
        self.save_path = os.path.join(args.result_dir, args.alg, args.map)
        os.makedirs(self.save_path, exist_ok=True)

        # 中文字体设置
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass  # 字体设置失败不影响功能

    def get_epsilon(self, time_steps):
        """计算当前的epsilon值（线性衰减）"""
        if time_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        else:
            progress = time_steps / self.epsilon_decay_steps
            return self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress

    def run(self, num=0):
        """主训练循环（优化版）"""
        time_steps = 0
        train_steps = 0
        last_eval_steps = 0

        mode_desc = "经验回放和自适应探索" if self.use_replay_buffer else "标准on-policy和自适应探索"
        print(f"\n开始训练（使用{mode_desc}）...")
        print(f"初始探索率: {self.epsilon_start:.3f}, 最终探索率: {self.epsilon_end:.3f}")

        while time_steps < self.args.n_steps:
            # 计算当前epsilon
            current_epsilon = self.get_epsilon(time_steps)

            # 收集训练数据
            episodes = []
            total_reward = 0
            wins = 0

            for episode_idx in range(self.args.n_episodes):
                # 使用当前的epsilon进行探索
                episode, episode_reward, victory, steps = self.rolloutWorker.generate_episode(
                    episode_idx, evaluate=False, epsilon=current_epsilon)

                # 添加到经验回放buffer（如果启用）
                if self.use_replay_buffer and self.replay_buffer is not None:
                    self.replay_buffer.push(episode)

                episodes.append(episode)
                total_reward += episode_reward
                wins += int(victory)
                time_steps += steps

            # 准备训练batch
            if self.use_replay_buffer and self.replay_buffer is not None and len(self.replay_buffer) >= self.args.n_episodes:
                # 使用经验回放：混合新旧经验
                new_batch_size = max(self.args.n_episodes // 2, 1)
                old_batch_size = self.args.n_episodes - new_batch_size

                # 合并新数据
                episode_batch = episodes[0]
                for episode in episodes[1:new_batch_size]:
                    for key in episode_batch.keys():
                        try:
                            episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
                        except ValueError as e:
                            print(f"警告: episode维度不匹配，跳过 - {e}")
                            continue

                # 从buffer采样历史数据
                if old_batch_size > 0:
                    try:
                        old_batch = self.replay_buffer.sample(old_batch_size)
                        for key in episode_batch.keys():
                            episode_batch[key] = np.concatenate((episode_batch[key], old_batch[key]), axis=0)
                    except Exception as e:
                        print(f"警告: 从buffer采样失败，仅使用新数据 - {e}")
            else:
                # 标准训练：只用新数据
                episode_batch = episodes[0]
                for episode in episodes[1:]:
                    for key in episode_batch.keys():
                        try:
                            episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
                        except ValueError as e:
                            print(f"警告: episode维度不匹配，跳过 - {e}")
                            continue

            # 执行训练
            self.agents.train(episode_batch, train_steps, time_steps)
            train_steps += 1

            # 检查是否需要评估（每evaluate_cycle步进行一次）
            if time_steps - last_eval_steps >= self.args.evaluate_cycle:
                last_eval_steps = time_steps

                # 收集当前评估数据
                win_rate, episode_reward = self.evaluate()

                # 添加到统计
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.train_steps_record.append(time_steps)

                # 简洁输出评估结果
                avg_train_reward = total_reward / self.args.n_episodes
                train_win_rate = wins / self.args.n_episodes
                print(f"步数: {time_steps:6d}/{self.args.n_steps} | 训练: 胜率={train_win_rate*100:5.1f}% 奖励={avg_train_reward:6.1f} | "
                      f"评估: 胜率={win_rate*100:5.1f}% 奖励={episode_reward:6.1f} | ε={current_epsilon:.3f}")

                # 更新图表（降低频率以节省内存）
                if len(self.win_rates) % 2 == 0:
                    try:
                        self.plt(num)
                    except Exception as e:
                        print(f"  警告: 保存图表失败 - {e}")

                    # 定期清理内存
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        print("\n训练完成！")


    def evaluate(self):
        """评估当前策略"""
        win_number = 0
        episode_rewards = 0
        
        for epoch in range(self.args.evaluate_epoch):
            try:
                _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
                episode_rewards += episode_reward
                if win_tag:
                    win_number += 1
            except ZeroDivisionError:
                # 处理初始化时battles_game=0的情况
                print(f"警告: 评估第{epoch+1}回合时遇到统计错误，使用默认值")
                episode_rewards += 0
            except Exception as e:
                print(f"警告: 评估第{epoch+1}回合时出错: {e}")
                episode_rewards += 0

        win_rate = win_number / self.args.evaluate_epoch if self.args.evaluate_epoch > 0 else 0
        avg_reward = episode_rewards / self.args.evaluate_epoch if self.args.evaluate_epoch > 0 else 0

        return win_rate, avg_reward
    
    def plt(self, num):
        """绘制并保存训练曲线"""

        if len(self.win_rates) == 0:
            return
            
        # 清理之前的图形，释放内存
        plt.close('all')

        # 准备数据
        x_all = self.train_steps_record if len(self.train_steps_record) > 0 else list(range(len(self.win_rates)))
        wr_all = self.win_rates
        er_all = self.episode_rewards

        # 仅绘制最近N个点，避免内存暴涨
        max_points = 100
        if len(x_all) > max_points:
            x_steps = x_all[-max_points:]
            wr = wr_all[-max_points:]
            er = er_all[-max_points:]
        else:
            x_steps = x_all
            wr = wr_all
            er = er_all
        
        # 创建更小的双子图，节省内存
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=100)

        # 胜率曲线（简化绘制）
        ax1.plot(x_steps, wr, 'b-', linewidth=1.2, markersize=3)
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('胜率')
        ax1.set_title('训练胜率')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # 奖励曲线（简化绘制）
        ax2.plot(x_steps, er, 'g-', linewidth=1.2, markersize=3)
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('平均奖励')
        ax2.set_title('训练奖励')
        ax2.grid(True, alpha=0.3)
        
        # 总标题和保存
        fig.suptitle(f'{self.args.map} 训练结果', fontsize=12)
        plt.tight_layout()
        
        save_file = os.path.join(self.save_path, f'training_curve_{num}.png')

        # 使用低DPI避免内存问题
        try:
            plt.savefig(save_file, dpi=100, bbox_inches='tight', facecolor='white')
        except Exception as e:
            print(f"  警告: 图表保存失败 - {e}")
        finally:
            plt.close('all')

        # 保存数据
        try:
            np.save(os.path.join(self.save_path, f'{num}_win_rates'), np.array(self.win_rates))
            np.save(os.path.join(self.save_path, f'{num}_episode_rewards'), np.array(self.episode_rewards))
        except:
            pass
