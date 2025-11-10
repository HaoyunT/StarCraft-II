"""
训练统计和监控模块

用于记录训练过程中的奖励、成功率等指标，并定期输出统计报告
"""

import numpy as np
from collections import deque


class TrainingStats:
    """训练统计和监控类"""

    def __init__(self, stat_interval=100, eval_episodes=10):
        """
        初始化训练统计器

        Args:
            stat_interval: 统计间隔步数（每多少步输出一次统计）
            eval_episodes: 评估回合数（评估时运行多少次实验取平均）
        """
        self.stat_interval = stat_interval
        self.eval_episodes = eval_episodes
        self.reward_buffer = deque(maxlen=stat_interval)
        self.success_buffer = deque(maxlen=eval_episodes)
        self.total_steps = 0
        self.episode_count = 0

    def add_reward(self, reward):
        """添加单步或单回合奖励"""
        self.reward_buffer.append(reward)
        self.total_steps += 1

    def add_episode(self):
        """增加回合计数"""
        self.episode_count += 1

    def add_success(self, is_success):
        """添加评估结果（成功/失败）"""
        self.success_buffer.append(1 if is_success else 0)

    def should_report(self):
        """判断是否应该输出统计"""
        return self.total_steps % self.stat_interval == 0 and self.total_steps > 0

    def get_report(self):
        """获取统计报告字典"""
        avg_reward = np.mean(self.reward_buffer) if self.reward_buffer else 0
        avg_success = np.mean(self.success_buffer) if self.success_buffer else 0

        report = {
            'step': self.total_steps,
            'episode': self.episode_count,
            'avg_reward': avg_reward,
            'success_rate': avg_success,
            'reward_buffer_size': len(self.reward_buffer),
            'success_count': int(sum(self.success_buffer))
        }
        return report

    def print_report(self):
        """打印统计报告到控制台"""
        report = self.get_report()
        success_display = f"{report['success_count']}/{len(self.success_buffer)}"
        print(f"[Step {report['step']:6d} | Ep {report['episode']:5d}] "
              f"Avg Reward: {report['avg_reward']:7.2f} | "
              f"Success Rate: {report['success_rate']:6.1%} ({success_display})")

    def reset_buffers(self):
        """重置缓冲区（用于新的评估周期）"""
        self.success_buffer.clear()

