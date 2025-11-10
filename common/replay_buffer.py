"""
经验回放缓冲区

提高样本利用效率，减少训练波动
"""

import numpy as np
import random


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, args, buffer_size=5000):
        """
        初始化缓冲区

        Args:
            args: 参数配置
            buffer_size: 缓冲区大小（存储的episode数量）
        """
        self.args = args
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def push(self, episode):
        """
        添加一个episode到缓冲区

        Args:
            episode: 完整的episode数据
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(episode)
        else:
            # 循环覆盖旧数据
            self.buffer[self.position] = episode
            self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        """
        从缓冲区随机采样

        Args:
            batch_size: 采样的episode数量

        Returns:
            batch: 组合后的batch数据
        """
        if len(self.buffer) < batch_size:
            # 缓冲区数据不足，返回全部
            episodes = self.buffer
        else:
            # 随机采样
            episodes = random.sample(self.buffer, batch_size)

        # 合并episodes成batch
        batch = {}
        for key in episodes[0].keys():
            # 检查维度一致性，确保所有episode的第二维（时间步）相同
            try:
                batch[key] = np.concatenate([ep[key] for ep in episodes], axis=0)
            except ValueError as e:
                # 如果维度不匹配，找出最大长度并填充
                max_len = max(ep[key].shape[1] for ep in episodes)
                padded_episodes = []
                for ep in episodes:
                    if ep[key].shape[1] < max_len:
                        # 需要填充
                        pad_len = max_len - ep[key].shape[1]
                        pad_shape = list(ep[key].shape)
                        pad_shape[1] = pad_len

                        # 根据key类型选择填充值
                        if key in ['padded', 'terminated']:
                            pad_value = 1.0  # 填充标记和终止标记用1
                        else:
                            pad_value = 0.0  # 其他用0

                        padding = np.full(pad_shape, pad_value, dtype=ep[key].dtype)
                        padded_ep = np.concatenate([ep[key], padding], axis=1)
                        padded_episodes.append(padded_ep)
                    else:
                        padded_episodes.append(ep[key])

                batch[key] = np.concatenate(padded_episodes, axis=0)

        return batch

    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)

    def clear(self):
        """清空缓冲区"""
        self.buffer = []
        self.position = 0


class PrioritizedReplayBuffer:
    """带优先级的经验回放缓冲区"""

    def __init__(self, args, buffer_size=5000, alpha=0.6):
        """
        初始化优先级缓冲区

        Args:
            args: 参数配置
            buffer_size: 缓冲区大小
            alpha: 优先级指数（0=均匀采样，1=完全按优先级）
        """
        self.args = args
        self.buffer_size = buffer_size
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha

    def push(self, episode, priority=None):
        """
        添加episode到缓冲区

        Args:
            episode: episode数据
            priority: 优先级（通常使用TD error）
        """
        if priority is None:
            # 默认高优先级，确保新数据被采样
            priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(episode)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = episode
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size, beta=0.4):
        """
        按优先级采样

        Args:
            batch_size: 采样数量
            beta: 重要性采样权重指数

        Returns:
            batch: 采样的batch
            indices: 采样的索引
            weights: 重要性采样权重
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # 计算采样概率
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # 归一化

        # 获取采样的episodes
        episodes = [self.buffer[idx] for idx in indices]

        # 合并成batch（处理维度不匹配）
        batch = {}
        for key in episodes[0].keys():
            try:
                batch[key] = np.concatenate([ep[key] for ep in episodes], axis=0)
            except ValueError:
                # 维度不匹配，进行填充
                max_len = max(ep[key].shape[1] for ep in episodes)
                padded_episodes = []
                for ep in episodes:
                    if ep[key].shape[1] < max_len:
                        pad_len = max_len - ep[key].shape[1]
                        pad_shape = list(ep[key].shape)
                        pad_shape[1] = pad_len
                        pad_value = 1.0 if key in ['padded', 'terminated'] else 0.0
                        padding = np.full(pad_shape, pad_value, dtype=ep[key].dtype)
                        padded_ep = np.concatenate([ep[key], padding], axis=1)
                        padded_episodes.append(padded_ep)
                    else:
                        padded_episodes.append(ep[key])
                batch[key] = np.concatenate(padded_episodes, axis=0)

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        """
        更新指定索引的优先级

        Args:
            indices: episode索引
            priorities: 新的优先级值
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        self.priorities = []
        self.position = 0

