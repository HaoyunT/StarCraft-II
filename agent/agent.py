"""
多智能体管理模块

负责智能体的动作选择、策略更新和训练管理
支持MAPPO和IPPO算法的多智能体协作学习
"""

import numpy as np
import torch
from policy.mappo import MAPPO
from torch.distributions import Categorical  # type: ignore


class Agents:
    """多智能体管理器"""
    
    def __init__(self, args):
        # 环境参数
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        
        # 根据算法类型初始化策略 — 目前只支持 MAPPO（即便用户传入 ippo，也会降级并警告）
        if args.alg == 'ippo':
            print("⚠️ 警告: 当前环境未启用 IPPO，已自动使用 MAPPO 替代")
            self.policy = MAPPO(args)
        else:
            self.policy = MAPPO(args)
    
    def choose_action(self, obs, last_action, agent_num, avail_actions, evaluate=False, epsilon=0.05):
        """
        智能体动作选择（改进的探索策略）

        Args:
            obs: 当前观察信息
            last_action: 历史动作（用于网络输入）
            agent_num: 智能体ID
            avail_actions: 可执行动作掩码
            evaluate: 评估模式（True时使用确定性策略）
            epsilon: 探索概率（训练时使用）

        Returns:
            action_id: 选择的动作编号
        """
        inputs = obs.copy()
        # 确保 avail_actions 是 numpy 数组
        avail_actions = np.array(avail_actions)
        avail_actions_ind = np.nonzero(avail_actions)[0]
        
        # 添加智能体ID
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.
        
        # 如果使用上一个动作
        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        
        # 如果网络复用
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        
        # 获取策略网络的隐藏状态
        policy_hidden_state = self.policy.policy_hidden[:, agent_num, :]
        
        # 转换为张量（保持在CPU上，后续可选地移动到GPU）
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)

        if self.args.use_gpu and torch.cuda.is_available():
            inputs = inputs.cuda()
            policy_hidden_state = policy_hidden_state.cuda()

        # 前向传播
        policy_q_value, self.policy.policy_hidden[:, agent_num, :] = self.policy.policy_rnn.forward(
            inputs, policy_hidden_state)
        
        # 数值稳定的动作选择
        with torch.no_grad():
            logits = policy_q_value.squeeze(0).cpu()

            # 使用 torch boolean mask，避免 numpy/torch 混用索引问题
            mask = torch.tensor(avail_actions == 1, dtype=torch.bool)
            masked_logits = logits.clone()
            # 将不可用动作置为极小值
            masked_logits[~mask] = -1e10

            if evaluate:
                # 评估模式：选择最优合法动作
                action = int(torch.argmax(masked_logits).item())
                # 若选到不可用动作则回退
                if avail_actions[action] == 0:
                    action = int(np.random.choice(avail_actions_ind)) if len(avail_actions_ind) > 0 else 0
            else:
                # 训练模式：epsilon-greedy探索
                if np.random.rand() < epsilon and len(avail_actions_ind) > 0:
                    # 随机探索
                    action = int(np.random.choice(avail_actions_ind))
                else:
                    # 按概率采样
                    action_probs = torch.softmax(masked_logits, dim=-1)
                    # 确保不可用动作概率为0并归一化
                    action_probs[~mask] = 0.0
                    sum_probs = action_probs.sum().item()
                    if sum_probs <= 0 or np.isnan(sum_probs):
                        # 如果概率全为0或数值异常，随机选择可用动作
                        action = int(np.random.choice(avail_actions_ind)) if len(avail_actions_ind) > 0 else 0
                    else:
                        action_probs = action_probs / action_probs.sum()
                        dist = Categorical(action_probs)
                        action = int(dist.sample().item())
                        # 双重检查：如果采样到不可用动作，回退到随机可用动作
                        if avail_actions[action] == 0:
                            action = int(np.random.choice(avail_actions_ind)) if len(avail_actions_ind) > 0 else 0

        return action

    def _get_max_episode_len(self, batch):
        """获取batch中的最大回合长度"""
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        
        return max_episode_len
    
    def train(self, batch, train_step=0, time_steps=0, epsilon=None):
        """
        训练智能体
        
        Args:
            batch: 训练数据批次
            train_step: 训练步数
            time_steps: 时间步数
            epsilon: 探索率（未使用）
        """
        max_episode_len = self._get_max_episode_len(batch)
        
        # 裁剪batch到最大回合长度
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        
        # 调用PPO学习 (MAPPO或IPPO)
        self.policy.learn(batch, max_episode_len, train_step, time_steps)
        
        # 定期保存模型
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
            print(f"训练步数: {train_step}, 模型已保存")