"""
环境交互与数据收集模块

负责智能体与SMAC环境的交互，收集训练数据
"""

import numpy as np


class RolloutWorker:
    """环境交互工作器"""
    
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.args = args
        
        # 环境参数
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
    
    def generate_episode(self, episode_num=None, evaluate=False, epsilon=0.05):
        """
        执行一个完整的游戏回合
        
        Args:
            episode_num: 回合序号（用于回放保存）
            evaluate: 评估模式标志（True时使用贪心策略）
            epsilon: 探索概率（仅在训练时使用）

        Returns:
            episode_data: 完整回合的训练数据
            total_reward: 累计奖励值  
            victory: 是否获胜
            steps: 回合步数
        """
        # 初始化回合数据
        if self.args.replay_dir and evaluate and episode_num == 0:
            self.env.close()
        
        # 数据收集容器
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        
        # 环境重置
        self.env.reset()
        terminated = False
        victory = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        
        # 初始化隐藏状态
        self.agents.policy.init_hidden(1)
        
        # 主循环
        while not terminated and step < self.episode_limit:
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            
            # 为每个智能体选择动作
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], 
                                                 agent_id, avail_action, evaluate, epsilon)

                # 创建one-hot编码
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            
            # 执行动作
            reward, terminated, info = self.env.step(actions)
            victory = True if terminated and 'battle_won' in info and info['battle_won'] else False
            
            # 优化的奖励塑形 - 平衡且稳定
            shaped_reward = reward  # 保留原始奖励作为基础

            if not terminated:
                # 存活奖励 - 鼓励智能体存活
                shaped_reward += 0.02
            elif victory:
                # 胜利大奖励
                shaped_reward += 20.0
            else:
                # 失败小惩罚
                shaped_reward -= 1.0

            # 使用塑形后的奖励
            reward = shaped_reward

            # 存储转移数据
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            
            episode_reward += reward
            step += 1
        
        # 获取最后一步的观察和状态
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        
        # 获取最后一步的可用动作
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]
        
        # 填充到episode_limit长度
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            s.append(np.zeros(self.state_shape))
            u.append(np.zeros([self.n_agents, 1]))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            r.append([0.])
            terminate.append([1.])
            padded.append([1.])
            
        for i in range(step, self.episode_limit):
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
        
        episode = dict(o=o.copy(),
                      s=s.copy(),
                      u=u.copy(),
                      r=r.copy(),
                      avail_u=avail_u.copy(),
                      o_next=o_next.copy(),
                      s_next=s_next.copy(),
                      avail_u_next=avail_u_next.copy(),
                      u_onehot=u_onehot.copy(),
                      padded=padded.copy(),
                      terminated=terminate.copy())
        
        # 转换为numpy数组
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        
        # 保存回放（如果需要）
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        
        return episode, episode_reward, victory, step
