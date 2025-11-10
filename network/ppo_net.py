"""
PPO神经网络架构

包含Actor（策略网络）和Critic（价值网络）的定义
使用GRU处理序列信息，适配多智能体部分观测环境
增强版网络架构：多层MLP + 层归一化 + 残差连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOActor(nn.Module):
    """策略网络 - 负责动作选择（增强版）"""
    
    def __init__(self, input_shape, args):
        super(PPOActor, self).__init__()
        self.args = args
        
        # 简化的两层特征提取网络
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.ln1 = nn.LayerNorm(args.rnn_hidden_dim)
        
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.ln2 = nn.LayerNorm(args.rnn_hidden_dim)

        # RNN层
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        # 输出层
        self.fc_out = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """简化而稳定的初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier初始化，更适合多智能体环境
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRUCell):
                # GRU参数初始化
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, obs, hidden_state):
        """
        前向传播计算动作概率
        
        Args:
            obs: 智能体观察 [batch_size, obs_dim]
            hidden_state: RNN隐藏状态 [batch_size, hidden_dim]
            
        Returns:
            action_logits: 动作logits [batch_size, n_actions]
            new_hidden: 更新后的隐藏状态 [batch_size, hidden_dim]
        """
        # 简化的特征提取
        x = F.relu(self.ln1(self.fc1(obs)))
        x = F.relu(self.ln2(self.fc2(x)))

        # RNN处理时序信息
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        
        # 输出动作logits
        action_logits = self.fc_out(h_out)
        
        return action_logits, h_out


class PPOCritic(nn.Module):
    """价值网络 - 负责状态价值估计"""
    
    def __init__(self, input_shape, args):
        super(PPOCritic, self).__init__()
        self.args = args
        
        # 简化的两层特征提取网络
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.ln1 = nn.LayerNorm(args.rnn_hidden_dim)
        
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.ln2 = nn.LayerNorm(args.rnn_hidden_dim)
        


        # RNN层
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        # 输出层
        self.fc_out = nn.Linear(args.rnn_hidden_dim, 1)
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """简化而稳定的初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRUCell):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, state_info, hidden_state):
        """
        前向传播估计状态价值
        
        Args:
            state_info: 状态信息 [batch_size, state_dim]
            hidden_state: RNN隐藏状态 [batch_size, hidden_dim]
            
        Returns:
            value: 状态价值估计 [batch_size, 1]
            new_hidden: 更新后的隐藏状态 [batch_size, hidden_dim]
        """
        # 简化的特征提取
        x = F.relu(self.ln1(self.fc1(state_info)))
        x = F.relu(self.ln2(self.fc2(x)))

        # RNN处理时序信息
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        
        # 输出价值估计
        value = self.fc_out(h_out)
        
        return value, h_out