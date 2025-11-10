import torch
import os
from network.ppo_net import PPOActor, PPOCritic
from torch.distributions import Categorical # type: ignore


class MAPPO:
    """Multi-Agent Proximal Policy Optimization算法实现
    
    支持MAPPO（集中式价值函数）和IPPO（独立式价值函数）两种模式
    """
    
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        
        # Actor输入包括观察
        actor_input_shape = self.obs_shape
        
        # 如果使用上一个动作作为输入
        if args.last_action:
            actor_input_shape += self.n_actions
        
        # 如果网络复用（需要智能体ID）
        if args.reuse_network:
            actor_input_shape += self.n_agents
            
        self.args = args
        
        # 计算Critic输入维度
        critic_input_shape = self._get_critic_input_shape()
        
        # 创建Actor和Critic网络
        self.policy_rnn = PPOActor(actor_input_shape, args)
        self.eval_critic = PPOCritic(critic_input_shape, self.args)
        
        # 算法类型信息
        algorithm_type = "MAPPO (集中式价值函数)" if args.alg == 'mappo' else "IPPO (独立式价值函数)"
        print(f"初始化 {algorithm_type}")
        print(f"  Actor输入维度: {actor_input_shape}")
        print(f"  Critic输入维度: {critic_input_shape}")
        
        # GPU设置
        if self.args.use_gpu:
            device = torch.cuda.current_device()
            print(f"正在使用GPU设备: {torch.cuda.get_device_name(device)}")
            self.policy_rnn.cuda()
            self.eval_critic.cuda()
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 模型保存路径
        self.model_dir = os.path.join(args.model_dir, args.alg, args.map)
        
        # 优化器
        # 使用参数组分别设置actor和critic学习率，利于更精细的调参
        self.actor_parameters = list(self.policy_rnn.parameters())
        self.critic_parameters = list(self.eval_critic.parameters())

        if args.optimizer == "RMS":
            self.ac_optimizer = torch.optim.RMSprop([
                {'params': self.actor_parameters, 'lr': args.lr_actor},
                {'params': self.critic_parameters, 'lr': args.lr_critic}
            ])
        else:  # 默认使用Adam
            self.ac_optimizer = torch.optim.Adam([
                {'params': self.actor_parameters, 'lr': args.lr_actor},
                {'params': self.critic_parameters, 'lr': args.lr_critic}
            ], eps=1e-5, betas=(0.9, 0.999))

        # 学习率调度器 - 改为指数衰减，收敛更快
        # 使用更温和的多步衰减，防止学习率下降过快
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.ac_optimizer, milestones=[5000, 10000, 15000], gamma=0.5
        )

        # 价值函数归一化 - 运行均值和方差
        self.value_normalizer_mean = 0.0
        self.value_normalizer_var = 1.0
        self.value_normalizer_count = 1e-8  # 使用更小的初始计数避免偏差

        # 隐藏状态
        self.policy_hidden = None
        self.eval_critic_hidden = None

    def _update_value_normalizer(self, rewards, mask):
        """更新价值归一化统计量（使用Welford算法，数值更稳定）"""
        # 支持传入 returns（或 rewards），只统计有效位置
        valid_rewards = rewards[mask == 1]
        if len(valid_rewards) > 0:
            batch_mean = valid_rewards.mean().item()
            batch_var = valid_rewards.var(unbiased=False).item()  # 使用总体方差
            batch_count = len(valid_rewards)

            # Welford增量更新均值和方差（更数值稳定）
            delta = batch_mean - self.value_normalizer_mean
            tot_count = self.value_normalizer_count + batch_count

            new_mean = self.value_normalizer_mean + delta * batch_count / tot_count
            m_a = self.value_normalizer_var * self.value_normalizer_count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.value_normalizer_count * batch_count / tot_count
            new_var = M2 / tot_count

            # 使用指数移动平均平滑更新，减少突变
            alpha = 0.99  # 平滑系数
            self.value_normalizer_mean = alpha * self.value_normalizer_mean + (1 - alpha) * new_mean
            self.value_normalizer_var = alpha * self.value_normalizer_var + (1 - alpha) * max(new_var, 1e-6)
            self.value_normalizer_count = min(tot_count, 1e6)  # 限制最大计数避免溢出

    def _get_critic_input_shape(self):
        """计算Critic网络输入维度"""
        if self.args.alg == 'mappo' and self.args.use_centralized_V:
            # MAPPO: 全局状态 + 智能体ID (如果需要)
            input_shape = self.state_shape
            if self.args.reuse_network:
                input_shape += self.n_agents
        else:
            # IPPO: 局部观察 + 智能体ID (如果需要)
            input_shape = self.obs_shape
            if self.args.reuse_network:
                input_shape += self.n_agents
        
        # 如果使用历史动作
        if self.args.last_action:
            input_shape += self.n_actions
            
        return input_shape
    
    def learn(self, batch, max_episode_len, train_step, time_steps=0):
        """MAPPO学习过程"""
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        
        # 数据类型转换
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        
        u, r, avail_u, terminated, s = batch['u'], batch['r'], batch['avail_u'], batch['terminated'], batch['s']
        
        # 创建mask（用于处理不同长度的回合）
        mask = (1 - batch["padded"].float())
        
        # GPU设置
        device = torch.device('cuda') if self.args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
        u = u.to(device)
        mask = mask.to(device)
        r = r.to(device)
        terminated = terminated.to(device)
        s = s.to(device)

        # 重复维度以匹配智能体数量
        mask = mask.repeat(1, 1, self.n_agents)
        r = r.repeat(1, 1, self.n_agents)
        terminated = terminated.repeat(1, 1, self.n_agents)
        
        # 更新价值归一化统计
        self._update_value_normalizer(r, mask)

        # 获取旧的价值和动作概率（不保存到计算图）
        with torch.no_grad():
            old_values, _ = self._get_values(batch, max_episode_len)
            old_action_prob = self._get_action_prob(batch, max_episode_len)
        old_values = old_values.squeeze(dim=-1)

        # 确保没有NaN值，用掩码处理填充部分
        # 对于填充的时间步，使用均匀分布
        if torch.isnan(old_action_prob).any():
            print(f"警告: old_action_prob包含NaN值，正在修复...")
            old_action_prob = torch.nan_to_num(old_action_prob, nan=0.0)

        # 对于全零的概率分布（填充部分），设置为均匀分布避免Categorical错误
        zero_prob_mask = (old_action_prob.sum(dim=-1) == 0)
        if zero_prob_mask.any():
            # 使用可用动作的均匀分布，确保在同一设备上
            uniform_prob = avail_u.float() / (avail_u.sum(dim=-1, keepdim=True).float() + 1e-10)
            if self.args.use_gpu:
                uniform_prob = uniform_prob.cuda()
                zero_prob_mask = zero_prob_mask.cuda()
            old_action_prob[zero_prob_mask] = uniform_prob[zero_prob_mask]

        # 再次检查并添加小常数确保数值稳定
        old_action_prob = old_action_prob + 1e-10
        old_action_prob = old_action_prob / old_action_prob.sum(dim=-1, keepdim=True)

        old_dist = Categorical(probs=old_action_prob)
        old_log_pi_taken = old_dist.log_prob(u.squeeze(dim=-1))
        old_log_pi_taken[mask == 0] = 0.0
        
        # PPO更新循环
        # 诊断指标收集
        ppo_metrics = []
        for epoch in range(self.args.ppo_n_epochs):
            # 每个epoch重新初始化隐藏状态，确保独立的前向传播
            # 这样可以避免不同epoch之间的梯度累积和状态污染
            self.init_hidden(episode_num)

            # 获取当前价值估计
            values, target_values = self._get_values(batch, max_episode_len)
            values = values.squeeze(dim=-1)
            
            # 计算回报和优势 - 使用改进的GAE算法
            returns = torch.zeros_like(r).detach()
            advantages = torch.zeros_like(r).detach()

            # 反向计算GAE和回报 - 使用no_grad确保不会构建计算图
            with torch.no_grad():
                gae = 0.0
                for transition_idx in reversed(range(max_episode_len)):
                    if transition_idx == max_episode_len - 1:
                        next_value = 0.0
                    else:
                        next_value = values[:, transition_idx + 1].detach()

                    # TD误差
                    delta = r[:, transition_idx] + self.args.gamma * next_value * (
                        1 - terminated[:, transition_idx]) - values[:, transition_idx].detach()

                    # GAE优势估计
                    gae = delta + self.args.gamma * self.args.lamda * (
                        1 - terminated[:, transition_idx]) * gae

                    advantages[:, transition_idx] = gae
                    returns[:, transition_idx] = gae + values[:, transition_idx].detach()

            # 应用价值归一化（保证在同一设备）
            # 在使用归一化前，先用本次计算得到的 returns 来更新统计量（比使用即时 reward 更合理）
            try:
                # returns 是跟 values 维度相同的张量 (episode, max_len, n_agents)
                self._update_value_normalizer(returns, mask)
            except Exception:
                pass

            if self.value_normalizer_count > 0:
                var_tensor = torch.tensor(self.value_normalizer_var, device=device)
                mean_tensor = torch.tensor(self.value_normalizer_mean, device=device)
                returns_normalized = (returns.to(device) - mean_tensor) / (torch.sqrt(var_tensor) + 1e-8)
            else:
                returns_normalized = returns.to(device)

            # 优势标准化 - 更鲁棒的方法
            valid_advantages = advantages[mask == 1]
            if len(valid_advantages) > 1:
                adv_mean = valid_advantages.mean().detach()
                adv_std = valid_advantages.std().detach()
                # 使用更保守的标准化，防止极值
                adv_std = torch.clamp(adv_std, min=1e-5, max=5.0)
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            # 更温和的截断，保留更多信息
            advantages = torch.clamp(advantages, min=-10.0, max=10.0)

            # 确保优势是detached的
            advantages = advantages * mask  # 应用mask

            advantages = advantages.to(device)

            # 计算当前动作概率
            action_prob = self._get_action_prob(batch, max_episode_len)

            # 确保没有NaN值
            if torch.isnan(action_prob).any():
                action_prob = torch.nan_to_num(action_prob, nan=0.0)

            # 处理零概率分布
            zero_prob_mask = (action_prob.sum(dim=-1) == 0)
            if zero_prob_mask.any():
                uniform_prob = avail_u.float() / (avail_u.sum(dim=-1, keepdim=True).float() + 1e-10)
                if self.args.use_gpu:
                    uniform_prob = uniform_prob.cuda()
                    zero_prob_mask = zero_prob_mask.cuda()
                action_prob[zero_prob_mask] = uniform_prob[zero_prob_mask]

            # 确保数值稳定
            action_prob = action_prob + 1e-10
            action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)

            dist = Categorical(probs=action_prob)
            log_pi_taken = dist.log_prob(u.squeeze(dim=-1))
            log_pi_taken[mask == 0] = 0.0
            
            # 近似KL散度监控（便于提前停止）
            with torch.no_grad():
                # old_action_prob 已经是上一策略的概率分布
                kl_div = (old_action_prob * (torch.log(old_action_prob + 1e-10) - torch.log(action_prob + 1e-10))).sum(dim=-1)
                kl_mean = (kl_div * mask.squeeze(-1)).sum() / mask.sum()

            # PPO损失计算 - 改进稳定性
            ratios = torch.exp(log_pi_taken - old_log_pi_taken.detach())
            
            # 不再对 ratios 进行额外硬裁剪，让 PPO 的截断项负责限制更新
            # 但可以在需要时启用自适应上限（这里保留为正常流程）

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantages
            
            entropy = dist.entropy()
            entropy[mask == 0] = 0.0

            # 熵系数随训练线性衰减，减少后期探索（保持下界）
            try:
                entropy_min = getattr(self.args, 'entropy_min', 0.01)
                entropy_initial = getattr(self, 'entropy_initial', None)
                if entropy_initial is None:
                    self.entropy_initial = getattr(self.args, 'entropy_coeff', 0.01)
                    entropy_initial = self.entropy_initial
                # 线性衰减到下界
                frac = min(1.0, float(train_step) / max(1, getattr(self.args, 'n_steps', 1)))
                entropy_coef = max(entropy_min, entropy_initial * (1.0 - frac))
            except Exception:
                entropy_coef = getattr(self.args, 'entropy_coeff', 0.01)

            # 策略损失（带掩码）
            policy_loss = torch.min(surr1, surr2)
            # 增加熵奖励，鼓励探索
            policy_loss = policy_loss + entropy_coef * entropy
            policy_loss = -(policy_loss * mask).sum() / mask.sum()
            
            # 价值函数损失 - 使用归一化后的returns
            value_loss_unclipped = (values - returns_normalized) ** 2

            # 可选的裁剪以防止价值函数过度更新
            value_pred_clipped = old_values.detach() + torch.clamp(
                values - old_values.detach(), -self.args.clip_param, self.args.clip_param
            )
            value_loss_unclipped = (values - returns_normalized) ** 2
            value_loss_clipped = (value_pred_clipped - returns_normalized) ** 2

            # 使用最大的损失，更加保守
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
            # 截断异常值，防止价值函数发散
            value_loss = torch.clamp(value_loss, max=100.0)
            value_loss = (mask * value_loss).sum() / mask.sum()
            
            # 总损失 - 平衡策略和价值学习
            loss = policy_loss + self.args.value_loss_coef * value_loss

            # KL过大时提前停止当前batch的后续epoch更新
            if hasattr(self.args, 'kl_target') and kl_mean.item() > self.args.kl_target * self.args.kl_stop_multiplier:
                print(f"KL过高({kl_mean.item():.4f})，提前停止剩余epoch")
                # 可选：降低学习率以更稳定
                if getattr(self.args, 'kl_adapt', False):
                    for g in self.ac_optimizer.param_groups:
                        g['lr'] = max(g['lr'] * 0.9, g['lr'] * 0.5)
                break

            # 反向传播
            self.ac_optimizer.zero_grad()

            # 确保loss是有效的标量值
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 检测到无效损失值 {loss}，跳过此次更新")
                continue

            # 清零梯度
            self.ac_optimizer.zero_grad()

            # 反向传播 - 对于第一个epoch不需要retain_graph
            loss.backward(retain_graph=False)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor_parameters + self.critic_parameters, self.args.grad_norm_clip)
            self.ac_optimizer.step()

            # 收集诊断指标（在CPU上）
            try:
                pl = float(policy_loss.detach().cpu().item()) if isinstance(policy_loss, torch.Tensor) else float(policy_loss)
            except Exception:
                pl = 0.0
            try:
                vl = float(value_loss.detach().cpu().item()) if isinstance(value_loss, torch.Tensor) else float(value_loss)
            except Exception:
                vl = 0.0
            try:
                ent = float(entropy.mean().detach().cpu().item())
            except Exception:
                ent = 0.0
            try:
                klv = float(kl_mean.detach().cpu().item()) if isinstance(kl_mean, torch.Tensor) else float(kl_mean)
            except Exception:
                klv = 0.0

            ppo_metrics.append({'policy_loss': pl, 'value_loss': vl, 'entropy': ent, 'kl': klv})

            # 清理计算图，防止内存泄漏
            del loss, policy_loss, value_loss
            if self.args.use_gpu:
                torch.cuda.empty_cache()

        # 在一个批次的所有 epoch 完成后，打印诊断信息帮助定位不稳定来源
        if len(ppo_metrics) > 0:
            avg_pl = sum(m['policy_loss'] for m in ppo_metrics) / len(ppo_metrics)
            avg_vl = sum(m['value_loss'] for m in ppo_metrics) / len(ppo_metrics)
            avg_ent = sum(m['entropy'] for m in ppo_metrics) / len(ppo_metrics)
            avg_kl = sum(m['kl'] for m in ppo_metrics) / len(ppo_metrics)
            print(f"[MAPPO] train_step={train_step} time_steps={time_steps} ppo_epochs={len(ppo_metrics)} avg_policy_loss={avg_pl:.4f} avg_value_loss={avg_vl:.4f} avg_entropy={avg_ent:.4f} avg_kl={avg_kl:.6f}")

            # 也把诊断结果保存到 CSV 便于后续可视化
            try:
                metrics_dir = os.path.join(self.args.result_dir, self.args.alg, self.args.map)
                os.makedirs(metrics_dir, exist_ok=True)
                csv_path = os.path.join(metrics_dir, 'ppo_metrics.csv')
                write_header = not os.path.exists(csv_path)
                with open(csv_path, 'a', encoding='utf-8') as f:
                    if write_header:
                        f.write('train_step,time_steps,ppo_epochs,avg_policy_loss,avg_value_loss,avg_entropy,avg_kl\n')
                    f.write(f"{train_step},{time_steps},{len(ppo_metrics)},{avg_pl:.6f},{avg_vl:.6f},{avg_ent:.6f},{avg_kl:.6f}\n")
            except Exception as e:
                print(f"警告: 无法保存 PPO 诊断 CSV - {e}")

        # 更新学习率
        self.lr_scheduler.step()

        # GPU内存清理 - 防止内存泄漏
        if self.args.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        """
        获取Critic网络输入
        MAPPO: 使用全局状态 + 智能体ID
        IPPO: 使用局部观察 + 智能体ID (如果网络复用)
        """
        episode_num = batch['o'].shape[0]
        
        if self.args.alg == 'mappo' and self.args.use_centralized_V:
            # MAPPO: 使用全局状态
            s, s_next = batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
            
            # 扩展到每个智能体
            s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
            s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
            
            inputs_list = [s]
            inputs_next_list = [s_next]
            
        else:
            # IPPO: 使用局部观察
            obs = batch['o'][:, transition_idx]  # shape: (episode_num, n_agents, obs_shape)
            if transition_idx < max_episode_len - 1:
                obs_next = batch['o'][:, transition_idx + 1]
            else:
                obs_next = obs  # 最后一步使用当前观察
            
            inputs_list = [obs]
            inputs_next_list = [obs_next]
        
        # 添加历史动作 (如果需要)
        if self.args.last_action:
            u_onehot = batch['u_onehot']
            if transition_idx == 0:
                last_action = torch.zeros_like(u_onehot[:, 0])
            else:
                last_action = u_onehot[:, transition_idx - 1]
            
            inputs_list.append(last_action)
            
            # 对于next，使用当前动作
            if transition_idx < max_episode_len - 1:
                next_action = u_onehot[:, transition_idx]
            else:
                next_action = last_action
            inputs_next_list.append(next_action)
        
        # 添加智能体ID (如果网络复用)
        if self.args.reuse_network:
            agent_id = torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1)
            inputs_list.append(agent_id)
            inputs_next_list.append(agent_id)
        
        # 拼接所有输入
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_list], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next_list], dim=1)
        
        return inputs, inputs_next
    
    def _get_values(self, batch, max_episode_len):
        """获取价值估计"""
        episode_num = batch['o'].shape[0]
        v_evals = []

        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)

            if self.args.use_gpu:
                inputs = inputs.cuda()

            # 使用当前隐藏状态并更新
            local_hidden = self.eval_critic_hidden.reshape(-1, self.args.rnn_hidden_dim)
            
            v_eval, new_hidden = self.eval_critic(inputs, local_hidden)
            
            # 更新隐藏状态以保持时序连贯性
            self.eval_critic_hidden = new_hidden.view(episode_num, self.n_agents, -1)
            
            v_eval = v_eval.view(episode_num, self.n_agents, -1)
            v_evals.append(v_eval)

        v_evals = torch.stack(v_evals, dim=1)
        return v_evals, None

    def _get_actor_inputs(self, batch, transition_idx):
        """获取Actor网络输入"""
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot']
        episode_num = obs.shape[0]
        inputs = []
        
        inputs.append(obs)
        
        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs
    
    def _get_action_prob(self, batch, max_episode_len):
        """获取动作概率"""
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []

        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)

            if self.args.use_gpu:
                inputs = inputs.cuda()

            # 使用当前隐藏状态并更新
            local_hidden = self.policy_hidden.reshape(-1, self.args.rnn_hidden_dim)

            outputs, new_hidden = self.policy_rnn(inputs, local_hidden)
            
            # 更新隐藏状态以保持时序连贯性
            self.policy_hidden = new_hidden.view(episode_num, self.n_agents, -1)
            
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1).cpu()

        # 添加小常数防止数值下溢
        action_prob = action_prob + 1e-10

        # 应用可用动作掩码
        action_prob[avail_actions == 0] = 0.0

        # 重新归一化，添加数值稳定性检查
        action_sum = action_prob.sum(dim=-1, keepdim=True)
        # 防止除以零或极小值
        action_sum = torch.clamp(action_sum, min=1e-8)
        action_prob = action_prob / action_sum

        # 再次应用掩码确保不可用动作概率为0
        action_prob[avail_actions == 0] = 0.0

        # 检查并修复NaN值
        if torch.isnan(action_prob).any():
            print("警告: 检测到NaN值，正在修复...")
            action_prob = torch.nan_to_num(action_prob, nan=0.0)
            # 如果某个时间步所有动作都不可用，设置均匀分布
            zero_sum_mask = (action_prob.sum(dim=-1) == 0)
            if zero_sum_mask.any():
                # 对于全零的情况，使用可用动作的均匀分布
                uniform_prob = avail_actions.float() / (avail_actions.sum(dim=-1, keepdim=True).float() + 1e-10)
                action_prob[zero_sum_mask] = uniform_prob[zero_sum_mask]

        if self.args.use_gpu:
            action_prob = action_prob.cuda()

        return action_prob

    def init_hidden(self, episode_num):
        """初始化隐藏状态"""
        self.policy_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.eval_critic_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

        # 初始化时就放到GPU上，避免后续反复迁移
        if self.args.use_gpu:
            self.policy_hidden = self.policy_hidden.cuda()
            self.eval_critic_hidden = self.eval_critic_hidden.cuda()

    def save_model(self, train_step):
        """保存模型"""
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), os.path.join(self.model_dir, f'{num}_critic_params.pkl'))
        torch.save(self.policy_rnn.state_dict(), os.path.join(self.model_dir, f'{num}_rnn_params.pkl'))
        print(f"模型已保存到 {self.model_dir}")