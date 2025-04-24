import torch
import numpy as np
import gym


class Agent_PPO2_continuous():
    def __init__(self, Critic, Actor, Env, ppo_epoch, batch_size, action_space, state_space, env_name, device='cpu',
                 shuffle=True, up_bound=1.0, down_bound=0.0, render=False, dist_type='beta'):
        """
        初始化PPO智能体（支持Beta分布和高斯分布）
        :param Critic: 价值网络
        :param Actor: 策略网络
        :param Env: 环境对象
        :param ppo_epoch: PPO训练轮数
        :param batch_size: 批量大小
        :param action_space: 动作维度
        :param state_space: 状态维度
        :param env_name: 环境名称
        :param device: 计算设备
        :param shuffle: 是否打乱数据
        :param render: 是否渲染环境
        :param dist_type: 分布类型 ('beta' 或 'gaussian')
        """
        self.Critic = Critic
        self.Actor = Actor
        self.Env = Env
        self.ppo_epoch = ppo_epoch
        self.batch_size = batch_size
        self.action_space = action_space
        self.state_space = state_space
        self.env_name = env_name
        self.device = device
        self.shuffle = shuffle
        self.render = render
        self.dist_type = dist_type.lower()
        self.up_bound = up_bound
        self.down_bound = down_bound

    def train_show_train(self):
        """展示一轮游戏"""
        self.Env.close()
        env = gym.make(self.env_name, render_mode='human')
        state = env.reset()
        state = state[0]
        terminated = False
        while not terminated:
            with torch.no_grad():
                action, _, _ = self.act(state)
            state, reward, terminated, _, _ = env.step(action)
        env.close()
        self.Env = gym.make(self.env_name, render_mode=None)

    def show(self):
        """循环展示"""
        self.Env.close()
        env = gym.make(self.env_name, render_mode='human')
        while True:
            state = env.reset()
            state = state[0]
            terminated = False
            while not terminated:
                with torch.no_grad():
                    action, _, _ = self.act(state)
                state, reward, terminated, _, _ = env.step(action)


    def run_episode(self):
        """
        进行一次episode周期
        :return: avg_actor_loss, avg_critic_loss, avg_episode_score, episode_cnt
        """
        state = self.Env.reset()
        state = state[0]
        terminated, score = False, 0

        # 初始化经验缓存
        states, next_states, actions, rewards, log_probs, dones = [], [], [], [], [], []
        episode_score = 0
        episode_cnt = 0
        batch_cnt = 0

        for t in range(self.batch_size):
            if self.render:
                self.Env.render()

            # 1. 选择动作
            with torch.no_grad():
                action, action_tensor, log_prob = self.act(state)

            # 2. 与环境交互
            next_state, reward, terminated, _, _ = self.Env.step(action)

            # 3. 存储数据
            if batch_cnt == 400:
                terminated = True
            states.append(state)
            next_states.append(torch.FloatTensor(next_state).unsqueeze(0))
            actions.append(action_tensor)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(terminated)

            # 4. 更新状态
            state = next_state
            batch_cnt += 1
            score += reward

            # 5. 处理回合终止
            if terminated:
                episode_cnt += 1
                state = self.Env.reset()
                state = state[0]
                episode_score += score
                score = 0
                terminated = False

        if episode_cnt == 0:
            episode_cnt = 1
            episode_score = score
        avg_episode_score = episode_score / episode_cnt

        # 6. 模型训练
        avg_actor_loss, avg_critic_loss = self.train(
            states=states,
            actions=actions,
            rewards=rewards,
            old_log_probs=log_probs,
            dones=dones,
            next_states=next_states
        )
        return avg_actor_loss, avg_critic_loss, avg_episode_score, episode_cnt

    def act(self, state):
        """
        根据状态选择动作
        :param state: 状态观测值 (list) 形状: [state_space]
        :return:
            action: 动作值 (float或list), 与 Evn的输入相适配
            action_tensor: 动作张量 (torch.Tensor) 形状: [action_space]
            log_prob: 概率对数 (torch.Tensor) 形状: [1](action_space大于1时为联合概率)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            action_tensor, log_prob = self.Actor.predict(state_tensor)

            return action_tensor.tolist()[0], action_tensor, log_prob

    def train(self, states, actions, rewards, old_log_probs, dones, next_states):
        """
        PPO训练步骤
        :param states: 状态列表 [batch_size, state_space]
        :param actions: 动作张量 [batch_size, action_space]
        :param rewards: 奖励列表 [batch_size]
        :param old_log_probs: 旧策略对数概率 [batch_size]
        :param dones: 终止标志 [batch_size]
        :param next_states: 下一状态列表 [batch_size, state_space]
        :return: actor_loss, critic_loss (float)
        """
        # 转换为PyTorch张量
        # 转换为PyTorch张量
        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        actions = torch.FloatTensor(np.vstack(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算GAE
        with torch.no_grad():
            values = self.Critic.predict(states)
            next_values = self.Critic.predict(next_states)
            advantages, targets = self._compute_gae(rewards, dones, values, next_values)

        # 训练循环
        actor_losses, critic_losses = [], []
        for _ in range(self.ppo_epoch):
            # 打乱数据
            if self.shuffle:
                indices = torch.randperm(len(states))
                states = states[indices]
                actions = actions[indices]
                advantages = advantages[indices]
                targets = targets[indices]
                old_log_probs = old_log_probs[indices]

            y_true = torch.cat([
                advantages, old_log_probs,
                actions
            ], dim=1)

            # 获取新策略分布
            new_dist = self.Actor(states)

            # Actor训练
            self.Actor.optimizer.zero_grad()
            actor_loss = self.Actor.ppo_loss(y_true, new_dist)
            actor_loss.backward()
            self.Actor.optimizer.step()

            # Critic训练
            self.Critic.optimizer.zero_grad()
            critic_loss = self.Critic.critic_PPO2_loss(values, targets, self.Critic(states))
            critic_loss.backward()
            self.Critic.optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)

    @staticmethod
    def _compute_gae(rewards, dones, values, next_values, gamma=0.99, lamda=0.9):
        """
        计算广义优势估计(GAE)和目标价值
        :param rewards: 奖励序列 [batch_size, 1]
        :param dones: 终止标志 [batch_size, 1]
        :param values: 状态价值估计 [batch_size, 1]
        :param next_values: 下一状态价值估计 [batch_size, 1]
        :param gamma: 折扣因子
        :param lamda: GAE参数
        :return: advantages, targets
        """
        deltas = rewards + gamma * (1 - dones) * next_values - values
        gaes = torch.zeros_like(deltas)
        gaes[-1] = deltas[-1]

        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = deltas[t] + gamma * lamda * (1 - dones[t]) * gaes[t + 1]

        targets = gaes + values
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return gaes, targets
