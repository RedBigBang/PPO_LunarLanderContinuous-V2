import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal


class ActorPPO2_continuous(nn.Module):
    def __init__(self, state_space, action_space, device, lr=0.001, up_bound=1.0, down_bound=0.0, dist_type='beta'):
        """
        初始化PPO Actor网络，支持Beta分布和高斯分布。

        Args:
            state_space (int): 输入状态的维度（例如状态向量长度）。
            action_space (int): 动作空间的维度（例如2表示二维动作）。
            lr (float): 优化器学习率，默认0.001。
            dist_type (str): 分布类型，可选 'beta' 或 'gaussian'。
            device : 计算设备（'cpu' 或 'cuda'）。
        """
        super(ActorPPO2_continuous, self).__init__()
        self.action_space = action_space
        self.dist_type = dist_type.lower()  # 统一转为小写
        self.up_bound = up_bound
        self.down_bound = down_bound

        # 共享网络层
        self.fc1 = nn.Linear(state_space, 512)  # 输入层 → 隐藏层1
        self.fc2 = nn.Linear(512, 256)  # 隐藏层1 → 隐藏层2
        self.fc3 = nn.Linear(256, 64)  # 隐藏层2 → 隐藏层3

        # 根据分布类型定义输出层
        if self.dist_type == 'beta':
            # Beta分布需要输出α和β参数（需满足α>0, β>0）
            self.fc_alpha = nn.Linear(64, action_space)  # 输出α参数
            self.fc_beta = nn.Linear(64, action_space)  # 输出β参数
        elif self.dist_type == 'gaussian':
            # 高斯分布输出均值（μ）和对数标准差（log_std）
            self.fc_mean = nn.Linear(64, action_space)  # 输出均值
            self.fc_log_std = nn.Linear(64, action_space)  # 输出对数标准差
        else:
            raise ValueError("dist_type must be 'beta' or 'gaussian'")

        # 初始化权重
        self._initialize_weights()

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # 指定计算设备
        self.to(device)

    def _initialize_weights(self):
        """
        初始化网络权重。
        - 线性层：权重用正态分布初始化（mean=0, std=0.01），偏置初始化为0。
        - Beta分布：α和β初始化为接近1的值（避免极端分布）。
        - 高斯分布：对数标准差初始化为较小值（鼓励初期探索）。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

        # 分布相关参数的特殊初始化
        if self.dist_type == 'beta':
            nn.init.constant_(self.fc_alpha.weight, 0.5)  # α初始值
            nn.init.constant_(self.fc_beta.weight, 0.5)  # β初始值
        elif self.dist_type == 'gaussian':
            nn.init.constant_(self.fc_log_std.weight, -0.5)  # log_std初始值

    def forward(self, x):
        """
        前向传播
        :param x: 输入状态 (torch.Tensor)
                  形状 [batch_size, state_space]
        :return: 动作分布(torch.distributions.Distribution)
                 形状[batch_size, action_space]
        """
        x = F.relu(self.fc1(x))  # 激活函数：ReLU
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        if self.dist_type == 'beta':
            # Beta分布：α和β通过softplus确保>0（加1e-5防止数值下溢）
            alpha = F.softplus(self.fc_alpha(x)) + 1e-5  # 形状 [batch_size, action_space]
            beta = F.softplus(self.fc_beta(x)) + 1e-5
            return Beta(alpha, beta)  # 创建Beta分布

        elif self.dist_type == 'gaussian':
            # 高斯分布：均值用tanh约束到[-1, 1]，标准差通过exp(log_std)得到
            mean = torch.tanh(self.fc_mean(x))  # 形状 [batch_size, action_space]
            log_std = self.fc_log_std(x)  # 对数标准差
            std = torch.exp(log_std)  # 标准差（正数）
            return Normal(mean, std)  # 创建高斯分布

    def ppo_loss(self, y_true, y_pred):
        """
        PPO的损失函数
        :param y_true: 真实数据 (torch.Tensor)
                       结构:[advantages, old_log_prob, actions]
                       形状:[batch_size, 2 + action_space]
        :param y_pred: 当前策略的动作分布(torch.distributions.Distribution)
                       形状:[batch_size, action_space]
        :return:总损失 torch.Tensor
        """
        # 拆分y_true
        advantages = y_true[:, :1]  # 优势函数，形状 [batch_size, 1]
        old_log_prob = y_true[:, 1:2]
        actions = y_true[:, 2:]

        # 新策略的分布（y_pred）
        new_dist = y_pred

        # action缩放回原有分布
        actions = self._back_to_bounds(actions)  # [-1, 1]
        if self.dist_type == 'beta':
            actions = (actions + 1)/2 # [0, 1]
        if self.dist_type == 'gaussian':
            actions = actions
        # 计算概率比（重要性采样比）
        new_log_prob= new_dist.log_prob(actions).sum(dim=1, keepdim=True)  # 新策略对数概率log(p(a1,a2)) = sum(log(a)) =
        # log(p(a1)+p(a2)) [batch_size, 1]

        prob_ratio = torch.exp(new_log_prob - old_log_prob)  # 概率比，形状 [batch_size, 1]

        # Clipped Surrogate Loss
        LOSS_CLIPPING = 0.2  # PPO裁剪超参数
        p1 = prob_ratio * advantages  # 未裁剪的损失
        p2 = torch.clamp(prob_ratio, 1 - LOSS_CLIPPING,  # 裁剪后的损失
                         1 + LOSS_CLIPPING) * advantages
        actor_loss = -torch.mean(torch.min(p1, p2))  # 取最小值保证保守更新

        # 熵正则化（鼓励探索）
        ENTROPY_LOSS = 0.01  # 熵系数
        entropy = new_dist.entropy().mean()  # 计算分布熵的均值
        entropy_loss = -ENTROPY_LOSS * entropy  # 最大化熵

        # 总损失 = Actor Loss + Entropy Loss
        return actor_loss + entropy_loss

    def predict(self, state):
        """
        预测单步动作（自动处理数值稳定性）

        Args:
            state (torch.Tensor): 输入状态，形状 [state_space] 或 [batch_size, state_space]。

        Returns:
            action (torch.Tensor): 缩放至目标范围的动作，形状与输入batch维度相同。
            log_prob (torch.Tensor): 修正后的对数概率，形状 [batch_size, 1]。
        """
        with torch.no_grad():
            # 确保输入有batch维度
            state = state.unsqueeze(0) if state.dim() == 1 else state

            # 获取原始分布和采样
            dist = self.forward(state)
            raw_action = dist.sample()

            # 边界保护（优先处理原始采样）
            if self.dist_type == 'beta':
                raw_action = torch.clamp(raw_action, 1e-6, 1 - 1e-6)
                action = 2.0 * raw_action - 1.0  # [0,1] -> [-1,1]
            elif self.dist_type == 'gaussian':
                action = torch.clamp(raw_action, -1.0, 1.0)  # 硬截断

            # 线性缩放至目标范围
            action = self._scale_to_bounds(action)

            # 计算修正后的log_prob
            log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
            if self.dist_type == 'beta':
                log_prob -= torch.log(torch.tensor(0.5 * (self.up_bound - self.down_bound)))

            return action, log_prob

    def _scale_to_bounds(self, x):
        """将[-1,1]区间线性映射到[down_bound, up_bound]"""
        return 0.5 * (self.up_bound - self.down_bound) * x + 0.5 * (self.up_bound + self.down_bound)

    def _back_to_bounds(self, x):
        """将[down_bound, up_bound]区间线性映射到[-1, 1]"""
        return (x - 0.5 * (self.up_bound + self.down_bound))/0.5 / (self.up_bound - self.down_bound)
