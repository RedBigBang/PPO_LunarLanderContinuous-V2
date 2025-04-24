import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic_PPO2_continuous(nn.Module):
    def __init__(self, state_space, action_space, lr=0.001, device='cpu'):
        super(Critic_PPO2_continuous, self).__init__()
        self.input_shape = state_space
        self.action_space = action_space

        # 网络结构
        self.fc1 = nn.Linear(state_space, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, 1)  # 输出单值V(s)

        # He初始化
        self._initialize_weights()

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        """
        前向传播
        :param state: 输入状态(torch.Tensor)
                      形状[batch_size, state_dim]
        :return: value: prediction(torch.Tensor)
                        形状[batch_size, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc_out(x)  # 无激活函数，输出任意实数
        return value

    @staticmethod
    def critic_PPO2_loss(old_y_pred, Target, y_pred):
        """
        Clipped Value Loss
        :param old_y_pred: 旧价值估计(torch.Tensor)
                          形状[batch_size, 1]
        :param Target: 目标价值(V(s)+GAE或者标准化后的V(s)+GAE)(torch.Tensor)
                      形状[batch_size, 1]
        :param y_pred: Critic当前价值估计(torch.Tensor)
                      形状[batch_size, 1]
        :return: value_loss: 计算得到的损失值(torch.Tensor)
                            形状[1]（标量）
        """
        """LOSS_CLIPPING = 0.2
        clipped_values = old_y_pred + torch.clamp(y_pred - old_y_pred, -LOSS_CLIPPING, LOSS_CLIPPING)

        # 计算两种损失
        v_loss1 = (Target - clipped_values).pow(2)
        v_loss2 = (Target - y_pred).pow(2)

        # 取最大值并平均(保守更新，使critic变化不要过大)
        value_loss = 0.5 * torch.max(torch.min(v_loss1, v_loss2))"""
        value_loss = 0.5 * torch.mean((Target - y_pred).pow(2))
        return value_loss

    def predict(self, state):
        with torch.no_grad():
            return self.forward(state)


