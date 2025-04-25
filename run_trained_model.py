from src.Run.Player_Beta import Player

# 模型训练参数:
"""
env_name = 'LunarLanderContinuous-v2'
action_space = 2  # 动作空间
state_space = 8  # 状态空间
actor_lr = 0.001  # 学习率(actor)
critic_lr = 0.0008  # 学习率(critic)
ppo_epoch = 6  # 单组数据训练次数
batch_size = 2048  # batch大小
total_episode = 5000  # 总训练数
up_bound = 1.0  # 动作上界
down_bound = -1.0  # 动作下界
dist_type = 'beta'  # 可选择'beta'和'gaussian'
"""

player = Player()
while True:
    # 模型效果展示
    player.show()
