import gym
from src.Library import Actor, Agent, Critic
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def show_save_fig(Score, Actor_loss, Critic_loss, Figname='NN'):
    """
    储存并展示图片
    :param Figname: 图像名
    :param Score:得分列表(list)
    :param Actor_loss: actor损失列表(list)
    :param Critic_loss: critic损失列表(list)
    :return: None
    """
    fig = plt.figure(figsize=(9, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])  # 3 行 1 列，高度比例 1:1:1

    ax1 = plt.subplot(gs[0])
    ax1.plot(Score, label='Score', color='blue')
    ax1.set_title('Score')

    ax2 = plt.subplot(gs[1])
    ax2.plot(Actor_loss, label='Actor Loss', color='red')
    ax2.set_title('Actor Loss')

    ax3 = plt.subplot(gs[2])
    ax3.plot(Critic_loss, label='Critic Loss', color='green')
    ax3.set_title('Critic Loss')

    plt.tight_layout()
    plt.savefig("data/picture/" + Figname + ".png", dpi=600, bbox_inches='tight')
    plt.show()

    return None


env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name, render_mode=None)
device = torch.device('cpu')
figname = 'ppo_train'

action_space = 2  # 动作空间
state_space = 8  # 状态空间
actor_lr = 0.001  # 学习率(actor)
critic_lr = 0.0008  # 学习率(critic)
ppo_epoch = 6  # 单组数据训练次数
batch_size = 2048  # batch大小
total_episode = 1000  # 总训练数
up_bound = 1.0  # 动作上界
down_bound = -1.0  # 动作下界
dist_type = 'beta'  # 可选择'beta'和'gaussian'

# 实例化Actor, Critic, Agent
actor = Actor.ActorPPO2_continuous(state_space=state_space,
                                   action_space=action_space,
                                   lr=actor_lr,
                                   device=device,
                                   up_bound=up_bound,
                                   down_bound=down_bound,
                                   dist_type='beta')
critic = Critic.Critic_PPO2_continuous(state_space=state_space,
                                       action_space=action_space,
                                       lr=critic_lr,
                                       device=device)
agent = Agent.Agent_PPO2_continuous(Critic=critic,
                                    Actor=actor,
                                    Env=env,
                                    ppo_epoch=ppo_epoch,
                                    batch_size=batch_size,
                                    action_space=action_space,
                                    state_space=state_space,
                                    env_name=env_name,
                                    device=device,
                                    shuffle=True,
                                    up_bound=up_bound,
                                    down_bound=down_bound,
                                    render=False,
                                    dist_type=dist_type)

# 存储训练数据
score, actor_loss, critic_loss = [], [], []
bar = tqdm(range(total_episode))

# 模型训练
for episode in bar:
    avg_actor_loss, avg_critic_loss, avg_episode_score, game_episode = agent.run_episode()
    bar.set_description(
        f"Ep {episode} | "
        f"AvgScore: {avg_episode_score:.1f} | "
        f"AvgActorLoss: {avg_actor_loss:.3f} | "
        f"AvgCriticLoss: {avg_critic_loss:.3f} | "
        f"Game_episode {game_episode} | "
    )
    score.append(avg_episode_score)
    actor_loss.append(avg_actor_loss)
    critic_loss.append(avg_critic_loss)

# 保存模型
torch.save(agent.Critic, "data/model/ppo2_Critic_else.pth")
torch.save(agent.Actor, "data/model/ppo2_Actor_else.pth")

# 绘制训练过程
show_save_fig(Score=score,
              Actor_loss=actor_loss,
              Critic_loss=critic_loss,
              Figname=figname)
