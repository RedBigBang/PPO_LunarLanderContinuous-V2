import gym
import torch
from src.Library import Agent


class Player():
    def __init__(self):
        """
        使用PPO2_Beta解决LunarlanderContinuous-v2问题
        """
        env_name = 'LunarLanderContinuous-v2'
        env = gym.make(env_name, render_mode='human')
        actor = torch.load('data/model/ppo2_continue_Actor_Beta.pth', weights_only=False)
        critic = torch.load('data/model/ppo2_continue_Critic_Beta.pth', weights_only=False)
        device = torch.device('cpu')
        dis_type = 'beta'
        self.Agent = \
            Agent.Agent_PPO2_continuous(Critic=critic,
                                        Actor=actor,
                                        Env=env,
                                        ppo_epoch=4,
                                        batch_size=2048,
                                        state_space=8,
                                        action_space=2,
                                        env_name=env_name,
                                        device=device,
                                        shuffle=True,
                                        render=False,
                                        dist_type=dis_type,
                                        up_bound=1.0,
                                        down_bound=-1.0)

    def show(self):
        """
        使用模型play一次游戏
        :return: None
        """
        self.Agent.show()
        return None
