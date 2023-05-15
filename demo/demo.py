import gymnasium as gym
import pen_world
import numpy as np
from PPO import PPO
rng = np.random.default_rng()
import time

# create ppo agent and environment
ppo_agent = PPO.load("pen_world_agent.pth")
env = gym.make("PenWorld-v0", ngen=5, n=10, render_mode="human")

observation, info = env.reset()
time.sleep(3)
start_position = env.coordinates
for t in range(1, env.max_ep_length+1):
    # select action with policy
    action_mask = env.get_action_mask()
    action = ppo_agent.select_action(observation, action_mask)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if reward == 1:
        break
time.sleep(3)
env.close()