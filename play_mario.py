import numpy as np
import tensorflow as tf
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from dqnagent import DQNAgent  # Assuming you have the DQNAgent class from the previous example

# Create the Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)  # Use SIMPLE_MOVEMENT to reduce action space

# Create the DQN agent
state_size = (240, 256, 4)  # 4 stacked frames of 240x256 pixels
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)  # You might need to modify DQNAgent to handle image input

# Training loop
episodes = 1000
batch_size = 32
for e in range(episodes):
    state = env.reset()
    state = np.stack([state] * 4, axis=2)  # Stack 4 frames
    total_reward = 0
    done = False
    while not done:
        env.render() # This will display the game screen
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        #next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.stack([next_state] * 4, axis=2)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            print(f"Episode: {e}/{episodes}, Score: {total_reward}")
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
