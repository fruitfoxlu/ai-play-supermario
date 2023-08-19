import numpy as np
import tensorflow as tf
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT
from dqnagent import DQNAgent  

# actions for very simple movement
MY_SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['A','A','A'],
    ['A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A'],
    ['A','right', 'A'],
    ['left']
]

# actions for more complex movement
MY_COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]

# Create the Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, MY_SIMPLE_MOVEMENT)  # Use SIMPLE_MOVEMENT to reduce action space

# Create the DQN agent
state_size = (240, 256, 3)  #  240x256X3 for each frame
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)  # You might need to modify DQNAgent to handle image input

# Training loop
episodes = 1000
batch_size = 16
for e in range(episodes):
    state = env.reset()
    agent.new_lesson(state)

    total_reward = 0
    done = False

    
    while not done:
        env.render() # This will display the game screen

        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
       
        agent.remember(state, action, info,reward, next_state, done)
        
        state = next_state
        total_reward += reward
        if done:
            print(f"Episode: {e}/{episodes}, Score: {total_reward}")
            break
    if len(agent.memory) > batch_size: #kick-off learning after one episodes
        agent.replay(batch_size) #update the NN by so far what we learn
