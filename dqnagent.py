import numpy as np
import tensorflow as tf
import gym
import copy


class DQNAgent:
    def __init__(self, input_shape, action_size):
        self.input_shape = input_shape
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.prev_x_position = 0
        self.prev_y_position = 0
        self.biggest_jump_x = 0
        self.biggest_jump_y = 0
        #self.new_lesson()  # Initialize to an invalid value


    def _build_model(self):
        model = tf.keras.models.Sequential()
        
        # Convolutional layers
        model.add(tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), padding='valid', activation='relu', input_shape=self.input_shape))
        model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='valid', activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))

        # Pooling layer to reduce spatial dimensions
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        
        # Fully connected layers
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def new_lesson(self,state):
        self.prev_x_position = -1  # Initialize to an invalid value
        # Initialization before starting the episode
        self.state_buffer = [state]*4
        self.stacked_buffer = np.stack(self.state_buffer, axis=0)
        
   
    def remember(self, state, action, info,reward, next_state, done):
        shapped_reward = self.reward_shaping(info,reward)
        old_buffer = copy.deepcopy(self.stacked_buffer)
       
        # Update the buffer with the new state and drop the oldest state
        self.state_buffer.pop(0)
        self.state_buffer.append(next_state)

        # Stack the states from the buffer to form the input for the neural network
        self.stacked_buffer = np.stack(self.state_buffer, axis=0)
        #print("last 4 stack")
        #print(self.stacked_buffer.shape)
        
        self.memory.append((old_buffer,action, shapped_reward, self.stacked_buffer , done))

   

    def predict_with_current_frame(self):
        
        return self.model.predict(self.stacked_buffer)

    def predict_with_state(self,state):
            
        return self.model.predict(state) 
        

    def act(self, state):
        #chose random actions at the first few rounds of the training 
       
        if np.random.rand() <= self.epsilon:
             act_values = np.random.randint(self.action_size)
             #print(f"random act {act_values} {np.random.rand()} {self.epsilon}")
             return act_values
       
        act_values =  self.predict_with_current_frame()
        print(f"act {act_values}")
        return np.argmax(act_values[0])

    def reward_shaping(self,info,reward):
        # If Mario's x position hasn't changed, modify the reward
        #print(info)
        if self.prev_x_position <= info['x_pos']:
             #print("prize for not moving forward")
             reward += 0.2  # Penalize for not moving forward
        if info['y_pos'] - self.prev_y_position > 29:
             print("prize for jumping high")
             reward += 0.2
        jump_high = info['y_pos'] - self.prev_y_position
        if (jump_high > self.biggest_jump_y):
            print("highest jump %d"%jump_high)
            self.biggest_jump_y = jump_high
        self.prev_x_position = info['x_pos']
        self.prev_y_position = info['y_pos']

        return reward

    def replay(self, batch_size):
        minibatch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        minibatch = [self.memory[index] for index in minibatch_indices] 
        for state, action, reward, next_state, done in minibatch:
            if (next_state.shape[1:] != (240, 256,3) or state.shape[1:] != (240, 256,3)):
                #print(f"replay : Skipped prediction due to shape mismatch: {next_state.shape}")
                continue 
            
            target = reward
            # Check the shape of next_state
            if not done :
                #print("replay")
                #print(next_state.shape)
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            #here we update the current predict action to the new action (target??) 
            target_f = self.predict_with_state(state)
            target_f[0][action] = target
            self.model.train_on_batch(state, target_f)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
