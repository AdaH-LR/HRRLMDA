
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
import torch.nn as nn
import torch.optim as optim

class ReplayMemory:
    def __init__(self, capacity,device):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.device = device


    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, BATCH_SIZE):
        batch = random.sample(self.memory, BATCH_SIZE)          
        states, actions, rewards, next_states, dones = zip(*batch)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 6)

    def forward(self, x):
        num_tensors = len(x)
        outputs = []
        for i in range(num_tensors):
            x1 = self.flatten(x[i])
            x1 = x1.unsqueeze(0)
            x1 = x1.view(x1.size(0), -1)
            shape = x1.shape[-1]
            self.fc1 = nn.Linear(shape, 64)
            x1 = self.fc1(x1)
            x1 = self.relu(x1)
            x1 = self.fc2(x1)
            x1 = self.relu(x1)
            x1 = self.output(x1)
            if i <= 0:
                outputs = x1
            elif 0 < i <= 1:
                outputs = torch.stack((outputs, x1), dim=0)
            else:
                x1 = x1.unsqueeze(0)
                outputs = torch.cat((outputs, x1), dim=0)
        outputs = outputs.squeeze(1)
        return outputs

class DQNAgent:
    episode_rewards = []
    loss_value = 0
    def __init__(self, nb_states, nb_actions, REPLAY_MEMORY_SIZE, BATCH_SIZE, DISCOUNT, LEARNING_RATE,EPI_START, EPI_END, epsilon_decay,device):
        self.replay_memory_size = REPLAY_MEMORY_SIZE
        self.learning_rate = LEARNING_RATE
        self.nb_states = nb_states
        self.nb_actions = nb_actions  
        self.BATCH_SIZE = BATCH_SIZE
        self.discount = DISCOUNT
        self.epsilon = EPI_START
        self.episilon_end = EPI_END
        self.epsilon_decay = epsilon_decay

        self.device = device
        self.policy_net = DQN(nb_states, nb_actions).to(self.device)
        self.target_net = DQN(nb_states, nb_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(self.replay_memory_size,self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):     
        if random.random() < self.epsilon:
            return random.randrange(self.nb_actions)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def update_epsilon(self):               
        self.epsilon = max(self.episilon_end, self.epsilon * self.epsilon_decay)

    def push_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update_model(self):
        if len(self.memory) < self.replay_memory_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)
        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (~dones) * self.discount * next_q_values
        loss = self.loss_fn(q_values, expected_q_values)
        self.loss_value = loss.item()
        self.optimizer.zero_grad()
        self.optimizer.step()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        
        
