import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from itertools import cycle

pygame.init()

width = 640
height = 480
grid_size = 20
whie = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)
black = (0, 0, 0)
action_space = [(grid_size, 0), (-grid_size, 0), (0, grid_size), (0, -grid_size)]

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return map(np.array, zip(*batch))
    
    def __len__(self):
        return len(self.memory)
    
gamma = 0.99
lr = 0.001
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory_capacity = 100000
target_update_freq = 1000
train_start = 1000

input_dim = 9
output_dim = len(action_space)
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = ReplayMemory(memory_capacity)

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("SnakeRL(Double DQN)")

def get_state(snake, food, direction):
    head_x, head_y = snake[0]
    food_x, food_y = food
    danger_straight = (head_x + direction[0], head_y + direction[1]) in snake or head_x + direction[0] < 0 or head_x + direction[0] >= width or head_y + direction[1] < 0 or head_y + direction[1] >= height
    direction_left = (direction[1], -direction[0])
    danger_left = (head_x + direction_left[0], head_y + direction_left[1]) in snake 
    direction_right = (-direction[1], direction[0])
    danger_right = (head_x + direction_right[0], head_y + direction_right[1]) in snake
    return np.array([danger_straight, danger_left, danger_right, direction[0], direction[1], food_x < head_x, food_x > head_x, food_y < head_y, food_y > head_y])

def select_action(state):
    global epsilon
    if random.random() < epsilon:
        return random.randint(0, output_dim-1)
    else:
        with torch.no_grad():
            return policy_net(torch.tensor(state, dtype=torch.float32)).argmax().item()

def optimize_model():
    if len(memory) < train_start:
        return

    states, actions, rewards, next_state, dones = memory.sample(batch_size)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_state, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(1, keepdim=True)   
        next_q_values = target_net(next_states).gather(1, next_actions).squeeze()
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

step_counter = 0
while True:
    snake = [(width // 2, height // 2)]
    direction = action_space[0]
    food = (random.randrange(0, width, grid_size), random.randrange(0, height, grid_size))
    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill(black)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        state = get_state(snake, food, direction)
        action_index = select_action(state)
        direction = action_space[action_index]
        
        new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        reward = -1000 if new_head in snake or new_head[0] < 0 or new_head[0] >= width or new_head[1] < 0 or new_head[1] >= height else 100
        if new_head == food:
            reward = 10000
            food = (random.randrange(0, width, grid_size), random.randrange(0, height, grid_size))
        else:
            snake.pop()
        snake.insert(0, new_head)
        done = new_head in snake[1:] or new_head[0] < 0 or new_head[0] >= width or new_head[1] >= height

        next_state = get_state(snake, food, direction)
        memory.push(state, action_index, reward, next_state, done)

        optimize_model()

        for segment in snake:
            pygame.draw.rect(screen, green, (segment[0], segment[1], grid_size, grid_size))
        pygame.draw.rect(screen, red, (food[0], food[1], grid_size, grid_size))
        pygame.display.flip()
        clock.tick(10)

        step_counter += 1
        if step_counter % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if done:
            break



