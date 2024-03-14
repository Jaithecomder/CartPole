import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(29)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

env = gym.make("CartPole-v1")
n_actions = env.action_space.n

state, info = env.reset()
n_observations = len(state)

BATCH_SIZE = 128
GAMMA = 0.99
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 1000
eps = 0.9
TAU = 0.005
LR = 1e-4

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

def select_action(state):
    global eps
    sample = random.random()
    eps -= 1/5000
    if sample > eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
def optimize():
    if len(memory) < BATCH_SIZE :
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[2])), device=device, dtype=torch.bool)

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    next_state_batch = torch.cat([s for s in batch[2] if s is not None])
    reward_batch = torch.cat(batch[3])

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(next_state_batch).max(1)[0]

    expected_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

nEps = 1000

for i in range(nEps) :
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    if (i+1)%100 == 0 :
        print(i+1)

    while True :
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else :
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        memory.push((state, action, next_state, reward))
        state = next_state

        optimize()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done :
            break

torch.save(policy_net, 'agent1.pt')