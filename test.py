import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

policy_net = torch.load('agent1.pt')

env = gym.make('CartPole-v1', render_mode='human')
# env = gym.make('CartPole-v1')
# env = gym.wrappers.Monitor(env, "record_dir")
for i in range(2):
    obs, _ = env.reset()
    done, rew = False, 0
    while (done != True) :
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            A = policy_net(state).max(1)[1].view(1, 1)
        obs, reward, term, trunc, info = env.step(A.item())

        done = term
        
        rew += reward
        if rew % 200 == 0:
            print(rew)
        # sleep(0.01)
        env.render()
    print("episode : {}, reward : {}".format(i,rew)) 