import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from scipy.spatial.transform import Rotation as R

class Map:
    def __init__(self, blocks_num, random_seed) -> None:
        self.size = np.array([100, 100, 100])
        self.dsize = 1 / self.size
        self.dagsize = 1 / np.linalg.norm(self.size)
        self.random_seed = random_seed
        self.blocks_num = blocks_num
        self.point_size = 1
        self.init_size = self.point_size * 8
        self._random_map(random_seed)
        return

    def _random_map(self, random_seed) -> None:
        np.random.seed(random_seed)
        self.blocks = np.zeros((self.blocks_num, 3))
        self.BEGIN = np.random.rand(3) * self.size
        self.END = np.random.rand(3) * self.size

        i = 0
        while i < self.blocks_num:
            point = np.random.rand(3) * self.size
            fitted = True
            d1 = np.linalg.norm(point-self.BEGIN)
            d2 = np.linalg.norm(point-self.END)
            if d1 <= self.init_size or d2 <= self.init_size:
                fitted = False

            for j in range(i):
                d = np.linalg.norm(point-self.blocks[j])
                if d <= self.init_size:
                    fitted = False

            if fitted:
                self.blocks[i] = point
                i += 1

        self.vd_blocks = self.blocks.flatten()
        return
    
    def reRandom(self):
        fitted = False
        while not fitted:
            fitted = True
            self.BEGIN = np.random.rand(3) * self.size
            if np.min(np.linalg.norm(self.blocks - self.BEGIN, axis=1)) <= self.init_size:
                fitted = False
        
        fitted = False
        while not fitted:
            fitted = True
            self.END = np.random.rand(3) * self.size
            if np.linalg.norm(self.END - self.BEGIN) <= self.init_size:
                fitted = False
            if np.min(np.linalg.norm(self.blocks - self.END, axis=1)) <= self.init_size:
                fitted = False

        return
    



class Agent:
    pos = np.zeros(3)
    vel = np.zeros(3)

    def __init__(self, Map) -> None:
        self.map = Map
        self.max_step = 300
        self.reset()
        return

    def reset(self) -> None:
        self.pos = self.map.BEGIN.copy()
        self.vel = np.zeros(3)
        self.steps = 0
        self.old_dst = np.linalg.norm(self.map.END - self.pos)
        self.old_vel = np.zeros(3)
        return

    def step(self, act):
        Done = False

        self.steps += 1
        if self.steps >= self.max_step: Done = True

        tmp_v = self.vel.copy()
        tmp_p = self.pos.copy()
        

        dir = R.from_euler(
            'zyx',
            np.array([np.pi*act[1], np.pi*0.5*act[2], 0])).apply(np.array([1, 0, 0]))

        f = act[0] + 1
        self.vel = dir * f *0.2 + self.vel *0.8
        self.pos += self.vel - (np.random.rand(3) *2-1) *0.02

        target = np.linalg.norm(self.map.END - self.pos)
        R1 = - target *self.map.dagsize
        self.old_dst = target.copy()
        self.old_vel = self.vel.copy()

        R2 = f *-0.001

        R3 = 0
        if target <= self.map.point_size:
            R3 = 10
            Done = True

        len = np.min(np.linalg.norm(self.map.blocks - self.pos, axis=1))
        R4 = len *self.map.dagsize *-0.002
        if len <= self.map.point_size:
            R4 -= 10
            # Done = True
            
        if (self.pos < 0).any() or (self.pos > self.map.size).any():
            R4 -= 0.1
            self.pos = tmp_p.copy()
            self.vel = tmp_v.copy()

        return R1+R2+R3+R4, Done
    
    def get_state(self):
        tmp = np.multiply(self.map.blocks, self.map.dsize)
        np.random.shuffle(tmp)
        return np.concatenate((
            tmp.flatten(),
            self.pos *self.map.dsize,
            self.vel *self.map.dsize,
            self.old_dst *self.map.dsize,
            self.old_vel *self.map.dsize,
            self.map.END *self.map.dsize
        ))
    


class Actor_Net(nn.Module):
    def __init__(self, dim):
        super(Actor_Net, self).__init__()
        self.liner1 = nn.Linear(dim, 512)
        self.liner2 = nn.Linear(512, 512)
        self.liner3 = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.liner1(x))
        x = F.relu(self.liner2(x))
        return torch.tanh(self.liner3(x))
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
actor_net=torch.load('actor_net160000.pt', map_location=device)

seed=1111
AMap=Map(20, seed)
agent=Agent(AMap)

agent.reset()
episode_reward = 0
positions = [agent.pos.copy()]

for _ in trange(agent.max_step):
    state1 = agent.get_state()

    act = actor_net(torch.FloatTensor(state1).to(device))
    act = act.cpu().detach().numpy()

    reward, Done = agent.step(act)
    episode_reward += reward

    positions.append(agent.pos.copy())  # 存储位置

positions = np.array(positions)

# 绘制智能体路径
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Agent Path')
ax.scatter(agent.map.BEGIN[0], agent.map.BEGIN[1], agent.map.BEGIN[2], color='b', label='Begin', s=20)
ax.scatter(agent.map.END[0], agent.map.END[1], agent.map.END[2], color='r', label='Goal', s=20)
ax.scatter(agent.map.blocks[:, 0], agent.map.blocks[:, 1], agent.map.blocks[:, 2], color='k', label='Obstacle', s=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.title('Test seed: {} loss: {:.3f}'.format(seed, episode_reward))
plt.show()