import collections
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from Common import arguements


class ReplayMemory:
        def __init__(self, n_s, n_a) :

            self.n_s = n_s
            self.n_a = n_a
            self.args= arguements.get_args()
            self.MEMORY_SIZE = 1
            self.BATCH_SIZE = 64
            self.all_s = np.empty(shape=(self.MEMORY_SIZE,self.args.num_of_node, 64), dtype=np.float16)
            self.all_a = np.random.randint(low=0, high=self.n_a, size=self.MEMORY_SIZE, dtype=np.int16)
            self.all_r = np.empty(self.MEMORY_SIZE, dtype=np.float64)
            #self.all_done = np.random.randint(low=0, high=2, size=self.MEMORY_SIZE, dtype=np.uint8)
            self.all_s_ = np.empty(shape=(self.MEMORY_SIZE,self.args.num_of_node,64), dtype=np.float16)
            self.count = 0
            self.t = 0

            # self.a1 = np.random.randint(low=0,high=)

        def add_memo(self, s, a, r, s_):
            self.all_s[self.t] = s
            self.all_a[self.t] = a
            self.all_r[self.t] = r
            # self.all_done[self.t] = done
            self.all_s_[self.t] = s_
            self.count = max(self.count, self.t + 1)
            self.t = (self.t + 1) % self.MEMORY_SIZE

        def sample(self):
            if self.count < self.BATCH_SIZE:
                indexes = range(0, self.count)
            else:
                indexes = random.sample(range(0, self.count), self.BATCH_SIZE)

            batch_s = []
            batch_a = []
            batch_r = []
            #batch_done = []
            batch_s_ = []
            for idx in indexes:
                batch_s.append(self.all_s[idx])
                batch_a.append(self.all_a[idx])
                batch_r.append(self.all_r[idx])
                #batch_done.append(self.all_done[idx])
                batch_s_.append(self.all_s_[idx])

            batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
            batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1)
            batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
            batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)
            # batch_s_tensor = torch.as_tensor(batch_s, dtype=torch.float32)
            # batch_a_tensor = torch.as_tensor(batch_a, dtype=torch.int64).unsqueeze(-1)
            # batch_r_tensor = torch.as_tensor(batch_r, dtype=torch.float32).unsqueeze(-1)
            # batch_s__tensor = torch.as_tensor(batch_s_, dtype=torch.float32)

            return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_s__tensor


class ReplayBuffer():
    def __init__(self, capacity):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.buffer = collections.deque(maxlen=capacity)

    # 将数据以元组形式添加进经验池
    def add_memo(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        # print(reward)

    # 随机采样batch_size行数据
    def sample(self):
        transitions = random.sample(self.buffer, min(64,self.size()))
        # transitions = np.random.choice(self.buffer, min(64, self.size()))
        # print(transitions)# list, len=32
        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state = zip(*transitions)
        batch_s_tensor = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(action), dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.asarray(reward), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(next_state), dtype=torch.float32)


        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_s__tensor
        # return np.array(state), action, reward, np.array(next_state)

    # 目前队列长度
    def size(self):
        # print(len(self.buffer))
        return len(self.buffer)



Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'next_obs'])

class ReplayMemory1:
    def __init__(self, capacity):
        self.capacity = capacity # 允许存储多少状态
        self.memory = []  # 存入的状态
        self.position = 0  # memory的list的下标.用于控制经验池的大小。

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        exp = random.sample(self.memory, min(64,self.position))
        return exp


class DQN(nn.Module):

        def __init__(self, n_input, n_output,device):
            super().__init__()  # Reuse the param of nn.Module
            in_features = n_input  # ?
            self.device=device
            self.fc1=nn.Linear(64,32)
            self.fc2=nn.Linear(32,1)
            # nn.Sequential() ?
            # self.net = nn.Sequential(
            #     nn.Linear(in_features, 256),
            #     nn.Tanh(),
            #     nn.Linear(256, n_output))

        def forward(self, x1):
            x1=x1.to(self.device)
            out1=self.fc1(x1)
            out2=F.relu(out1)
            out=self.fc2(out2)

            return out
            # return self.net(x)

        def act(self, to_nodes,obs):

            q_values = self(obs)  # ?
            # print((q_values))
            # max_q_index = torch.argmax(q_values, dim=1)[0]  # ?
            q_values_numpy=q_values.t().squeeze(0).detach().cpu().numpy()
            # print(q_values_numpy)
            # a = q_values_numpy[0]
            add_dic = {}
            to_nodes=np.array(to_nodes).astype(dtype=int).tolist()
            for tovex in to_nodes:
                add_dic.update({tovex:q_values_numpy[tovex]})
            # print(add_dic)

            # print(to_nodes,q_values,q_2,q_values_numpy,max_q_index)

            return add_dic



class Agent:
        def __init__(self, n_input, n_output,device, mode="train"):
            # self.idx = idx
            self.device = device
            self.mode = mode
            self.n_input = n_input
            self.n_output = n_output

            self.GAMMA = 0.99
            self.learning_rate = 1e-3
            # self.MIN_REPLAY_SIZE = 1000
            # self.memo =ReplayBuffer(400)
            self.memo = ReplayMemory(n_s=self.n_input, n_a=self.n_output)
            # self.memo=ReplayMemory1(1000)

            # Initialize the replay buffer of agent i
            if self.mode == "train":
                self.online_net = DQN(self.n_input, self.n_output,self.device).to(device)
                self.target_net = DQN(self.n_input, self.n_output,self.device).to(device)

                self.target_net.load_state_dict(self.online_net.state_dict())  # copy the current state of online_net

                self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
