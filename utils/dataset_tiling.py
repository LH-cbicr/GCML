"""
This script is used to generate tiling dataset in dataset/ directory
"""
import torch
from torch.utils.data import Dataset, DataLoader  
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
import random
import seaborn as sns
import copy
import torch.nn.functional as F
import h5py
import os

# Set the device to GPU if available
device = torch.device("cpu")

seed = 123456
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

corner = np.zeros((2, 2))
corner[:1, :] = 1
corner[:, :1] = 1

bar1 = np.ones((2, 1))
bar2 = np.ones((1, 2))
bar3 = np.ones((3, 1), dtype=np.int64)
bar4 = np.ones((1, 3), dtype=np.int64)

objects = [
    corner.copy(),
    corner[::-1, :].copy(),
    corner[:, ::-1].copy(),
    corner[::-1, ::-1].copy(),
    bar1.copy(),
    bar2.copy(),
    bar3.copy(),
    bar4.copy() 
]

fig, axs = plt.subplots(1, len(objects), figsize=(15, 15))
for i in range(len(objects)):
    img = np.zeros((11, 11))
    img[1:1+objects[i].shape[0], 1:1+objects[i].shape[1]] = objects[i]
    sns.heatmap(img, square=True, xticklabels=False, yticklabels=False, cmap='Greys', cbar=False, ax=axs[i])

class CompositionEnv:
    def __init__(self, edge=10, objects=objects, min_object_num=8, max_object_num=8, train_dataset_size=10, test_dataset_size=10, dense=False):
        self.edge = edge  # 图片大小
        self.dim = 2  # 图片的维度
        self.objects = objects

        self.min_object_num = min_object_num  
        self.max_object_num = max_object_num

        self.cal_action_to_observation_change()
        self.s_dim = self.edge * self.edge  # 图片大小, 即Qo的维度

    def cal_action_to_observation_change(self):
        self.obj_action_rng = {}  # 保存每个形状的action范围
        action_idx = 0
        action_to_observation_change = {}
        action_to_postion = {}  # action 到位置的映射
        action_to_shape = {}
        for i in range(len(self.objects)):
            height = self.objects[i].shape[0]
            width = self.objects[i].shape[1]
            self.obj_action_rng[i] = [action_idx]
            for h in range(self.edge - height + 1):
                for w in range(self.edge - width + 1):
                    tmp_img = np.zeros((self.edge, self.edge), dtype=np.int64)
                    tmp_img[h:h+height, w:w+width] = self.objects[i]
                    action_to_observation_change[action_idx] = tmp_img
                    action_to_postion[action_idx] = [h, w]
                    action_to_shape[action_idx] = i
                    action_idx += 1
            self.obj_action_rng[i].append(action_idx)
        self.action_to_position = action_to_postion
        self.action_to_shape = action_to_shape
        self.action_to_observation_change = action_to_observation_change
        self.a_dim = action_idx
        print("Total a dim:", self.a_dim)

    def check_affordance(self, cur_state, action_idx):
        assert cur_state.shape == (self.edge, self.edge)
        obs_chg = self.action_to_observation_change[action_idx]
        if torch.all(cur_state[obs_chg == 1] == 1) == True:
            next_state = copy.deepcopy(cur_state)
            next_state[obs_chg == 1] = 0  # 添加了一个形状->将对应位置置为0
            return True, next_state
        else:
            return False, cur_state

    def gain_affordance_vector(self, cur_state):
        aff_vec = torch.zeros((1, self.a_dim), dtype=torch.float32, device=device)
        # print("aff_vec.shape:", aff_vec.shape)
        for i in range(self.a_dim):
            reachable, _ = self.check_affordance(cur_state, i)
            if reachable:
                aff_vec[0, i] = 1.0
        return aff_vec

    def check_arrive(self, cur_state):
        if np.all(cur_state == 0):
            return True
        else:
            return False

    def generate_one_sampling_trajectory(self, obj_seen_idx):
        cur_img = np.zeros((self.edge, self.edge), dtype=np.int64)
        obj_num = np.random.randint(self.min_object_num, self.max_object_num + 1)

        traj_info = {
            "action_index": [],
            "observation": []
        }

        traj_info["observation"].append(cur_img)
        for _ in range(obj_num):
            while True:
                if obj_seen_idx == -1:  
                    action_idx = np.random.randint(0, self.a_dim)
                else:  
                    action_idx = np.random.randint(0, self.obj_action_rng[obj_seen_idx][0])
                obj_chg = self.action_to_observation_change[action_idx]
                if np.all(cur_img[obj_chg == 1] == 0) == False:
                    continue
                else:
                    new_img = copy.deepcopy(cur_img)
                    new_img[obj_chg == 1] = 1
                    traj_info["observation"].append(new_img)
                    traj_info["action_index"].append(action_idx)
                    cur_img = new_img
                    break
        
        traj_info["action_index"].reverse()
        traj_info["observation"].reverse()

        return traj_info

    def check_adjacent_edge(self, obj_idx, cur_img, obj_chg):
        if obj_idx == 0:
            return False
        idx = np.where(obj_chg > 0)
        direc = [
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0]
        ]
        coincidence_edge = 0
        for i in range(len(idx[0])):
            x = idx[0][i]
            y = idx[1][i]
            for d in direc:
                x_ = x + d[0]
                y_ = y + d[1]
                if x_ >= self.edge or y_ >= self.edge or x_ < 0 or y_ < 0:
                    continue
                if cur_img[x_, y_] > 0:
                    coincidence_edge += 1
                    # return False
                if coincidence_edge > 0:  # 至少有一个像素点相邻
                    return False

        return True

    def generate_one_sampling_trajectory_dense(self, obj_seen_idx):
        cur_img = np.zeros((self.edge, self.edge), dtype=np.int64)
        obj_num = np.random.randint(self.min_object_num, self.max_object_num + 1)

        traj_info = {
            "action_index": [],
            "observation": []
        }

        traj_info["observation"].append(cur_img)
        for obj_idx in range(obj_num):
            while True:
                if obj_seen_idx == -1:
                    action_idx = np.random.randint(0, self.a_dim)
                else: 
                    action_idx = np.random.randint(0, self.obj_action_rng[obj_seen_idx][0])
                obj_chg = self.action_to_observation_change[action_idx]
                if np.all(cur_img[obj_chg == 1] == 0) == False:
                    continue
                elif self.check_adjacent_edge(obj_idx, cur_img, obj_chg):
                    continue
                else:
                    new_img = copy.deepcopy(cur_img)
                    new_img[obj_chg == 1] = 1
                    traj_info["observation"].append(new_img)
                    traj_info["action_index"].append(action_idx)
                    cur_img = new_img
                    break
        
        traj_info = self.reorder_action_sequence(traj_info)
        
        traj_info["action_index"].reverse()
        traj_info["observation"].reverse()
        traj_info["element_index"].reverse()

        return traj_info
    
    def reorder_action_sequence(self, traj_info):
        pos = []
        for action in traj_info["action_index"]:
            pos.append(self.action_to_position[action])
        pos = np.array(pos)
        sorted_index = np.random.permutation(pos.shape[0])
        sorted_data = pos[sorted_index]

        cur_img = copy.deepcopy(traj_info["observation"][0])
        new_traj_info = {}
        new_traj_info["observation"] = []
        new_traj_info["action_index"] = []
        new_traj_info["element_index"] = []
        new_traj_info["observation"].append(cur_img)
        for obj_idx in range(len(traj_info["action_index"])):
            action_idx = traj_info["action_index"][sorted_index[obj_idx]]  # 排序后的action_index
            obj_chg = self.action_to_observation_change[action_idx]
            new_img = copy.deepcopy(cur_img)
            new_img[obj_chg == 1] = 1
            new_traj_info["observation"].append(new_img)
            new_traj_info["action_index"].append(action_idx)
            new_traj_info["element_index"].append(self.action_to_shape[action_idx])
            cur_img = new_img
        
        return new_traj_info

    def generate_dataset(self, data_num, obj_seen_idx=-1, dense=False):
        dataset = []
        for i in range(data_num):
            if i % 1000 == 0:
                print(i, "/", data_num)
            if dense == True:
                dataset.append(self.generate_one_sampling_trajectory_dense(obj_seen_idx))
            else:
                dataset.append(self.generate_one_sampling_trajectory(obj_seen_idx))
        return dataset

class CompositionEnvDataset(Dataset):
    def __init__(self, env, dataset_size, obj_seen_idx=-1, dense=False):
        data = env.generate_dataset(dataset_size, obj_seen_idx, dense=dense)
        self.data = []
        for i in range(len(data)):
            action_num = len(data[i]["observation"]) - 1
            traj = []
            for j in range(action_num):
                obs = torch.tensor(data[i]["observation"][j].reshape((-1,)), dtype=torch.float32, device=device)
                action_idx = torch.tensor(data[i]["action_index"][j], dtype=torch.float32, device=device)
                element_idx = torch.tensor(data[i]["element_index"][j], dtype=torch.float32, device=device)
                next_obs = torch.tensor(data[i]["observation"][j+1].reshape((-1,)), dtype=torch.float32, device=device)
                traj.append((obs, action_idx, element_idx, next_obs))
            self.data.append(traj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

composition_env = CompositionEnv(dense=True, min_object_num=5, max_object_num=5)  # 5 BBs
# composition_env = CompositionEnv(dense=True, min_object_num=8, max_object_num=8)  # 8 BBs
composition_env_dataset = CompositionEnvDataset(composition_env, 20000, dense=True)

demo_data = composition_env_dataset.data[0]
data_num = len(composition_env_dataset.data)
traj_len = len(demo_data)
data_size = demo_data[0][0].shape[0]
pre_obs = np.zeros((data_num, traj_len, data_size))
post_obs = np.zeros((data_num, traj_len, data_size))
action = np.zeros((data_num, traj_len))
element_idx = np.zeros((data_num, traj_len))
for i in tqdm(range(data_num), desc="data_num"):
    for j in range(traj_len):
        pre_obs[i, j, :] = composition_env_dataset.data[i][j][0].numpy()
        action[i, j] = composition_env_dataset.data[i][j][1].numpy()
        element_idx[i, j] = composition_env_dataset.data[i][j][2].numpy()
        post_obs[i, j, :] = composition_env_dataset.data[i][j][3].numpy()

data_dir = "/home/linhui/linhui21/cml/dataset/"  # Change to project directory 
with h5py.File(os.path.join(data_dir, 'tiling_10x10_5obj.h5'), 'w') as f:
    single = f.create_group('tiling')
    single.create_dataset('pre_obs', data=pre_obs, compression='gzip', chunks=(100, traj_len, data_size))
    single.create_dataset('post_obs', data=post_obs, compression='gzip', chunks=(100, traj_len, data_size))
    single.create_dataset('action', data=action, compression='gzip', chunks=(100, traj_len))
    single.create_dataset('element_idx', data=element_idx, compression='gzip', chunks=(100, traj_len))
