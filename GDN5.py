import os
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
from models import *
import pickle
from collections import deque
from torch.distributions import Categorical
import numpy as np
from GLCN.GLCN2 import GLCN, GAT2, GAT, GIN
from cfg import get_cfg

cfg = get_cfg()
from GAT.layers import device
from copy import deepcopy


class Replay_Buffer:
    def __init__(self, buffer_size, batch_size, num_agent, h=1):
        self.buffer = deque()
        self.step_count_list = list()
        for _ in range(11):
            self.buffer.append(deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.num_agent = num_agent
        self.agent_id = np.eye(self.num_agent).tolist()
        self.batch_size = batch_size
        self.step_count = 0
        self.h = h

    def pop(self):
        self.buffer.pop()

    def memory(self, node_feature, action, action_feature, edge_index_enemy, edge_index_ally, reward, done,
               avail_action, dead_masking, agent_feature, sum_state):
        self.buffer[0].append(node_feature)
        self.buffer[1].append(action)
        self.buffer[2].append(action_feature)
        self.buffer[3].append(edge_index_enemy)
        self.buffer[4].append(edge_index_ally)
        self.buffer[5].append(reward)
        self.buffer[6].append(done)
        self.buffer[7].append(avail_action)
        self.buffer[8].append(dead_masking)
        self.buffer[9].append(agent_feature)
        self.buffer[10].append(sum_state)
        if self.step_count < self.buffer_size - 1:
            self.step_count_list.append(self.step_count)
            self.step_count += 1

    def save_buffer(self):
        buffer_dict = {'buffer': self.buffer, 'step_count_list': self.step_count_list, 'step_count': self.step_count}
        with open('deque.pkl', 'wb') as f:
            pickle.dump(buffer_dict, f)

    def load_buffer(self):
        with open('deque.pkl', 'rb') as f:
            loaded_d = pickle.load(f)
            self.buffer = loaded_d['buffer']
            self.step_count_list = loaded_d['step_count_list']
            self.step_count = loaded_d['step_count']

    def generating_mini_batch(self, datas, batch_idx, cat):
        for s in batch_idx:
            if cat == 'node_feature':
                yield datas[0][s]
            if cat == 'node_feature_next':
                yield datas[0][s + 1]

            if cat == 'action':
                yield datas[1][s]

            if cat == 'action_feature':
                yield datas[2][s]
            if cat == 'action_feature_next':
                yield datas[2][s + 1]

            if cat == 'edge_index_enemy':
                yield datas[3][s]
            if cat == 'edge_index_enemy_next':
                yield datas[3][s + 1]

            if cat == 'edge_index_ally':
                yield datas[4][s]
            if cat == 'edge_index_ally_next':
                yield datas[4][s + 1]

            if cat == 'reward':
                yield datas[5][s]

            if cat == 'done':
                yield datas[6][s]

            if cat == 'avail_action_next':
                yield datas[7][s + 1]

            if cat == 'dead_masking':
                yield datas[8][s]

            if cat == 'dead_masking_next':
                yield datas[8][s + 1]

            if cat == 'agent_feature':
                yield datas[9][s]
            if cat == 'agent_feature_next':
                yield datas[9][s + 1]

            if cat == 'sum_state':
                yield datas[10][s]
            if cat == 'sum_state_next':
                yield datas[10][s + 1]

    def sample(self):
        step_count_list = self.step_count_list[:]
        step_count_list.pop()

        sampled_batch_idx = random.sample(step_count_list, self.batch_size)

        node_feature = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature')
        node_features = list(node_feature)

        action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action')
        actions = list(action)

        action_feature = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action_feature')
        action_features = list(action_feature)

        edge_index_enemy = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_enemy')
        edge_indices_enemy = list(edge_index_enemy)

        edge_index_ally = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_ally')
        edge_indices_ally = list(edge_index_ally)

        reward = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='reward')
        rewards = list(reward)

        done = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='done')
        dones = list(done)

        node_feature_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_next')
        node_features_next = list(node_feature_next)

        action_feature_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action_feature_next')
        action_features_next = list(action_feature_next)

        edge_index_enemy_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_enemy_next')
        edge_indices_enemy_next = list(edge_index_enemy_next)

        edge_index_ally_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_ally_next')
        edge_indices_ally_next = list(edge_index_ally_next)
        avail_action_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_action_next')
        avail_actions_next = list(avail_action_next)

        dead_masking = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='dead_masking')
        dead_masking = list(dead_masking)

        dead_masking_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='dead_masking_next')
        dead_masking_next = list(dead_masking_next)

        agent_feature = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='agent_feature')
        agent_feature = list(agent_feature)

        agent_feature_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='agent_feature_next')
        agent_feature_next = list(agent_feature_next)

        sum_state = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='sum_state')
        sum_state = list(sum_state)

        sum_state_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='sum_state_next')
        sum_state_next = list(sum_state_next)

        return node_features, \
               actions, \
               action_features, \
               edge_indices_enemy, \
               edge_indices_ally, \
               rewards, \
               dones, node_features_next, action_features_next, edge_indices_enemy_next, edge_indices_ally_next, avail_actions_next, dead_masking, \
               dead_masking_next, agent_feature, agent_feature_next, sum_state, sum_state_next


class Agent(nn.Module):
    def __init__(self,
                 num_agent,
                 num_enemy,
                 feature_size,

                 hidden_size_obs,
                 hidden_size_comm,
                 hidden_size_action,
                 hidden_size_Q,

                 n_representation_obs,
                 n_representation_comm,
                 n_representation_action,

                 graph_embedding,
                 graph_embedding_comm,

                 buffer_size,
                 batch_size,
                 learning_rate,
                 learning_rate_graph,
                 gamma,
                 gamma1,
                 gamma2,
                 anneal_episodes_graph_variance,
                 min_graph_variance,
                 env
                 ):
        torch.manual_seed(81)
        random.seed(81)
        np.random.seed(81)
        super(Agent, self).__init__()
        self.num_agent = num_agent
        self.num_enemy = num_enemy
        self.feature_size = feature_size
        self.hidden_size_obs = hidden_size_obs
        self.hidden_size_comm = hidden_size_comm
        self.hidden_size_action = hidden_size_action

        self.n_representation_obs = n_representation_obs
        self.n_representation_comm = n_representation_comm
        self.n_representation_action = n_representation_action

        self.graph_embedding = graph_embedding
        self.graph_embedding_comm = graph_embedding_comm

        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.gamma = gamma
        self.agent_id = np.eye(self.num_agent).tolist()

        self.max_norm = 10

        self.VDN = VDN().to(device)
        self.VDN_target = VDN().to(device)
        self.VDN_target.load_state_dict(self.VDN.state_dict())
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = Replay_Buffer(self.buffer_size, self.batch_size, self.num_agent)

        self.anneal_episodes_graph_variance = anneal_episodes_graph_variance
        self.min_graph_variance = min_graph_variance

        self.node_representation = NodeEmbedding(feature_size=self.feature_size,
                                                 hidden_size=self.hidden_size_obs,
                                                 n_representation_obs=self.n_representation_obs).to(device)  # 수정사항

        self.node_representation_tar = NodeEmbedding(feature_size=self.feature_size,
                                                     hidden_size=self.hidden_size_obs,
                                                     n_representation_obs=self.n_representation_obs).to(device)  # 수정사항

        self.node_representation_comm = NodeEmbedding(feature_size=2 * self.feature_size + 5 - 1,
                                                      hidden_size=self.hidden_size_comm,
                                                      n_representation_obs=self.n_representation_comm).to(device)
        self.node_representation_comm_tar = NodeEmbedding(feature_size=2 * self.feature_size + 5 - 1,
                                                          hidden_size=self.hidden_size_comm,
                                                          n_representation_obs=self.n_representation_comm).to(device)

        if env == 'pp':
            self.action_representation = NodeEmbedding(feature_size=5,
                                                       hidden_size=self.hidden_size_action,
                                                       n_representation_obs=self.n_representation_action).to(
                device)  # 수정사항
        else:
            self.action_representation = NodeEmbedding(feature_size=self.feature_size + 5,
                                                       hidden_size=self.hidden_size_action,
                                                       n_representation_obs=self.n_representation_action).to(
                device)  # 수정사항

            self.action_representation_tar = NodeEmbedding(feature_size=self.feature_size + 5,
                                                           hidden_size=self.hidden_size_action,
                                                           n_representation_obs=self.n_representation_action).to(
                device)  # 수정사항

        self.func_obs = GAT(feature_size=self.n_representation_obs, graph_embedding_size=self.graph_embedding).to(
            device)
        self.func_obs_tar = GAT(feature_size=self.n_representation_obs, graph_embedding_size=self.graph_embedding).to(
            device)




        self.func_comm = GAT(feature_size=self.graph_embedding,graph_embedding_size=self.graph_embedding_comm).to(device)
        self.func_comm_tar = GAT(feature_size=self.graph_embedding,graph_embedding_size=self.graph_embedding_comm).to(device)

        self.func_comm2 = GAT(feature_size=self.graph_embedding_comm, graph_embedding_size=self.graph_embedding_comm).to(device)
        self.func_comm2_tar = GAT(feature_size=self.graph_embedding_comm, graph_embedding_size=self.graph_embedding_comm).to(device)

        self.func_glcn = GLCN(feature_size=self.graph_embedding,
                              feature_obs_size=self.graph_embedding,
                              graph_embedding_size=self.graph_embedding_comm).to(device)

        self.func_glcn_tar = GLCN(feature_size=self.graph_embedding,
                                  feature_obs_size=self.graph_embedding,
                                  graph_embedding_size=self.graph_embedding_comm).to(device)

        print(self.graph_embedding_comm + self.n_representation_action)
        self.Q = Network(self.graph_embedding_comm + self.graph_embedding + n_representation_comm+ self.n_representation_action, hidden_size_Q).to(device)
        self.Q_tar = Network(self.graph_embedding_comm + self.graph_embedding + n_representation_comm+ self.n_representation_action, hidden_size_Q).to(device)

        self.C = Network(self.graph_embedding + self.n_representation_comm + self.n_representation_action,
                         hidden_size_Q).to(device)
        self.C_tar = Network(self.graph_embedding + self.n_representation_comm + self.n_representation_action,
                             hidden_size_Q).to(device)

        self.node_representation_tar.load_state_dict(self.node_representation.state_dict())
        self.node_representation_comm_tar.load_state_dict(self.node_representation_comm.state_dict())
        self.action_representation_tar.load_state_dict(self.action_representation.state_dict())
        self.func_obs_tar.load_state_dict(self.func_obs.state_dict())
        self.func_glcn_tar.load_state_dict(self.func_glcn.state_dict())
        self.Q_tar.load_state_dict(self.Q.state_dict())
        self.C_tar.load_state_dict(self.C.state_dict())
        self.func_comm_tar.load_state_dict(self.func_comm.state_dict())
        self.func_comm2_tar.load_state_dict(self.func_comm2.state_dict())
        self.eps_clip = 1000
        self.original_loss = None
        self.eval_params = list(self.func_glcn.parameters()) + \
                           list(self.VDN.parameters()) + \
                           list(self.Q.parameters()) + \
                           list(self.C.parameters()) + \
                           list(self.node_representation.parameters()) + \
                           list(self.node_representation_comm.parameters()) + \
                           list(self.func_obs.parameters()) + \
                           list(self.func_comm.parameters()) + \
                           list(self.func_comm2.parameters()) + \
                           list(self.action_representation.parameters())
        param_groups = [
            {'params': self.eval_params},
        ]
        self.optimizer = optim.Adam(param_groups, lr=learning_rate)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_ratio)

    def save_model(self, file_dir, e, t, win_rate):
        torch.save({

            "1": self.Q.state_dict(),
            "2": self.Q_tar.state_dict(),
            "3": self.func_glcn.state_dict(),
            "4": self.func_obs.state_dict(),
            "5": self.action_representation.state_dict(),
            "6": self.node_representation_comm.state_dict(),
            "7": self.node_representation.state_dict(),
            "8": self.node_representation_tar.state_dict(),
            "9": self.node_representation_comm_tar.state_dict(),
            "10": self.action_representation_tar.state_dict(),
            "11": self.func_obs_tar.state_dict(),
            "12": self.func_glcn_tar.state_dict(),
            "13": self.C.state_dict(),
            "14": self.C_tar.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        },
            file_dir + "episode{}_t_{}_win_{}.pt".format(e, t, win_rate))

    def load_model(self, path):
        try:
            checkpoint = torch.load(path)
            self.Q.load_state_dict(checkpoint["1"])
            self.Q_tar.load_state_dict(checkpoint["1"])
            self.func_glcn.load_state_dict(checkpoint["3"])
            self.func_glcn_tar.load_state_dict(checkpoint["3"])
            self.func_obs.load_state_dict(checkpoint["4"])
            self.func_obs_tar.load_state_dict(checkpoint["4"])
            self.action_representation.load_state_dict(checkpoint["5"])
            self.action_representation_tar.load_state_dict(checkpoint["5"])
            self.node_representation_comm.load_state_dict(checkpoint["6"])
            self.node_representation_comm_tar.load_state_dict(checkpoint["6"])
            self.node_representation.load_state_dict(checkpoint["7"])
            self.node_representation_tar.load_state_dict(checkpoint["7"])

            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except KeyError as e:
            print(f"Missing key in state_dict: {e}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")


    def get_node_representation_rollout(self, node_feature, agent_feature, edge_index_obs, n_agent,
                                     mini_batch=False, target=False, A_old=None):
        if mini_batch == False:
            with torch.no_grad():
                node_feature = torch.tensor(node_feature, dtype=torch.float, device=device).unsqueeze(0)
                agent_feature = torch.tensor(agent_feature, dtype=torch.float, device=device).unsqueeze(0)
                batch_size = node_feature.shape[0]
                num_nodes = node_feature.shape[1]
                num_agents = agent_feature.shape[1]
                node_feature = node_feature.reshape(batch_size * num_nodes, -1)
                agent_feature = agent_feature.reshape(batch_size * num_agents, -1)
                node_embedding_obs = self.node_representation(node_feature)
                node_embedding_comm = self.node_representation_comm(agent_feature)

                node_embedding_obs = node_embedding_obs.reshape(batch_size, num_nodes, -1)
                node_embedding_comm = node_embedding_comm.reshape(batch_size, num_agents, -1)
                edge_index_obs = torch.tensor(edge_index_obs).long().to(device).unsqueeze(0)
                node_embedding_obs = self.func_obs(X=node_embedding_obs, A=edge_index_obs)[:, :n_agent, :]
                cat_embedding = node_embedding_obs
                A_new, logits = self.func_glcn(cat_embedding, rollout = True)
                cat_embedding = self.func_comm(X=cat_embedding, A=A_new, dense=True)
                cat_embedding = torch.cat([cat_embedding, node_embedding_obs, node_embedding_comm], dim=2)

                return cat_embedding, A_new

    def get_node_representation_temp(self, node_feature, agent_feature, edge_index_obs, n_agent,
                                     mini_batch=False, target=False, A_old=None):
        if mini_batch == False:
            with torch.no_grad():
                node_feature = torch.tensor(node_feature, dtype=torch.float, device=device).unsqueeze(0)
                agent_feature = torch.tensor(agent_feature, dtype=torch.float, device=device).unsqueeze(0)
                batch_size = node_feature.shape[0]
                num_nodes = node_feature.shape[1]
                num_agents = agent_feature.shape[1]
                node_feature = node_feature.reshape(batch_size * num_nodes, -1)
                agent_feature = agent_feature.reshape(batch_size * num_agents, -1)
                node_embedding_obs = self.node_representation(node_feature)
                node_embedding_comm = self.node_representation_comm(agent_feature)

                node_embedding_obs = node_embedding_obs.reshape(batch_size, num_nodes, -1)
                node_embedding_comm = node_embedding_comm.reshape(batch_size, num_agents, -1)
                edge_index_obs = torch.tensor(edge_index_obs).long().to(device).unsqueeze(0)
                node_embedding_obs = self.func_obs(X=node_embedding_obs, A=edge_index_obs)[:, :n_agent, :]
                cat_embedding = node_embedding_obs
                A_new, logits = self.func_glcn(cat_embedding, rollout = False)
                cat_embedding = self.func_comm(X=cat_embedding, A=A_new, dense=True, check = True)
                cat_embedding = torch.cat([cat_embedding, node_embedding_obs, node_embedding_comm], dim=2)

                return cat_embedding, A_new
        else:
            if target == False:
                node_feature = torch.tensor(node_feature, dtype=torch.float, device=device)
                agent_feature = torch.tensor(agent_feature, dtype=torch.float, device=device)
                batch_size = node_feature.shape[0]
                num_nodes = node_feature.shape[1]
                num_agents = agent_feature.shape[1]
                node_feature = node_feature.reshape(batch_size * num_nodes, -1)
                agent_feature = agent_feature.reshape(batch_size * num_agents, -1)
                node_embedding_obs = self.node_representation(node_feature)
                node_embedding_comm = self.node_representation_comm(agent_feature)
                node_embedding_obs = node_embedding_obs.reshape(batch_size, num_nodes, -1)
                node_embedding_comm = node_embedding_comm.reshape(batch_size, num_agents, -1)
                node_embedding_obs = self.func_obs(X=node_embedding_obs, A=edge_index_obs)[:, :n_agent, :]

                cat_embedding = node_embedding_obs
                A_new, logits = self.func_glcn(cat_embedding, rollout = False)
                cat_embedding = self.func_comm(X=cat_embedding, A=A_new, dense=True)
                cat_embedding = torch.cat([cat_embedding, node_embedding_obs, node_embedding_comm], dim=2)
                return cat_embedding, A_new, logits
            else:
                with torch.no_grad():
                    node_feature = torch.tensor(node_feature, dtype=torch.float, device=device)
                    agent_feature = torch.tensor(agent_feature, dtype=torch.float, device=device)
                    batch_size = node_feature.shape[0]
                    num_nodes = node_feature.shape[1]
                    num_agents = agent_feature.shape[1]
                    node_feature = node_feature.reshape(batch_size * num_nodes, -1)
                    agent_feature = agent_feature.reshape(batch_size * num_agents, -1)

                    node_embedding_obs = self.node_representation_tar(node_feature)
                    node_embedding_comm = self.node_representation_comm_tar(agent_feature)

                    node_embedding_obs = node_embedding_obs.reshape(batch_size, num_nodes, -1)
                    node_embedding_comm = node_embedding_comm.reshape(batch_size, num_agents, -1)

                    node_embedding_obs = self.func_obs_tar(X=node_embedding_obs, A=edge_index_obs)[:, :n_agent, :]
                    cat_embedding = node_embedding_obs
                    A_new, logits = self.func_glcn_tar(cat_embedding, rollout=False)
                    cat_embedding = self.func_comm_tar(X=cat_embedding, A=A_new, dense=True)
                    cat_embedding = torch.cat([cat_embedding, node_embedding_obs, node_embedding_comm], dim=2)
                    return cat_embedding

    def cal_Q(self, obs, actions, action_features, avail_actions_next, A, target=False):
        """
        node_representation
        - training 시        : batch_size X num_nodes X feature_size
        - action sampling 시 : num_nodes X feature_size
        """
        if target == False:

            action_features = torch.tensor(action_features).to(device=device, dtype=torch.float32)
            action_size = action_features.shape[1]
            obs = obs.unsqueeze(2)
            obs = obs.expand([self.batch_size, self.num_agent, action_size, -1])

            action_features = action_features.reshape(self.batch_size * action_size, -1)
            action_embedding = self.action_representation(action_features)
            action_embedding = action_embedding.reshape(self.batch_size, action_size, -1).unsqueeze(1)
            action_embedding = action_embedding.expand([self.batch_size, self.num_agent, action_size, -1])

            obs_and_action = torch.concat([obs, action_embedding], dim=3)
            obs_and_action = obs_and_action.reshape([self.batch_size * self.num_agent * action_size, -1])

            Q = self.Q(obs_and_action)
            Q = Q.reshape([self.batch_size, self.num_agent, action_size])

            actions = torch.tensor(actions, device=device).long()
            act_n = actions.unsqueeze(2)  # action.shape : (batch_size, num_agent, 1)
            q = torch.gather(Q, 2, act_n)  # q.shape : (batch_size, num_agent, action_size)
            return q
        else:
            with torch.no_grad():
                avail_actions_next = torch.tensor(avail_actions_next, device=device).bool()
                mask = avail_actions_next

                action_features = torch.tensor(action_features).to(device=device, dtype=torch.float32)
                action_size = action_features.shape[1]
                obs = obs.unsqueeze(2)
                obs = obs.expand([self.batch_size, self.num_agent, action_size, -1])

                action_features = action_features.reshape(self.batch_size * action_size, -1)
                action_embedding = self.action_representation_tar(action_features)
                action_embedding = action_embedding.reshape(self.batch_size, action_size, -1).unsqueeze(1)
                action_embedding = action_embedding.expand([self.batch_size, self.num_agent, action_size, -1])

                obs_and_action = torch.concat([obs, action_embedding], dim=3)
                obs_and_action = obs_and_action.reshape([self.batch_size * self.num_agent * action_size, -1])

                Q_tar = self.Q_tar(obs_and_action)
                Q_tar = Q_tar.reshape([self.batch_size, self.num_agent, action_size])

                Q_tar = Q_tar.masked_fill(mask == 0, float('-inf'))
                Q_tar_max = torch.max(Q_tar, dim=2)[0]
                return Q_tar_max

    @torch.no_grad()
    def sample_action(self, node_representation, action_feature, avail_action, epsilon):
        obs = node_representation
        action_features = torch.tensor(action_feature).to(device=device, dtype=torch.float32).unsqueeze(0)
        action_size = action_features.shape[1]
        obs = obs.unsqueeze(2)
        obs = obs.expand([1, self.num_agent, action_size, -1])
        action_features = action_features.reshape(action_size, -1)
        action_embedding = self.action_representation(action_features)
        action_embedding = action_embedding.reshape(1, action_size, -1).unsqueeze(1)
        action_embedding = action_embedding.expand([1, self.num_agent, action_size, -1])
        obs_and_action = torch.concat([obs, action_embedding], dim=3)
        obs_and_action = obs_and_action.reshape([1 * self.num_agent * action_size, -1])
        Q = self.Q(obs_and_action)
        Q = Q.reshape([self.num_agent, action_size])
        """
        node_representation 차원 : n_agents X n_representation_comm
        action_feature 차원      : action_size X n_action_feature
        avail_action 차원        : n_agents X action_size
        """
        mask = torch.tensor(avail_action, device=device).bool()
        Q = Q.masked_fill(mask == 0, float('-inf'))
        action = []
        action_space = [i for i in range(action_size)]
        action_history = torch.zeros([self.num_agent, self.feature_size + 5])
        for n in range(self.num_agent):

            greedy_u = torch.argmax(Q[n, :])
            mask_n = np.array(avail_action[n], dtype=np.float64)

            if np.random.uniform(0, 1) >= epsilon:
                u = greedy_u
                action.append(u.item())
                action_history[n, :] = action_features[u.item(), :]


            else:
                u = np.random.choice(action_space, p=mask_n / np.sum(mask_n))
                action.append(u)
                action_history[n, :] = action_features[u, :]

        return action, action_history

    # @torch.no_grad()
    # def sample_action(self, node_representation, action_feature, avail_action, epsilon):
    #     """
    #     node_representation 차원 : n_agents X n_representation_comm
    #     action_feature 차원      : action_size X n_action_feature
    #     avail_action 차원        : n_agents X action_size
    #     """
    #     mask = torch.tensor(avail_action, device=device).bool()
    #     action_feature = torch.tensor(action_feature, device=device, dtype = torch.float64).float()
    #     action_size = action_feature.shape[0]
    #     action = []
    #     action_embedding = self.action_representation(action_feature)
    #     action_space = [i for i in range(action_size)]
    #     action_feature_size = action_feature.shape[1]
    #     selected_action_feature = torch.zeros(self.num_agent, action_feature_size).to(device)
    #     for n in range(self.num_agent):
    #         obs = node_representation[n].expand(action_size, node_representation[n].shape[0])  # 차원 : action_size X n_representation_comm
    #         obs_cat_action = torch.concat([obs, action_embedding], dim=1)  # 차원 : action_size
    #         Q = self.Q(obs_cat_action).squeeze(1)  # 차원 : action_size X 1
    #         Q = Q.masked_fill(mask[n, :] == 0, float('-inf'))
    #         greedy_u = torch.argmax(Q)
    #         mask_n = np.array(avail_action[n], dtype=np.float64)
    #         if np.random.uniform(0, 1) >= epsilon:
    #             u = greedy_u
    #             action.append(u.item())
    #             selected_action_feature[n, :] = action_feature[u.item()]
    #         else:
    #             u = np.random.choice(action_space, p=mask_n / np.sum(mask_n))
    #             action.append(u)
    #             selected_action_feature[n, :] = action_feature[u.item()]
    #     return action

    def eval(self, train=False):
        if train == False:
            self.func_glcn.eval()
            self.VDN.eval()
            self.Q.eval()
            self.Q_tar.eval()
            self.C.eval()
            self.C_tar.eval()
            self.node_representation.eval()
            self.node_representation_comm.eval()
            self.func_obs.eval()
            self.action_representation.eval()
            self.func_comm.eval()
            self.func_comm_tar.eval()
            self.func_comm2.eval()
            self.func_comm2_tar.eval()
        else:
            self.Q.train()
            self.func_glcn.train()
            self.func_obs.train()
            self.action_representation.train()
            self.node_representation_comm.train()
            self.node_representation.train()
            self.Q_tar.eval()
            self.C.train()
            self.C_tar.eval()
            self.func_comm.eval()
            self.func_comm_tar.train()
            self.func_comm2.eval()
            self.func_comm2_tar.train()
            self.node_representation_tar.eval()
            self.node_representation_comm_tar.eval()
            self.action_representation_tar.eval()
            self.func_obs_tar.eval()
            self.func_glcn_tar.eval()

    def learn(self, cum_losses_old, graph_learning_stop):
        self.eval(train=True)
        node_features, actions, action_features, edge_indices_enemy, edge_indices_ally, rewards, dones, node_features_next, action_features_next, edge_indices_enemy_next, edge_indices_ally_next, avail_actions_next, dead_masking, dead_masking_next, agent_feature, agent_feature_next, sum_state, sum_state_next = self.buffer.sample()
        A = edge_indices_ally
        A_next = edge_indices_ally_next
        A = torch.stack(A).to(device)
        A_next = torch.stack(A_next).to(device)
        """
        node_features : batch_size x num_nodes x feature_size
        actions : batch_size x num_agents
        action_feature :     batch_size x action_size x action_feature_size
        avail_actions_next : batch_size x num_agents x action_size 
        """
        num_nodes = torch.tensor(node_features).shape[1]
        n_agent = torch.tensor(avail_actions_next).shape[1]

        obs, A, logits = self.get_node_representation_temp(node_features, agent_feature, edge_indices_enemy,
                                                           n_agent=n_agent,
                                                           mini_batch=True, A_old=A)
        obs_next = self.get_node_representation_temp(node_features_next, agent_feature_next, edge_indices_enemy_next,
                                                     n_agent=n_agent,
                                                     mini_batch=True, target=True, A_old=A_next)

        gamma1 = self.gamma1
        gamma2 = self.gamma2
        lap_quad, sec_eig_upperbound = get_graph_loss(obs, A)

        dones = torch.tensor(dones, device=device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)

        q_tot = self.cal_Q(obs=obs,
                           actions=actions,
                           action_features=action_features,
                           avail_actions_next=None,
                           target=False, A=A)
        q_tot_tar = self.cal_Q(obs=obs_next,
                               actions=None,
                               action_features=action_features_next,
                               avail_actions_next=avail_actions_next,
                               target=True, A=A_next)

        var_ = torch.mean(torch.var(q_tot, dim=1))
        q_tot = self.VDN(q_tot, dead_masking)
        q_tot_tar = self.VDN_target(q_tot_tar, dead_masking_next)

        td_target = rewards * self.num_agent + self.gamma * (1 - dones) * q_tot_tar
        exp_adv = torch.exp((td_target.detach() - q_tot.squeeze(1)) / self.num_agent)
        comm_loss = -logits * exp_adv.detach()
        rl_loss = F.mse_loss(q_tot.squeeze(1), td_target.detach())
        graph_loss = gamma1 * lap_quad - gamma2 * sec_eig_upperbound
        loss = rl_loss + graph_loss + float(os.environ.get("var_reg", 0.5)) * var_ +0.001*comm_loss.mean()#######
        loss.backward()
        grad_clip = float(os.environ.get("grad_clip", 10))
        torch.nn.utils.clip_grad_norm_(self.eval_params, grad_clip)
        self.optimizer.step()

        self.optimizer.zero_grad()

        tau = 1e-4
        for target_param, local_param in zip(self.Q_tar.parameters(),
                                             self.Q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        for target_param, local_param in zip(self.VDN_target.parameters(),
                                             self.VDN.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        for target_param, local_param in zip(self.node_representation_tar.parameters(),
                                             self.node_representation.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        for target_param, local_param in zip(self.node_representation_comm_tar.parameters(),
                                             self.node_representation_comm.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        for target_param, local_param in zip(self.action_representation_tar.parameters(),
                                             self.action_representation.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        for target_param, local_param in zip(self.func_obs_tar.parameters(),
                                             self.func_obs.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        for target_param, local_param in zip(self.func_comm_tar.parameters(),
                                             self.func_comm.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        for target_param, local_param in zip(self.func_comm2_tar.parameters(),
                                             self.func_comm2.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        for target_param, local_param in zip(self.func_glcn_tar.parameters(),
                                             self.func_glcn.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        self.eval(train=False)
        if cfg.given_edge == True:
            return loss
        else:
            return loss, lap_quad.tolist(), sec_eig_upperbound.tolist(), rl_loss.tolist(), q_tot.tolist(), comm_loss.tolist()

