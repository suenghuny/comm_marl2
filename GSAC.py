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
from GLCN.GLCN2 import GLCN, GAT
from cfg import get_cfg
cfg = get_cfg()
#from GAT.model import GAT
from GAT.layers import device
from copy import deepcopy


class Replay_Buffer:
    def __init__(self, buffer_size, batch_size, num_agent, h = 1):
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

    def memory(self, node_feature, action, action_feature, edge_index_enemy, edge_index_ally, reward, done, avail_action, dead_masking, agent_feature, sum_state):
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
        buffer_dict = {'buffer':self.buffer, 'step_count_list':self.step_count_list, 'step_count':self.step_count}
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
                yield datas[0][s+1]

            if cat == 'action':
                yield datas[1][s]


            if cat == 'action_feature':
                yield datas[2][s]
            if cat == 'action_feature_next':
                yield datas[2][s+1]

            if cat == 'edge_index_enemy':
                yield datas[3][s]
            if cat == 'edge_index_enemy_next':
                yield datas[3][s+1]

            if cat == 'edge_index_ally':
                yield datas[4][s]
            if cat == 'edge_index_ally_next':
                yield datas[4][s+1]

            if cat == 'reward':
                yield datas[5][s]

            if cat == 'done':
                yield datas[6][s]


            if cat == 'avail_action':
                yield datas[7][s]

            if cat == 'avail_action_next':
                yield datas[7][s+1]

            if cat == 'dead_masking':
                yield datas[8][s]

            if cat == 'dead_masking_next':
                yield datas[8][s+1]

            if cat == 'agent_feature':
                yield datas[9][s]
            if cat == 'agent_feature_next':
                yield datas[9][s+1]


            if cat == 'sum_state':
                yield datas[10][s]
            if cat == 'sum_state_next':
                yield datas[10][s+1]



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

        avail_action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_action')
        avail_actions = list(avail_action)

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
               dones, node_features_next, action_features_next, edge_indices_enemy_next, edge_indices_ally_next, \
               avail_actions, avail_actions_next,dead_masking, \
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

        self.anneal_episodes_graph_variance=anneal_episodes_graph_variance
        self.min_graph_variance=min_graph_variance

        self.node_representation = NodeEmbedding(feature_size=self.feature_size,
                                                   hidden_size=self.hidden_size_obs,
                                                   n_representation_obs=self.n_representation_obs).to(device)  # 수정사항
        self.node_representation_tar = NodeEmbedding(feature_size=self.feature_size,
                                                 hidden_size=self.hidden_size_obs,
                                                 n_representation_obs=self.n_representation_obs).to(device)  # 수정사항
        self.node_representation_comm = NodeEmbedding(feature_size=self.feature_size,
                                                      hidden_size=self.hidden_size_comm,
                                                      n_representation_obs=self.n_representation_comm).to(device)
        self.node_representation_comm_tar = NodeEmbedding(feature_size=self.feature_size,
                                                      hidden_size=self.hidden_size_comm,
                                                      n_representation_obs=self.n_representation_comm).to(device)
        self.action_representation = NodeEmbedding(feature_size=self.feature_size + 5,
                                                   hidden_size=self.hidden_size_action,
                                                   n_representation_obs=self.n_representation_action).to(device)  # 수정사항


        self.action_representation_tar = NodeEmbedding(feature_size=self.feature_size + 5,
                                                   hidden_size=self.hidden_size_action,
                                                   n_representation_obs=self.n_representation_action).to(device)  # 수정사항
        self.func_obs = GAT(feature_size=self.n_representation_obs, graph_embedding_size=self.graph_embedding).to(device)
        self.func_obs_tar = GAT(feature_size=self.n_representation_obs, graph_embedding_size=self.graph_embedding).to(device)
        self.func_comm = GAT(feature_size=self.graph_embedding+n_representation_comm, graph_embedding_size=self.graph_embedding_comm).to(device)
        self.func_comm_tar = GAT(feature_size=self.graph_embedding+n_representation_comm, graph_embedding_size=self.graph_embedding_comm).to(device)
        self.func_glcn = GLCN(feature_size=self.graph_embedding+self.n_representation_comm,
                              feature_obs_size=self.graph_embedding,
                              graph_embedding_size=self.graph_embedding_comm).to(device)
        self.func_glcn_tar = GLCN(feature_size=self.graph_embedding + self.n_representation_comm,
                              feature_obs_size=self.graph_embedding,
                              graph_embedding_size=self.graph_embedding_comm).to(device)

        self.Pi = Pi_Attention(self.graph_embedding_comm, self.n_representation_action, hidden_size_Q).to(device)



        self.node_representation_tar.load_state_dict(self.node_representation.state_dict())
        self.node_representation_comm_tar.load_state_dict(self.node_representation_comm.state_dict())
        self.action_representation_tar.load_state_dict(self.action_representation.state_dict())
        self.func_obs_tar.load_state_dict(self.func_obs.state_dict())
        self.func_glcn_tar.load_state_dict(self.func_glcn.state_dict())


        self.Q1 = Network(self.graph_embedding_comm + self.n_representation_action, hidden_size_Q).to(device)
        self.Q1_tar = Network(self.graph_embedding_comm + self.n_representation_action, hidden_size_Q).to(device)
        self.Q1_tar.load_state_dict(self.Q1.state_dict())

        self.Q2 = Network(self.graph_embedding_comm + self.n_representation_action, hidden_size_Q).to(device)
        self.Q2_tar = Network(self.graph_embedding_comm + self.n_representation_action, hidden_size_Q).to(device)
        self.Q2_tar.load_state_dict(self.Q2.state_dict())

        self.func_comm_tar.load_state_dict(self.func_comm.state_dict())
        self.eps_clip = 1000
        self.original_loss = None
        self.eval_params = list(self.func_glcn.parameters()) + \
                   list(self.VDN.parameters()) + \
                   list(self.Pi.parameters()) + \
                   list(self.Q1.parameters()) + \
                   list(self.Q2.parameters()) + \
                   list(self.node_representation.parameters()) + \
                   list(self.node_representation_comm.parameters()) + \
                   list(self.func_obs.parameters()) + \
                   list(self.func_comm.parameters()) + \
                   list(self.action_representation.parameters())
        param_groups = [
            {'params': self.eval_params},
        ]
        self.optimizer = optim.Adam(param_groups, lr=learning_rate)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_ratio)


    def save_model(self, file_dir, e, t, win_rate):
        torch.save({
                        "1": self.Pi.state_dict(),


                        "3": self.func_glcn.state_dict(),
                        "4": self.func_obs.state_dict(),
                        "5": self.action_representation.state_dict(),
                        "6": self.node_representation_comm.state_dict() ,
                        "7": self.node_representation.state_dict(),
                        "8": self.node_representation_tar.state_dict(),
                        "9": self.node_representation_comm_tar.state_dict(),
                        "10": self.action_representation_tar.state_dict(),
                        "11": self.func_obs_tar.state_dict(),
                        "12": self.func_glcn_tar.state_dict(),
                        "13": self.Q1_tar.state_dict(),
                        "14": self.Q2_tar.state_dict(),
                        "15": self.Q1.state_dict(),
                        "16": self.Q2.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                        },
                       file_dir+ "episode{}_t_{}_win_{}.pt".format(e, t, win_rate))
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
    def get_node_representation_temp(self, node_feature, agent_feature, edge_index_obs,n_agent,
                                    mini_batch = False, target = False, A_old= None):
        if mini_batch == False:
            with torch.no_grad():
                edge_index_obs = torch.tensor(edge_index_obs).long().to(device).unsqueeze(0)
                node_feature = torch.tensor(node_feature, dtype=torch.float, device=device).unsqueeze(0)
                agent_feature = torch.tensor(agent_feature, dtype=torch.float, device=device).unsqueeze(0)
                batch_size = node_feature.shape[0]
                num_nodes = node_feature.shape[1]
                num_agents = agent_feature.shape[1]
                node_feature = node_feature.reshape(batch_size*num_nodes, -1)
                agent_feature = agent_feature.reshape(batch_size * num_agents, -1)
                node_embedding_obs  = self.node_representation(node_feature)
                node_embedding_comm = self.node_representation_comm(agent_feature)
                node_embedding_obs = node_embedding_obs.reshape(batch_size, num_nodes, -1)
                node_embedding_comm = node_embedding_comm.reshape(batch_size, num_agents, -1)
                node_embedding_obs = self.func_obs(X = node_embedding_obs, A = edge_index_obs)[:, :n_agent,:]
                cat_embedding = torch.cat([node_embedding_obs, node_embedding_comm], dim=2)
                A_new, logits = self.func_glcn(cat_embedding, rollout = False)
                cat_embedding= self.func_comm(X = cat_embedding, A = A_new, dense = True)
                return cat_embedding, A_new
        else:
            if target == False:
                node_feature = torch.tensor(node_feature, dtype=torch.float, device=device)
                agent_feature = torch.tensor(agent_feature, dtype=torch.float, device=device)
                batch_size = node_feature.shape[0]
                num_nodes = node_feature.shape[1]
                num_agents = agent_feature.shape[1]
                node_feature = node_feature.reshape(batch_size*num_nodes, -1)
                agent_feature = agent_feature.reshape(batch_size * num_agents, -1)
                node_embedding_obs  = self.node_representation(node_feature)
                node_embedding_comm = self.node_representation_comm(agent_feature)
                node_embedding_obs = node_embedding_obs.reshape(batch_size, num_nodes, -1)
                node_embedding_comm = node_embedding_comm.reshape(batch_size, num_agents, -1)
                node_embedding_obs = self.func_obs(X = node_embedding_obs, A = edge_index_obs)[:, :n_agent,:]
                cat_embedding = torch.cat([node_embedding_obs, node_embedding_comm], dim=2)
                A_new, logits = self.func_glcn(cat_embedding, rollout = False)
                cat_embedding= self.func_comm(X = cat_embedding, A = A_new,  dense = True)
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

                    node_embedding_obs  = self.node_representation_tar(node_feature)
                    node_embedding_comm = self.node_representation_comm_tar(agent_feature)

                    node_embedding_obs = node_embedding_obs.reshape(batch_size, num_nodes, -1)
                    node_embedding_comm = node_embedding_comm.reshape(batch_size, num_agents, -1)

                    node_embedding_obs = self.func_obs_tar(X = node_embedding_obs, A = edge_index_obs)[:, :n_agent,:]
                    cat_embedding = torch.cat([node_embedding_obs, node_embedding_comm], dim=2)
                    A_new, logits = self.func_glcn_tar(cat_embedding, rollout=False)
                    cat_embedding= self.func_comm_tar(X=cat_embedding, A=A_new, dense = True)
                    return cat_embedding





    def cal_Q(self, obs, actions, action_features, i = None):
        """
        node_representation
        - training 시        : batch_size X num_nodes X feature_size
        - action sampling 시 : num_nodes X feature_size
        """
        action_size = action_features.shape[1]
        obs = obs.unsqueeze(2)
        obs = obs.expand([self.batch_size, self.num_agent, action_size, -1])

        action_features = action_features.reshape(self.batch_size * action_size, -1)
        action_embedding = self.action_representation(action_features)
        action_embedding = action_embedding.reshape(self.batch_size, action_size, -1).unsqueeze(1)
        action_embedding = action_embedding.expand([self.batch_size, self.num_agent, action_size, -1])
        obs_and_action = torch.concat([obs, action_embedding], dim=3)
        obs_and_action = obs_and_action.reshape([self.batch_size * self.num_agent * action_size, -1])





        if i == 1:
            Q = self.Q1(obs_and_action)
        else:
            Q = self.Q2(obs_and_action)
        actions = torch.stack(actions)
        actions = torch.tensor(actions, device = device).long()
        #act_n = actions.unsqueeze(2)                    # action.shape : (batch_size, num_agent, 1)
        #print(Q.shape, actions.shape)
        q = torch.gather(Q, 2, actions)                   # q.shape : (batch_size, num_agent, action_size)
        return q

    def cal_Q_tar(self, obs, action_features, avail_actions, i=None):
        with torch.no_grad():
            avail_actions = torch.tensor(avail_actions, device=device).bool()
            mask = avail_actions
            action_size = action_features.shape[1]
            obs = obs.unsqueeze(2)
            obs = obs.expand([self.batch_size, self.num_agent, action_size, -1])

            action_features = action_features.reshape(self.batch_size * action_size, -1)
            action_embedding = self.action_representation(action_features)
            action_embedding = action_embedding.reshape(self.batch_size, action_size, -1).unsqueeze(1)
            action_embedding = action_embedding.expand([self.batch_size, self.num_agent, action_size, -1])
            obs_and_action = torch.concat([obs, action_embedding], dim=3)
            obs_and_action = obs_and_action.reshape([self.batch_size * self.num_agent * action_size, -1])

            probs = self.Pi(obs, action_embedding, mask)
            batch_size, num_agent, num_action = probs.shape
            probs_reshaped = probs.reshape(-1, num_action)  # [batch_size*num_agent, num_action]

            actions_flat = torch.multinomial(probs_reshaped, num_samples=1).squeeze(-1)  # [batch_size*num_agent]

            # 원래 배치 및 에이전트 구조로 다시 변환
            actions = actions_flat.reshape(batch_size, num_agent)  # [batch_size, num_agent]
            actions = torch.tensor(actions, device=device).long()
            #act_n = actions.unsqueeze(2)  # action.shape : (batch_size, num_agent, 1)


            Q1_tar = self.Q1_tar(obs_and_action)
            Q2_tar = self.Q2_tar(obs_and_action)
            probs = torch.gather(probs, 2, actions.unsqueeze(2))  # q.shape : (batch_size, num_agent, action_size)
            q1_tar = torch.gather(Q1_tar, 2, actions.unsqueeze(2))  # q.shape : (batch_size, num_agent, action_size)
            q2_tar = torch.gather(Q2_tar, 2, actions.unsqueeze(2))  # q.shape : (batch_size, num_agent, action_size)
            min_q_tar = torch.min(q1_tar, q2_tar)
            return min_q_tar, probs

    def cal_prob(self, obs, actions, action_features, avail_actions):
        """
        node_representation
        - training 시        : batch_size X num_nodes X feature_size
        - action sampling 시 : num_nodes X feature_size
        """
        avail_actions = torch.tensor(avail_actions, device=device).bool()
        mask = avail_actions
        action_features = torch.tensor(action_features).to(device=device, dtype=torch.float32)
        action_size = action_features.shape[1]
        action_features = action_features.reshape(self.batch_size * action_size, -1)
        action_embedding = self.action_representation(action_features)
        action_embedding = action_embedding.reshape(self.batch_size, action_size, -1)
        probs = self.Pi(obs, action_embedding, mask)

        batch_size, num_agent, num_action = probs.shape
        probs_reshaped = probs.reshape(-1, num_action)  # [batch_size*num_agent, num_action]

        actions_flat = torch.multinomial(probs_reshaped, num_samples=1).squeeze(-1)  # [batch_size*num_agent]

        # 원래 배치 및 에이전트 구조로 다시 변환
        actions = actions_flat.reshape(batch_size, num_agent)  # [batch_size, num_agent]
        actions = torch.tensor(actions, device=device).long()

        Q1 = self.Q1(obs, action_embedding)
        Q2 = self.Q2(obs, action_embedding)
        probs = torch.gather(probs, 2, actions.unsqueeze(2))  # q.shape : (batch_size, num_agent, action_size)
        q1 = torch.gather(Q1, 2, actions.unsqueeze(2))  # q.shape : (batch_size, num_agent, action_size)
        q2 = torch.gather(Q2, 2, actions.unsqueeze(2))  # q.shape : (batch_size, num_agent, action_siz
        min_q = torch.min(q1, q2)
        return probs, min_q

    @torch.no_grad()
    def sample_action(self, node_representation, action_feature, avail_action, epsilon):
        """
        node_representation 차원 : n_agents X n_representation_comm
        action_feature 차원      : action_size X n_action_feature
        avail_action 차원        : n_agents X action_size
        """
        mask = torch.tensor(avail_action, device=device).bool()
        action_feature = torch.tensor(action_feature, device=device, dtype = torch.float64).float().unsqueeze(0)
        action_size = action_feature.shape[1]
        action = []
        action_embedding = self.action_representation(action_feature)
        action_space = [i for i in range(action_size)]

        obs = node_representation
        prob = self.Pi(obs, action_embedding, mask)
        action = torch.multinomial(prob.squeeze(0), num_samples=1)
        return action


    def eval(self, train = False):
        if train == False:
            self.func_glcn.eval()
            self.VDN.eval()
            self.Q1.eval()
            self.Q1_tar.eval()
            self.Q2.eval()
            self.Q2_tar.eval()
            self.node_representation.eval()
            self.node_representation_comm.eval()
            self.func_obs.eval()
            self.action_representation.eval()
            self.func_comm.eval()
            self.func_comm_tar.eval()
        else:
            self.Q1.train()
            self.Q2.train()
            self.func_glcn.train()
            self.func_obs.train()
            self.action_representation.train()
            self.node_representation_comm.train()
            self.node_representation.train()
            self.Q1_tar.eval()
            self.Q2_tar.eval()
            self.func_comm.eval()
            self.func_comm_tar.train()
            self.node_representation_tar.eval()
            self.node_representation_comm_tar.eval()
            self.action_representation_tar.eval()
            self.func_obs_tar.eval()
            self.func_glcn_tar.eval()



    def learn(self, cum_losses_old, graph_learning_stop):
        self.eval(train = True)
        node_features, actions, action_features, edge_indices_enemy, edge_indices_ally, rewards, dones, node_features_next, action_features_next, edge_indices_enemy_next, edge_indices_ally_next, avail_actions,\
        avail_actions_next,dead_masking, dead_masking_next, agent_feature, agent_feature_next, sum_state, sum_state_next = self.buffer.sample()
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
                                                         mini_batch=True, A_old = A)
        obs_next = self.get_node_representation_temp(node_features_next, agent_feature_next, edge_indices_enemy_next,
                                                              n_agent=n_agent,
                                                              mini_batch=True, target = True, A_old = A_next)

        gamma1 = self.gamma1
        gamma2 = self.gamma2
        lap_quad, sec_eig_upperbound = get_graph_loss(obs, A)
        dones = torch.tensor(dones, device = device, dtype = torch.float)
        rewards = torch.tensor(rewards, device = device, dtype = torch.float)

        q1_tot = self.cal_Q(obs=obs,
                     actions=actions,
                     action_features=action_features, i = 1)
        q2_tot = self.cal_Q(obs=obs,
                     actions=actions,
                     action_features=action_features, i = 2)

        q_tot_tar, probs_next = self.cal_Q_tar(obs=obs_next,
                                   action_features=action_features_next,
                                   avail_actions=avail_actions_next)

        q1_tot = torch.sum(q1_tot, dim = 1)
        q2_tot = torch.sum(q2_tot, dim = 1)
        q_tot_tar = torch.sum(q_tot_tar, dim=1)
        probs_next = torch.sum(probs_next.squeeze(2), dim = 1, keepdim = True)

        #print(rewards.shape, dones.shape, q_tot_tar.shape,probs_next.shape)
        td_target = rewards.unsqueeze(1)*self.num_agent + self.gamma* (1-dones.unsqueeze(1))*(q_tot_tar-0.001*torch.log(probs_next))
        probs, min_q = self.cal_prob(obs, actions, action_features, avail_actions)
        pi_loss = torch.sum(min_q-0.001* torch.log(probs), dim = 1).mean()


        rl_loss1 = F.mse_loss(q1_tot, td_target.detach())
        rl_loss2 = F.mse_loss(q2_tot, td_target.detach())
        graph_loss = gamma1 * lap_quad - gamma2 * gamma1 * sec_eig_upperbound

        loss = rl_loss1+rl_loss2 +graph_loss+pi_loss
        loss.backward()
        grad_clip = float(os.environ.get("grad_clip", 10))
        torch.nn.utils.clip_grad_norm_(self.eval_params, grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        tau = 1e-4
        for target_param, local_param in zip(self.Q1_tar.parameters(),
                                             self.Q1.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        for target_param, local_param in zip(self.Q2_tar.parameters(),
                                             self.Q2.parameters()):
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

        for target_param, local_param in zip(self.func_glcn_tar.parameters(),
                                            self.func_glcn.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        self.eval(train=False)
        if cfg.given_edge == True:
            return loss
        else:
            return loss, lap_quad.tolist(), sec_eig_upperbound.tolist(), rl_loss1.tolist(), q1_tot.tolist(), pi_loss.tolist()

