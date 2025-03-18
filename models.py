import torch
import torch.nn as nn
import torch.nn.functional as F
class VDN(nn.Module):
    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, q_local, dead_masking):
        # num_agent = torch.tensor(dead_masking, dtype = torch.float).shape[1]
        # dm = torch.sum(torch.tensor(dead_masking, dtype = torch.float).to(device), dim=1, keepdim= True)
        #
        # return torch.sum(q_local*torch.tensor(dead_masking, dtype = torch.float).to(device)*num_agent / dm, dim = 1)
        return torch.sum(q_local, dim=1)


class MixingNet(nn.Module):
    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, q_local):
        return torch.sum(q_local, dim = 1)


class Network(nn.Module):
    def __init__(self, obs_and_action_size, hidden_size_q):
        super(Network, self).__init__()
        self.obs_and_action_size = obs_and_action_size
        print(obs_and_action_size, hidden_size_q)
        self.fcn_1 = nn.Linear(obs_and_action_size, hidden_size_q)
        self.fcn_2 = nn.Linear(hidden_size_q, int(hidden_size_q/2))
        self.fcn_3 = nn.Linear(int(hidden_size_q/2), int(hidden_size_q/4))
        self.fcn_4 = nn.Linear(int(hidden_size_q/4), int(hidden_size_q/8))
        self.fcn_5 = nn.Linear(int(hidden_size_q/8), 1)
        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)
        torch.nn.init.xavier_uniform_(self.fcn_4.weight)
        torch.nn.init.xavier_uniform_(self.fcn_5.weight)

    def forward(self, obs_and_action):
        #obs_and_action = torch.concat([obs, action])
        x = F.elu(self.fcn_1(obs_and_action))
        x = F.elu(self.fcn_2(x))
        x = F.elu(self.fcn_3(x))
        x = F.elu(self.fcn_4(x))
        q = self.fcn_5(x)
        return q


class Q_Attention(nn.Module):
    def __init__(self, obs_size, action_size, hidden_dim=64, num_heads=4):
        super(Q_Attention, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Encoders for observations and actions
        self.obs_encoder = nn.Linear(obs_size, hidden_dim)
        self.action_encoder = nn.Linear(action_size, hidden_dim)

        # Multi-head attention components
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj2 = nn.Linear(hidden_dim, hidden_dim)

        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.key_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc3 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.fc4 = nn.Linear(int(hidden_dim/4), int(hidden_dim/8))
        self.fc5 = nn.Linear(int(hidden_dim/8), 1)

    def forward(self, obs, action, mask):
        """
        Args:
            obs: [batch_size, num_agent, obs_size]
            action: [batch_size, num_action, action_size]
        Returns:
            q: [batch_size, num_agent, action_size]
        """
        batch_size, num_agent, num_action = obs.shape
        _, num_action,_ = action.shape

        # Encode observations and actions
        encoded_obs = self.obs_encoder(obs)  # [batch_size, num_agent, hidden_dim]
        encoded_action = self.action_encoder(action)  # [batch_size, num_action, hidden_dim]

        # Prepare for attention
        query = self.query_proj(encoded_obs)  # [batch_size, num_agent, hidden_dim]
        key = self.key_proj(encoded_action)  # [batch_size, num_action, hidden_dim]
        value = self.value_proj(encoded_action)  # [batch_size, num_action, hidden_dim]

        # Calculate attention scores
        # [batch_size, num_agent, num_action]
        attention_scores = torch.einsum('bdh,bah->bda', query, key) / (self.hidden_dim ** 0.5)
        # if mask.dim() == 2:
        #     attention_scores = attention_scores.masked_fill(mask.unsqueeze(0) == 0, float(-1e8))
        # else:
        #     attention_scores = attention_scores.masked_fill(mask == 0, float(-1e8))
        attention_weights = F.softmax(attention_scores, dim=-1)
        query = torch.einsum('bda,bah->bdh', attention_weights, value)
        query = self.query_proj2(query)
        key = self.key_proj2(encoded_action)  # [batch_size, num_action, hidden_dim]
        q = torch.einsum('bdh,bah->bda', query, key) #/ (self.hidden_dim ** 0.5)
        return q

class Pi_Attention(nn.Module):
    def __init__(self, obs_size, action_size, hidden_dim=64, num_heads=4):
        super(Pi_Attention, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # Encoders for observations and actions
        self.obs_encoder = nn.Linear(obs_size, hidden_dim)
        self.action_encoder = nn.Linear(action_size, hidden_dim)

        # Multi-head attention components
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.query_proj2 = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.query_proj3 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.key_proj2 = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.key_proj3 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.value_proj2 = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.value_proj3 = nn.Linear(hidden_dim, hidden_dim, bias=False)



    def forward(self, obs, action, mask):
        """
        Args:
            obs: [batch_size, num_agent, obs_size]
            action: [batch_size, num_action, action_size]
        Returns:
            q: [batch_size, num_agent, action_size]
        """
        batch_size, num_agent, _ = obs.shape
        _, num_action, _ = action.shape

        # Encode observations and actions
        encoded_obs = self.obs_encoder(obs)  # [batch_size, num_agent, hidden_dim]
        encoded_action = self.action_encoder(action)  # [batch_size, num_action, hidden_dim]

        # Prepare for attention
        query = self.query_proj(encoded_obs)  # [batch_size, num_agent, hidden_dim]
        key = self.key_proj(encoded_action)  # [batch_size, num_action, hidden_dim]
        value = self.value_proj(encoded_action)  # [batch_size, num_action, hidden_dim]

        attention_scores = torch.einsum('bdh,bah->bda', query, key) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        query = torch.einsum('bda,bah->bdh', attention_weights, value)


        query = self.query_proj2(query)  # [batch_size, num_agent, hidden_dim]
        key = self.key_proj2(encoded_action)  # [batch_size, num_action, hidden_dim]
        value = self.value_proj2(encoded_action)  # [batch_size, num_action, hidden_dim]

        attention_scores = torch.einsum('bdh,bah->bda', query, key) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        query = torch.einsum('bda,bah->bdh', attention_weights, value)



        query = self.query_proj3(query)
        key = self.key_proj3(encoded_action)  # [batch_size, num_action, hidden_dim]
        logit = torch.einsum('bdh,bah->bda', query, key) / (self.hidden_dim ** 0.5)

        if mask.dim() == 2:
            logit = logit.masked_fill(mask.unsqueeze(0) == 0, float(-1e15))
        else:
            logit = logit.masked_fill(mask == 0, float(-1e15))

        prob = F.softmax(logit, dim = -1)


        return prob


class NodeEmbedding(nn.Module):
    def __init__(self, feature_size, hidden_size, n_representation_obs):
        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size
        self.fcn_1 = nn.Linear(feature_size, hidden_size+10)
        self.fcn_2 = nn.Linear(hidden_size+10, hidden_size)
        self.fcn_3 = nn.Linear(hidden_size, n_representation_obs)
        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)

    def forward(self, node_feature):
        x = F.elu(self.fcn_1(node_feature))
        x = F.elu(self.fcn_2(x))
        node_representation = self.fcn_3(x)
        return node_representation



class ObservationEmbedding(nn.Module):
    def __init__(self, feature_size, hidden_size, n_representation_obs):
        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size
        self.fcn_1 = nn.Linear(feature_size, hidden_size+10)
        self.fcn_2 = nn.Linear(hidden_size+10, hidden_size)
        self.fcn_3 = nn.Linear(hidden_size, n_representation_obs)
        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)

    def forward(self, node_feature):
        x = F.elu(self.fcn_1(node_feature))
        x = F.elu(self.fcn_2(x))
        node_representation = self.fcn_3(x)
        return node_representation
