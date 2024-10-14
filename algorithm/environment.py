# import networkx as nx
# import numpy as np
# import torch
# import pulp
import time
import igraph as ig
from GraphEmbedding.S2V.struct2vec import Struc2Vec
from GraphEmbedding.machinery.spectral_machinery import WaveletMachine
from GraphEmbedding.machinery.param_parser import parameter_parser
#import matplotlib.pyplot as plt
import networkx as nx
import torch
from Common import arguements
import fractions
import numpy as np

# import on 2024.5.16
import argparse
import os.path as osp
import time
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
"""
This file contains the definition of the environment
in which the agents are run.
"""


class Environment():

    def __init__(self):
        # self.graphs = graph
        self.args= arguements.get_args()


    def reset(self):
        def generate_ring_lattice(n, k):
            G = nx.generators.random_graphs.watts_strogatz_graph(n, k, 0)
            return G

        # args = arguements.get_args()
        n = self.args.num_of_node  # Number of nodes
        k = self.args.degree # Degree of each node

        # Generate the regular ring lattice
        G = generate_ring_lattice(n, k)
        # G = nx.read_gexf('fig/cop/20.gexf')
        return G

    def reset2(self):
        # G = nx.erdos_renyi_graph(300, 0.1)
        G = nx.read_gexf('Tar31.gexf')
        int_graph = nx.relabel.convert_node_labels_to_integers(G)
        return int_graph




    def observe(self,G):
        """Returns the current observation that the agent can make
                 of the environment, if applicable.
        """
        # Two_ego_graph = nx.ego_graph(G, n, 2)
        # g = ig.Graph.from_networkx(G)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s2v_model = Struc2Vec(G, 10, 80, workers=4, verbose=40)# init model
        s2v_model.train(window_size=5, iter=3) # train model
        observation = s2v_model.get_embeddings()#todo array to tensor
        return observation

    def observe_2(self,G):
        settings = parameter_parser()
        machine = WaveletMachine(G, settings)

        machine.create_embedding()
        # print(machine.transform_and_save_embedding())
        return machine.transform_and_save_embedding()

    def observe_3(self, G):
            data = from_networkx(G)

            if data.edge_index is None:
                raise ValueError("The input graph does not contain edge indices.")

            if data.x is None:
                num_nodes = G.number_of_nodes()
                data.x = torch.eye(num_nodes)

            # GCN inner class
            class GCN(torch.nn.Module):
                def __init__(self, in_channels, hidden_channels, out_channels):
                    super(GCN, self).__init__()
                    self.conv1 = GCNConv(in_channels, hidden_channels)
                    self.conv2 = GCNConv(hidden_channels, out_channels)

                def forward(self, data):
                    x, edge_index = data.x, data.edge_index
                    x = self.conv1(x, edge_index)
                    x = F.relu(x)
                    x = F.dropout(x, training=self.training)
                    x = self.conv2(x, edge_index)
                    return F.log_softmax(x, dim=1)

            model = GCN(in_channels=data.num_node_features,
                        hidden_channels=16,
                        out_channels=32)
            model.eval()
            with torch.no_grad():
                observation = model(data)

            observation_dict = {}
            for i in range(data.num_nodes):
                observation_dict[i] = observation[i].cpu().numpy()

            return observation_dict



    def observe_4(self, G):
            data = from_networkx(G)
            if data.edge_index is None:
                raise ValueError("The input graph does not contain edge indices.")
            if data.x is None:

                num_nodes = G.number_of_nodes()
                data.x = torch.eye(num_nodes)

            # GraphSAGE inner class
            class SAGE(torch.nn.Module):
                def __init__(self, in_channels, hidden_channels, out_channels):
                    super(SAGE, self).__init__()
                    self.conv1 = SAGEConv(in_channels, hidden_channels)
                    self.conv2 = SAGEConv(hidden_channels, out_channels)

                def forward(self, data):
                    x, edge_index = data.x, data.edge_index
                    x = self.conv1(x, edge_index)
                    x = F.relu(x)
                    x = F.dropout(x, training=self.training)
                    x = self.conv2(x, edge_index)
                    return F.log_softmax(x, dim=1)

            model = SAGE(in_channels=data.num_node_features,
                         hidden_channels=16,
                         out_channels=32)
            model.eval()
            with torch.no_grad():
                observation = model(data)

            observation_dict = {}
            for i in range(data.num_nodes):
                observation_dict[i] = observation[i].cpu().numpy()

            return observation_dict


    def observe_node(self, G):


        from node2vec import Node2Vec
        node2vec = Node2Vec(G, dimensions=32, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(vector_size=32, window=10, min_count=1, batch_words=4)
        embeddings = {node: model.wv[node] for node in G.nodes()}

        return  embeddings

    def observe_4_2(self,G):

        data = from_networkx(G)
        # print(data)

        if data.edge_index is None:
            raise ValueError("The input graph does not contain edge indices.")

        # 检查是否存在节点特征
        if data.x is None:
            num_nodes = G.number_of_nodes()
            # data.x = torch.eye(num_nodes)
            adj_matrix = nx.adjacency_matrix(G)


            adj_matrix_tensor = torch.tensor(adj_matrix.todense(), dtype=torch.float32)
            data.x = adj_matrix_tensor
            # print(data.x)

        # GraphSAGE inner class
        class SAGE(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super(SAGE, self).__init__()
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index)
                return F.log_softmax(x, dim=1)

        model = SAGE(in_channels=data.num_node_features,
                     hidden_channels=16,
                     out_channels=32)
        model.eval()
        with torch.no_grad():
            observation = model(data)

        observation_dict = {}
        for i in range(data.num_nodes):
            observation_dict[i] = observation[i].cpu().numpy()

        return observation_dict

    def observe_5(self, G):

            data = from_networkx(G)

            # 检查是否存在边索引
            if data.edge_index is None:
                raise ValueError("The input graph does not contain edge indices.")


            if data.x is None:

                num_nodes = G.number_of_nodes()
                data.x = torch.eye(num_nodes)

            # GAT inner class
            class GAT(torch.nn.Module):
                def __init__(self, in_channels, hidden_channels, out_channels):
                    super(GAT, self).__init__()
                    self.conv1 = GATConv(in_channels, hidden_channels, heads=1)
                    self.conv2 = GATConv(hidden_channels, out_channels, heads=1)

                def forward(self, data):
                    x, edge_index = data.x, data.edge_index
                    x = self.conv1(x, edge_index)
                    x = F.relu(x)
                    x = F.dropout(x, training=self.training)
                    x = self.conv2(x, edge_index)
                    return F.log_softmax(x, dim=1)

            model = GAT(in_channels=data.num_node_features,
                        hidden_channels=16,
                        out_channels=32)
            model.eval()
            with torch.no_grad():
                observation = model(data)

            observation_dict = {}
            for i in range(data.num_nodes):
                observation_dict[i] = observation[i].cpu().numpy()

            return observation_dict
    # def act(self,node):
    #
    #     self.observation[:,node,:]=1
    #     # (s, a)
    #     reward = self.get_reward(self.observation, node)
    #     return reward

    # reverse
    def get_reward_1(self, G,s, n):
            time_1 = time.time()
            One_ego_graph = nx.ego_graph(G,n,1)
            d = nx.to_pandas_edgelist(G).values
            g = ig.Graph(d)
            #pr = nx.pagerank(One_ego_graph,0.85)
            pr=g.personalized_pagerank()
            #print(pr)
            time_2 = time.time()
            #
            nodes = list(nx.nodes(One_ego_graph))
            bo=0.
            # w=0
            if n in s:
              w=1/600
            else:
              w=1
            bo=[pr[int(i)] for i in nodes]

            bo_sum=sum(bo)

            bc = nx.centrality.betweenness_centrality(G)  # todo
            br = bc.get(n)
            # bc = g.betweenness() #todo

            time_3 = time.time()
            mix = w*bo_sum+(1-w)*br

            return mix

    def get_reward_3(self, G, s, n):

        One_ego_graph = nx.ego_graph(G, n, 1)
        g = ig.Graph.from_networkx(G)

        pr = g.personalized_pagerank()

        nodes = list(nx.nodes(One_ego_graph))
        # w=0
        if n in s:
            w = 1 / 600
        else:
            w = 1
        bo = [pr[int(i)] for i in nodes]

        bo_sum = sum(bo)

        bc = g.betweenness()

        br = bc[n]
        mix = w * bo_sum + (1 - w) * br

        return mix
    def get_reward(self, G, n):

        One_ego_graph = nx.ego_graph(G, n, 1)

        g = ig.Graph.from_networkx(G)

        pr = g.personalized_pagerank()

        nodes = list(nx.nodes(One_ego_graph))
        w=1

        bo = [pr[int(i)] for i in nodes]

        bo_sum = sum(bo)

        # bc = g.betweenness()
        # # # print(bc)
        # # # print(nx.betweenness_centrality(G))
        # br = bc[n]
        # mix = w * bo_sum + (1 - w) * br
        mix = bo_sum
        return mix

    def get_reward_4(self, G, n):
        One_ego_graph = nx.ego_graph(G, n, 1)
        nodes = list(nx.nodes(One_ego_graph))
        g = ig.Graph.from_networkx(G)
        pr = g.personalized_pagerank()

        # nodes = g.neighborhood(n,order=1)
        w=1
        # if n in s:
        #     w = 1 / 600
        # else:
        #     w = 1
        bo = [pr[int(i)] for i in nodes]
        #
        bo_sum = 1 - sum(bo)
        # bo_sum = 1 - pr[int(n)]
        # bc = g.betweenness()
        # # # print(bc)
        # br = 2000-bc[n]
        mix = bo_sum

        return mix

    def get_reward_2(self,G,s, n):

            One_ego_graph = nx.ego_graph(G, n, 1)
            pr = nx.pagerank(One_ego_graph, 0.85)
            nodes = list(nx.nodes(One_ego_graph))
            bo = 0.
            # w=0
            if n in s:
                w = 1 / 600
            else:
                w = 1
            for i in nodes:
                bo = bo + int(pr.get(i))

            bc = nx.centrality.betweenness_centrality(G)  # todo
            br = bc.get(n)

            mix = w * bo + (1 - w) * br

            return mix

    def get_reward_degreecentrality(self, G, n):

        dc = nx.degree_centrality(G)
        One_ego_graph = nx.ego_graph(G, n, 1)
        nodes = list(nx.nodes(One_ego_graph))
        w = 1

        bo = [dc[int(i)] for i in nodes]

        bo_sum = sum(bo)

        # bc = g.betweenness()
        # # print(bc)
        # # print(nx.betweenness_centrality(G))
        # br = bc[n]
        # # mix = w * bo_sum + (1 - w) * br
        mix = w * bo_sum
        # mix = br
        return mix
    def get_reward_hits(self, G, n):
        hits_tuple = nx.hits(G)
        hits = hits_tuple[0]
        hits_list = list(hits.values())
        One_ego_graph = nx.ego_graph(G, n, 1)
        nodes = list(nx.nodes(One_ego_graph))
        w = 1

        bo = [hits_list[int(i)] for i in nodes]

        bo_sum = sum(bo)

        # bc = g.betweenness()
        # # print(bc)
        # # print(nx.betweenness_centrality(G))
        # br = bc[n]
        # # mix = w * bo_sum + (1 - w) * br
        mix = w * bo_sum
        # mix = br
        return mix

    def utility_list(self,G,n):
        def get_utility(G,source,target,w):
            Utility = (1/G.degree(target) + (1/(G.degree(target)*(G.degree(source)+1)))
                      - (1/G.degree(source)*(G.degree(source)+1)))*w

            return  Utility
        g = ig.Graph.from_networkx(G)
        one_ego = g.neighborhood(n, order=1)
        one_ego.remove(n)
        sum = 0
        for i in one_ego:
            sum+=(1/G.degree(i))
        two_ego = [x for x in g.neighborhood(n, order=2) if x not in g.neighborhood(n, order=1)]
        dic = {}
        to_nodes = np.array(two_ego).astype(dtype=int).tolist()
        for tovex in to_nodes:
            dic.update({tovex: get_utility(G,n,tovex,sum)})


        return dic

    def utility_list2(self,G,n):
        def get_utility(G,source,target,w):
            Utility = (1/G.degree(target) + (1/(G.degree(target)*(G.degree(source)+1)))
                      - (1/G.degree(source)*(G.degree(source)+1)))*w

            return  Utility
        g = ig.Graph.from_networkx(G)
        one_ego = g.neighborhood(n, order=1)
        one_ego.remove(n)
        sum = 0
        for i in one_ego:
            sum+=(1/G.degree(i))
        two_ego = [x for x in g.neighborhood(n, order=2) if x not in g.neighborhood(n, order=1)]
        dic = {}
        to_nodes = np.array(one_ego).astype(dtype=int).tolist()
        for tovex in to_nodes:
            dic.update({tovex:-get_utility(G,n,tovex,sum)})

        # print(dic)
        return dic



    # def step(self,agentID, action):
    #     next_G=self.graphs.add_edge(agentID,action)
    #     obs = self.observe(next_G)
    #     reward = self.get_reward(next_G,agentID)
    #
    #     return obs,reward
