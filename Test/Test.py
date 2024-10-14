import csv
import pandas as pd
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
from matplotlib import pyplot as plt
from environment import Environment
import networkx as nx
import numpy as np
import torch
from environment import Environment
import torch.nn as nn
import random
from agent import Agent
from Common import Evaluation, arguements
import igraph as ig
from utils.S2V.struct2vec import Struc2Vec
from utils.src.spectral_machinery import WaveletMachine
from utils.src.param_parser import parameter_parser
#import matplotlib.pyplot as plt
import networkx as nx
import torch
import Common.arguements
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
import igraph as ig
import time
from tqdm import tqdm, trange
import Common.arguements
import multiprocessing
from joblib import Parallel,delayed
# # at()
# #     # rewards = torch.from_numpy(np.vstack(batch.reward)).float()
# #     # print(f"{type((batch.obs))}", batch.obs)
# # degree_hist1 = [0,0,2,1,1,0,3,3,2,4,3,1,5,1,3,0,3,1,2,1,0,1,1,0,0,1]
# # degree_hist2 = [0,0,2,1,1,0,3,2,1,4,3,3,4,3,2,0,1,1,2,2,0,1,1,1,1]
# # print(degree_hist1[0])
# # G = nx.read_gexf('fig/cop/20.gexf')
# # print(nx.degree_histogram(G))
# # print(int(list(G.nodes())))
# # import scipy.io as sio
# #
# # adata = sio.mmread('fig/Face/socfb-Amherst41.mtx')
# # v
# #
# # print(adata)


import networkx as nx
import matplotlib.pyplot as plt



# #按照时间序列排序并保存为csv文件
# dataset = pd.read_csv("fig/Coauthor/ca-cit-HepPh.edges", delim_whitespace=True,header=0,names=["Source", "Target", "Weight", "Timestamps"])
# df_loc = dataset.sort_values(axis=0, by='Timestamps', ascending=True)
# df_loc.to_csv("fig/Coauthor/ca-cit-HepPh.csv", index=False)

# #按照选出来的节点进行数据筛选
# df1 = pd.read_csv("fig/Coauthor/ca-cit-HepPh.csv")
# # input1.csv是那个小文件，其中他们有一行或者若干行存储的特征参数相同
# df2 = pd.read_csv("fig/Coauthor/Ph_node.csv")
# value = df2['node'].values.tolist()
# #随机选节点
# # sample_num = 200
# # value2 = random.sample(value, sample_num)
# index = df1['Source'].isin(value)
# # print(index)
# outfile = df1[index]
# # index2 = outfile['Target'].isin(df2['node'])
# index2 = outfile['Target'].isin(value)
# outfile2 = outfile[index2]
# outfile2.to_csv('fig/Coauthor/outfile_Ph.csv', index=False)



# df = pd.read_csv("fig/Coauthor/ca-cit-HepTh.csv")
# df1 = df.head(6515)
# df1.to_csv("fig/Coauthor/ca-cit-HepTh_6515.csv")


# dataset1 = pd.read_csv("fig/Coauthor/11.csv")
# value = dataset1['node'].values.tolist()
# print(value)
# # print(d1)

# # # print('df_loc=', df_loc)
# df_loc.to_csv('forall',header=True,index=False)
# # # df1 = dataset[(dataset["Timestamps"] == 1015887601)]
# # # dataset = pd.read_csv("fig/infect/ia-facebook-wall-wosn-dir.txt", delim_whitespace=True,header=0,names=["Source", "Target", "Timestamps"])
# # # dataset.drop([1,2])
 #保存为csv格式
# # # df1 = dataset[(dataset["Source"] < 200) & (dataset["Source"] >100) & (dataset["Target"] < 20) & (dataset["Target"] > 100)]
# # # df2 = pd.merge()
# # # df1 = dataset[(dataset["Source"] in dataset1) & (dataset["Target"] in dataset1)]
# # # df1.to_csv("fig/infect/ia-facebook-wall-wosn-dir100.csv", index=False)
#




#将第一年中的所有节点中随机选取150个然后在总数据里晒出来所有Source和Target中都包含的行。

#
# # input.csv是那个大文件，有很多很多行
# df1 = pd.read_csv("fig/Coauthor/ca-cit-HepPh.csv")
#
# # input1.csv是那个小文件，其中他们有一行或者若干行存储的特征参数相同
# df2 = pd.read_csv("fig/Coauthor/1993node.csv")
#
# value = df2['Id'].values.tolist()
# sample_num = 200
# value2 = random.sample(value, sample_num)
# # print(value2)
# # 加encoding=‘gbk’是因为文件中存在中文，不加可能出现乱码
# # index = df1['Source'].isin(df2['Id'])
# index = df1['Source'].isin(value)
# # print(index)
# #
# outfile = df1[index]
# #
# # index2 = outfile['Target'].isin(df2['node'])
# index2 = outfile['Target'].isin(value)
# #
# outfile2 = outfile[index2]
# #
# outfile2.to_csv('fig/Coauthor/1993_allnode_outfile.csv', index=False)





# 真实网络演化


# 读取CSV文件
# df = pd.read_csv('fig/Coauthor/1993_allnode_outfile2.csv')
# #
# # 创建无向图
# G = nx.Graph()
# year = 1993
#
# # 遍历数据
# for index, row in df.iterrows():
#
#     source = row['Source']
#     target = row['Target']
#     Weight = row['Weight']
#     Timestamps = row['Timestamps']
#     G.add_node(source)
#     G.add_node(target)
#
#     edge_data = {'Weight': Weight, 'Timestamps': Timestamps} # 假设属性在第三列和第四列
#     #每过一年，给所有边的权重减0.5
#     if Timestamps != year:
#         # pos = nx.spring_layout(G)
#         # nx.draw(G,pos, with_labels=True, font_weight='bold', node_size=300, node_color='skyblue',
#         #         font_color='black', width=2.0,
#         #         font_size=5)
#         # plt.show()
#         nx.write_gexf(G, 'fig/Coauthor/Year{}.gexf'.format(year))
#         for u, v, data in list(G.edges(data=True)):
#             # if 'Weight' in data:
#                 data['Weight'] -= 0.5
#                 if data['Weight'] == 0:
#                     G.remove_edge(u, v)
#
#
#         year = Timestamps
#
#     # 如果边不存在，则添加
#     if not G.has_edge(source, target):
#         G.add_edge(source, target, **edge_data)
#
#     else:
#         if G[source][target]['Timestamps'] != year:
#             G[source][target]['Timestamps']=year
#             G[source][target]['Weight'] += 0.5
#
#
#     # 删除权重为0的边
# # nx.draw(G,pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue',
# #         font_color='black',
# #         font_size=10)
# # plt.show()
# nx.write_gexf(G, 'fig/Coauthor/Year{}.gexf'.format(year))
# 最终的网络拓扑结构保存在变量 G 中


# import requests
# from bs4 import BeautifulSoup
# import networkx as nx
#
#
# def fetch_cvpr_paper_data(year):
#     base_url = f'https://openaccess.thecvf.com/CVPR{year}/'
#     response = requests.get(base_url)
#
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, 'html.parser')
#         papers = soup.find_all('dt', class_='ptitle')
#
#         paper_data = []
#         for paper in papers:
#             paper_title = paper.find('a').text.strip()
#             paper_authors = paper.find_next('dd').text.strip().split(', ')
#             paper_data.append({'title': paper_title, 'authors': paper_authors})
#
#         return paper_data
#     else:
#         print(f"Failed to fetch data from CVPR {year} website.")
#
#
# def generate_author_network(paper_data):
#     G = nx.Graph()
#
#     for paper in paper_data:
#         authors = paper['authors']
#         for i in range(len(authors)):
#             for j in range(i + 1, len(authors)):
#                 G.add_edge(authors[i], authors[j], paper=paper['title'])
#
#     return G
#
#
# def main():
#     years = range(2012, 2023)  # Adjust the range based on the years you are interested in
#
#     full_author_network = nx.Graph()
#
#     for year in years:
#         paper_data = fetch_cvpr_paper_data(year)
#         author_network = generate_author_network(paper_data)
 #         full_author_network.add_edges_from(author_network.edges(data=True))
#
#     # You can now use full_author_network for analysis or visualization
#     # For example, you can print the nodes and edges:
#     print("Nodes:", full_author_network.nodes())
#     print("Edges:", full_author_network.edges())
#
#     # Or visualize the network
#     import matplotlib.pyplot as plt
#     pos = nx.spring_layout(full_author_network)
#     nx.draw(full_author_network, pos, with_labels=True, font_size=8, font_color='black', node_size=10, alpha=0.5)
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()
# def generate_ring_lattice(n, k):
#     G = nx.generators.random_graphs.watts_strogatz_graph(n, k, 0)
#     return G
#
#
# n = 100 # Number of nodes
# k = 10  # Degree of each node
# G = generate_ring_lattice(n, k)
# env = Environment(G)
# # G1 = nx.erdos_renyi_graph(100,0.1)
# nx.write_gexf(G, 'fig/Test/1000modularityt{}.gexf')
# # nx.draw(G,with_labels=True, font_size=8, font_color='black', node_size=10, alpha=0.5)
# # plt.show()
# # print(Evaluation.Modularity(G))
# for agent_i in range(n):
#     print(env.get_reward(G,agent_i))
    # print(env.get_reward_4(G, agent_i))
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import community  # 这是 python-louvain 库，需要额外安装
#
# # 1. 读取 CSV 文件，构建无向图
# df = pd.read_csv('fig/node100 d5 5 2 0 0.95 step10/109911.csv')
# G = nx.from_pandas_edgelist(df, 'Source', 'Target', create_using=nx.Graph())
#
# # 2. 使用 Louvain 算法检测社区
# partition = community.best_partition(G)
#
# pos = nx.spring_layout(G)  # 使用 Spring layout 作为初始布局
#
# # 将节点按社区分组
# # node_groups = {}
# # for node, group in partition.items():
# #     if group not in node_groups:
# #         node_groups[group] = []
# #     node_groups[group].append(node)
# #
# # # 根据社区分组调整节点位置
# # for group, nodes in node_groups.items():
# #     subgraph = G.subgraph(nodes)
# #     subgraph_pos = nx.spring_layout(subgraph, k=0.1)  # 调整子图布局，k 控制节点之间的距离
# #     pos.update(subgraph_pos)
# #
# # # 4. 可视化图形
# # node_colors = [partition[node] for node in G.nodes()]
# #
# # plt.figure(figsize=(10, 8))
# # nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Set1, node_size=200)
# # nx.draw_networkx_edges(G, pos, alpha=0.5)
# # plt.title('Graph with Louvain Community Detection (Nodes Closer in Same Community)')
# # plt.axis('off')
# # plt.show()
# # 3. 将节点根据社区分配不同的颜色
# node_colors = [partition[node] for node in G.nodes()]
#
# # 4. 可视化图形
# pos = nx.spring_layout(G)  # 使用 Spring layout 来布局图形
#
# plt.figure(figsize=(10, 8))
# nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Set1, node_size=200)
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.title('Graph with Louvain Community Detection')
# plt.axis('off')
# plt.show()


#
# epsilon = np.interp(10 , [0, 3],[3, 1])
# print(epsilon)


# def loading2(n,s,a,d):
#     agentVecadd = []
#     agentVecdlt = []
#     cur_n = n
#     n_state = s
#     n_action = a
#     device = d
#     # for i in range(cur_n):
#     addAgent: Agent = Agent(n_input=n_state,
#               n_output=n_action,device=device,
#               mode='train')
#     # addAgent.online_net.load_state_dict(torch.load("fig/pre/addpara/addpara{}.pt".format(i)))
#     dltAgent: Agent = Agent(n_input=n_state,
#                             n_output=n_action, device=device,
#                             mode='train')
#     # dltAgent.online_net.load_state_dict(torch.load("fig/pre/dltpara/dltpara{}.pt".format(i)))
#     # 初始化agent的参数
#     # oneNewAgent.setAgentID(agentID)
#     agentVecadd.append(addAgent)
#     agentVecdlt.append(dltAgent)
# def loading(n,s,a,d):
#     agentVecadd = []
#     agentVecdlt = []
#     cur_n = n
#     n_state = s
#     n_action = a
#     device = d
#     for i in range(cur_n):
#         addAgent: Agent = Agent(n_input=n_state,
#                   n_output=n_action,device=device,
#                   mode='train')
#         # addAgent.online_net.load_state_dict(torch.load("fig/pre/addpara/addpara{}.pt".format(i)))
#         dltAgent: Agent = Agent(n_input=n_state,
#                                 n_output=n_action, device=device,
#                                 mode='train')
#         # dltAgent.online_net.load_state_dict(torch.load("fig/pre/dltpara/dltpara{}.pt".format(i)))
#         # 初始化agent的参数
#         # oneNewAgent.setAgentID(agentID)
#         agentVecadd.append(addAgent)
#         agentVecdlt.append(dltAgent)
#
#
# import threading
#
#
# # 定义一个函数来处理节点的计算任务
# def process_node(node,agentVecadd,agentVecdlt):
#     # 在这里进行节点的计算操作
#     addAgent: Agent = Agent(n_input=128,
#                            n_output=100, device="cpu",
#                            mode='train')
#     # addAgent.online_net.load_state_dict(torch.load("fig/pre/addpara/addpara{}.pt".format(i)))
#     dltAgent: Agent = Agent(n_input=128,
#                            n_output=100, device="cpu",
#                            mode='train')
#     # dltAgent.online_net.load_state_dict(torch.load("fig/pre/dltpara/dltpara{}.pt".format(i)))
#     # 初始化agent的参数
#     # oneNewAgent.setAgentID(agentID)
#     agentVecadd.append(addAgent)
#     agentVecdlt.append(dltAgent)
#     print("Processing node:", node)
#
#
# # 定义一个函数来创建并启动多个线程
# def run_threads(nodes):
#     threads = []
#     agentVecadd = []
#     agentVecdlt = []
#     for node in nodes:
#         thread = threading.Thread(target=process_node, args=(node,agentVecadd,agentVecdlt))
#         thread.start()
#         threads.append(thread)
#
#     # 等待所有线程执行完毕
#     for thread in threads:
#         thread.join()
# if __name__ == "__main__":
    # # 假设nodes是你要处理的节点列表
    # start_time = time.time()
    # nodes = list(range(1,100+1))
    #
    # # nodes = list(network_graph.keys())
    #
    # # 创建一个进程池，指定最大进程数为 4
    # pool = multiprocessing.Pool(processes=4)
    # agentVecadd = []
    # agentVecdlt = []
    # # 遍历所有节点并进行计算
    # for node in nodes:
    #     # 获取当前节点的邻居节点
    #     # neighbors = network_graph[node]
    #     # 向进程池提交计算任务
    #     pool.apply_async(process_node, args=(node,agentVecadd,agentVecdlt))
    #     # 如果节点有邻居节点，也进行计算
    #     # for neighbor in neighbors:
    #     #     pool.apply_async(compute, args=(neighbor,))
    #
    # # 关闭进程池
    # pool.close()
    # # 等待所有进程结束
    # pool.join()
    #
    # print("All computations finished.")
    #
    # end_time = time.time()
    # print(end_time-start_time)
    # print(multiprocessing.cpu_count())
    # import multiprocessing

    # from math import sqrt

    # from multi.single_thread_check_prime import check_prime

# CHECK_NUMBERS = 1000000
# NUM_PROCESSES =  4
#
#
# def worker(start,end,agentVecdlt,agentVecadd):
#      for node in range(start,end):
#        addAgent: Agent = Agent(n_input=128,
#                            n_output=100, device="cpu",
#                            mode='train')
#     # addAgent.online_net.load_state_dict(torch.load("fig/pre/addpara/addpara{}.pt".format(i)))
#        dltAgent: Agent = Agent(n_input=128,
#                            n_output=100, device="cpu",
#                            mode='train')
#     # dltAgent.online_net.load_state_dict(torch.load("fig/pre/dltpara/dltpara{}.pt".format(i)))
#     # 初始化agent的参数
#     # oneNewAgent.setAgentID(agentID)
#        agentVecadd.append(addAgent)
#        agentVecdlt.append(dltAgent)
#        # print("Processing node:", node)
#
#      # return agentVecadd,agentVecdlt
#   # result_mq.put(total)
#
#
#
# def run():
#     params = 100  # [(2, 250001), (250001, 500000), (500000, 749999), (749999, 999998), (999998, 1000001)]
#     result_mq = multiprocessing.Queue()
#     processes = []
#
#     for i in range(NUM_PROCESSES):
#         process = multiprocessing.Process(target=worker, args=(250*i,250*(i+1),agentVecadd,agentVecdlt))
#         processes.append(process)
#         process.start()
#
#     for process in processes:
#         process.join()
#
#     # total = 0
#     # for i in range(NUM_PROCESSES):
#     #     count = result_mq.get()
#     #     total += count
#     # print(total)
#
#     # return total
#     # pool = multiprocessing.Pool(processes=NUM_PROCESSES)
#     # result = pool.map(worker, params)
#     # total = sum(result)
#     # print(total)
#     # return total
#
#
# if __name__ == "__main__":
#     import timeit
#     agentVecadd = []
#     agentVecdlt = []
#     run()
#     print(timeit.timeit("run()", "from __main__ import run", number=1))
# start_time = time.time()
# agentVecadd = []
# agentVecdlt = []
# for i in range(10000):
#     addAgent: Agent = Agent(n_input=128,
#                                n_output=100, device="cpu",
#                                mode='train')
#         # addAgent.online_net.load_state_dict(torch.load("fig/pre/addpara/addpara{}.pt".format(i)))
#     dltAgent: Agent = Agent(n_input=128,
#                                n_output=100, device="cpu",
#                                mode='train')
#         # dltAgent.online_net.load_state_dict(torch.load("fig/pre/dltpara/dltpara{}.pt".format(i)))
#         # 初始化agent的参数
#         # oneNewAgent.setAgentID(agentID)
#     agentVecadd.append(addAgent)
#     agentVecdlt.append(dltAgent)
#
# end_time = time.time()
# print(end_time-start_time)
def get_reward(G, n):
    One_ego_graph = nx.ego_graph(G, n, 1)

    g = ig.Graph.from_networkx(G)

    pr = g.personalized_pagerank()

    nodes = list(nx.nodes(One_ego_graph))
    w = 1

    bo = [pr[int(i)] for i in nodes]

    bo_sum = sum(bo)

    # bc = g.betweenness()
    # # print(bc)
    # # print(nx.betweenness_centrality(G))
    # br = bc[n]
    # mix = w * bo_sum + (1 - w) * br
    mix = w * bo_sum

    return mix


# def generate_ring_lattice(n, k):
#     G = nx.generators.random_graphs.watts_strogatz_graph(n, k, 0)
#     return G
# #
# #
# # # args = arguements.get_args()
# n = 300  # Number of nodes
# k = 5  # Degree of each node
# # # #
# # # # Generate the regular ring lattice
# G = generate_ring_lattice(n, k)
# # #
# # print(Evaluation.averageDistance(G))
# # print(Evaluation.globalClustering(G))
#
# # G = nx.erdos_renyi_graph(200,0.05)
# nx.write_gexf(G, 'fig/RoleTrans/Test1.gexf')
# print(G.size())
# print(f"Modularity:{Evaluation.Modularity(G)}")
# print(f"Clustering:{Evaluation.globalClustering(G)}")
# print(nx.is_connected(G))
# print(f"ShortestPath:{Evaluation.averageDistance(G)}")

# print(G.size())
# print(f"Modularity:{Evaluation.Modularity(G)}")
# print(f"Clustering:{Evaluation.globalClustering(G)}")
# print(f"ShortestPath:{Evaluation.averageDistance(G)}")
# print(get_reward(G, 1))
# G=nx.read_gexf('fig/CommunityV3/300/tar31.gexf')
# print(Evaluation.globalClustering(G))
# print(Evaluation.globalClustering2(G))
# print(Evaluation.averageDistance(G))
# print(Evaluation.averageDistance2(G))
#
# env = Environment(G)
# print(env.observe_2(G))
from collections import Counter

# def max_same_values_count(dictionary):
#     # 使用Counter统计值的频率
#     hashable_values = [tuple(arr) for arr in dictionary.values()]
#
#     # 使用Counter统计元组的频率
#     value_counts = Counter(hashable_values)
#     if not value_counts:
#         return 0
#     # 找到最大的频率
#     max_count = max(value_counts.values())
#     return max_count
#
#
# max_count = max_same_values_count(env.observe_2(G))
# print("最大相同值的个数:", max_count)
# print(G.size())
# print( random.random())

# def selectincreasingnode(static_graph):
#     import networkx as nx
#
#     # 假设你有五个图的列表，每个图表示为一个 NetworkX 图对象
#     graphs = static_graph
#
#     # 计算每个图的 PageRank
#     pageranks = []
#     for graph in graphs:
#         pagerank = nx.pagerank(graph)
#         pageranks.append(pagerank)
#
#     # 找出在不同图中 PageRank 递增的节点
#     increasing_nodes = {}
#     for i, graph_pagerank in enumerate(pageranks):
#         sorted_pagerank = sorted(graph_pagerank.items(), key=lambda x: x[1])
#         nodes = [node for node, _ in sorted_pagerank]
#
#         # 如果是第一个图，则直接加入节点
#         if i == 0:
#             for node in nodes:
#                 increasing_nodes[node] = [i+1]
#         else:
#             # 对于其他图，判断节点是否 PageRank 递增
#             for node, _ in sorted_pagerank:
#                 if pageranks[i - 1][node] < pageranks[i][node]:
#                     # 如果递增，则将该节点加入递增节点列表
#                     if node not in increasing_nodes:
#                         increasing_nodes[node] = [i]
#                     else:
#                         increasing_nodes[node].append(i)
#
#     # 输出在不同图中 PageRank 递增的节点
#     for node, graphs in increasing_nodes.items():
#         if len(graphs) == 5:  # 在所有图中都递增的节点
#             print(f"在所有图中递增的节点: {node}")
#         # else:
#         #     print(f"在图{', '.join(map(str, graphs))}中递增的节点: {node}")
#     import networkx as nx
#
#     # 假设你有五个图的列表，每个图表示为一个 NetworkX 图对象

#     graphs = static_graph
#
#     # 计算每个图的 PageRank
#     pageranks = []
#     for graph in graphs:
#         pagerank = nx.pagerank(graph)
#         pageranks.append(pagerank)
#
#     # 找出在不同图中 PageRank 递增最大的节点
#     max_increasing_node = None
#     max_increase = 0
#
#     for node in pageranks[0].keys():
#         # increasing = True
#         # for i in range(1, len(pageranks)):
#         #     if pageranks[i][node] <= pageranks[i - 1][node]:
#         #         increasing = False
#         #         break
#         # if increasing:
#             increase_amount = pageranks[0][node] -pageranks[-1][node]
#             if increase_amount > max_increase:
#                 max_increase = increase_amount
#                 max_increasing_node = node
#
#     # 输出在所有图中 PageRank 递增最大的节点
#     print(f"在所有图中 PageRank 递增最大的节点: {max_increasing_node}")
#     print(f" PageRank 递增: {max_increase}")
#
#
# graphs = []
# for i in range(40):
#     # print(f"Graph {i+1}:")
#     # print("Nodes:", graph.nodes())
#     G=nx.read_gexf( f"fig/RoleTrans/Episode0_step{i}.gexf")
#     graphs.append(G)
#
# selectincreasingnode(graphs)
# G = nx.erdos_renyi_graph(20, 0.2)
# print(nx.degree_assortativity_coefficient(G))
# # 计算节点度
# # degrees = dict(G.degree())
# #
# # # 构建节点度的列表
# # degree_list = np.array([degrees[node] for node in G.nodes()])
# # # print(degree_list)
# # # 计算相关系数矩阵
# # correlation_matrix = np.corrcoef(degree_list,degree_list)
# #
# # # 检查相关系数矩阵的形状
# # print("Correlation matrix shape:", correlation_matrix.shape)
# #
# # # 绘制相关系数矩阵的热力图
# # plt.figure(figsize=(8, 6))
# # plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
# # plt.colorbar()
# # plt.title('Degree Correlation Matrix')
# # plt.show()
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 创建一个拓扑图



# 创建一个简单的图


# 计算节点的度
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
#
# # 生成ER图
#
# G = nx.read_gexf("fig/CommunityV3/300/Tar31.gexf")
# # 计算节点的度
# n = G.number_of_nodes()
#
# import networkx as nx
# from collections import (defaultdict)
# # 构造或加载一个networkx图
# # G = nx.gnm_random_graph(100, 200)
# # 例如，生成一个含有100个节点和200条边的随机图
# # 初始化度相关矩阵
# degree_correlation_matrix = defaultdict(int)
# # 获取图中所有节点的度数
# node_degrees = dict(G.degree())
# # 遍历图中的每条边
# print(node_degrees)
# for u, v in G.edges():
#     deg_u = node_degrees[u]
#     deg_v = node_degrees[v]
#     # 更新度相关矩阵
#     degree_correlation_matrix[(deg_u, deg_v)] += 1
# # print(G.edges())
#
# if deg_u != deg_v:
# # 对于无向图，(i, j)和(j, i)是相同的
#   degree_correlation_matrix[(deg_v, deg_u)] += 1
# # 归一化度相关矩阵
# total_edges = G.number_of_edges()
# # print(degree_correlation_matrix)
# for key in degree_correlation_matrix:
#     degree_correlation_matrix[key] /= (2*total_edges)
#     # 输出结果（部分）
# for key in sorted(degree_correlation_matrix):
#     print(f"Degree {key}: {degree_correlation_matrix[key]}")
#
#
# # import networkx as nx
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from collections import defaultdict
# # # 生成一个随机图作为例子
# # G = nx.gnm_random_graph(100, 200)
# # 基于前面的代码计算度相关矩阵（为简化，这里不再展示那部分代码）
# # 初始化度相关矩阵（示例代码已给出，此处省略）
# # ---到此为止，你应已计算得degree_correlation_matrix---
# # 确定矩阵大小
# max_degree = max(max(k) for k in degree_correlation_matrix.keys())
# matrix_size = max_degree + 1
# # 加1因为度数从0开始 # 构建矩阵
# degree_matrix = np.zeros((matrix_size, matrix_size))
# # 填充矩阵
# for (i, j), value in degree_correlation_matrix.items():
#     degree_matrix[i, j] = value
#     # 生成热力图
# plt.imshow(degree_matrix, cmap='hot', interpolation='nearest',origin='lower')
# plt.colorbar()
# # 显示颜色条
# # plt.title('Degree Correlation Matrix Heatmap')
# plt.xlabel('Degree K1',size = 90)
# plt.ylabel('Degree K2',size = 90)
# plt.show()





# # 获取所有节点的度
# degree_counts = nx.degree_histogram(G)
#
# # 计算可能的边的数量
# N = len(G.nodes())
# possible_edges = [(i * degree_counts[i]) for i in range(len(degree_counts))]
#
# # 计算每对度数之间的链接概率
# link_probabilities = np.zeros((len(degree_counts), len(degree_counts)))
# for i in range(len(degree_counts)):
#     for j in range(len(degree_counts)):
#         if possible_edges[i] == 0 or possible_edges[j] == 0:
#             link_probabilities[i][j] = 0
#         else:
#             link_probabilities[i][j] = min(degree_counts[i], degree_counts[j]) / (N * (N - 1) / 2)
#
# # 构建相关系数矩阵
# correlation_matrix = np.corrcoef(link_probabilities)
#
# # 绘制相关系数矩阵的热力图
# plt.figure(figsize=(8, 6))
# plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title('Link Probability Correlation Heatmap')
# plt.xlabel('Degree')
# plt.ylabel('Degree')
# plt.show()
def observe_2(G):
    settings = parameter_parser()
    machine = WaveletMachine(G, settings)

    machine.create_embedding()
    # print(machine.transform_and_save_embedding())
    return machine.transform_and_save_embedding()
G = nx.read_gexf("fig/CommunityV3/300/Tar31.gexf")
observation = observe_2(G)
print(observation)
values = list(observation.values())
    # 比较第一个值与其他所有值
print(values[0] - values[1])
for value in values[1:]:
    # 使用numpy的array_equal函数比较两个数组是否相等
    if not np.array_equal(values[0], value):
        print(False)
print(True)
# def check_value_same(my_dict):
#     # 使用set()函数将字典的值转换为集合，这样可以去除重复的值
#     # 然后使用len()函数检查集合的长度
#     # 如果长度为1，说明所有值都相同；否则，值不相同
#     return len(set(my_dict.values())) == 1
#
# # 测试
# my_dict = {'key1': 'value', 'key2': 'value', 'key3': 'value'}
# print(check_value_same(observation))  # 输出：True

# my_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
# print(check_value_same(my_dict))  # 输出：False