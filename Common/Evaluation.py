import networkx as nx
#import igraph as ig
import copy
import igraph as ig
#全局聚集系数
from igraph import Graph
from community import community_louvain
import numpy as np
#from networkx.generators import community

# def kk_core(G):
#     def part(G):
#       sub = copy.deepcopy(G)
#       degree_sequence = [d for n, d in sub.degree()]
#       k=max(degree_sequence)
#       for i in range(len(degree_sequence)):
#         if(degree_sequence[i]<k):
#             sub.remove_node(i)
#       node = list(sub)
#       sum_all=[]
#       closeness_all=nx.closeness_centrality(G)
#       for i in closeness_all.keys():
#           sum_all.append(i)
#           if i in node:
#
#       CC_all = sum(sum_all) / len(sum_all)
#
#
#
#     average = []
#     degree_sequence = [d for n, d in G.degree()]
#     # G_ran=nx.configuration_model(degree_sequence)
#     for i in range(9):
#         G_ran = nx.random_degree_sequence_graph(degree_sequence)
#         average.append(part(G_ran.to_undirected()))
#     avg = sum(average) / len(average)
#     # k_core = nx.core_number(G)
#     # k_max = max(k_core.values())
#     # k_cores = [G.subgraph(n for n, d in k_core.items() if d >= k) for k in range(k_max + 1)]
#     # print(k_cores)
#     return k_cores


def globalClustering2(G):
    # nodes, edges = highDimensionData2nodeAndEdge(filepath)
    # G = nx.Graph()
    # G.add_nodes_from(nodes)
    # G.add_edges_from(edges)
    transitivity = nx.algorithms.cluster.transitivity(G)
    # transitivity = nx.average_clustering(G)
    return transitivity
def globalClustering(G):
    # nodes, edges = highDimensionData2nodeAndEdge(filepath)
    # G = nx.Graph()
    # G.add_nodes_from(nodes)
    # G.add_edges_from(edges)
    # transitivity = nx.algorithms.cluster.transitivity(G)
    transitivity = nx.average_clustering(G)
    return transitivity

#平均最短路径
def averageDistance2(G):
        all_pairs_shortest_path_length = nx.all_pairs_shortest_path_length(G)
        count = 0
        length = 0
        for node1 in all_pairs_shortest_path_length:
            for node2 in node1[1]:
                if node1[1][node2] != 0:
                    count += 1
                    length += node1[1][node2]
        averageLength = length / count
        return averageLength

def averageDistance(G):
    return nx.average_shortest_path_length(G)

#模块度
def Modularity(G):
      partition = community_louvain.best_partition(G)
      # modularity_cur = nx.algorithms.community.modularity(G, [set(partition.keys())], partition.values())
      modularity = community_louvain.modularity(partition, G)

      return modularity


def CP_coef(G):
    def part(G):
        sub = copy.deepcopy(G)
        degree_sequence = [d for n, d in sub.degree()]
        k=max(degree_sequence)
        for i in range(len(degree_sequence)):
            if(degree_sequence[i]<k):
                sub.remove_node(i)
        # sub=nx.k_core(G,k=None)
        # sub=set
        #是要kcore这些点在原图当中的closene_centrality还是在kcore当中的closeness_centrality
        # closeness_sub=nx.closeness_centrality(sub)
        node = list(sub)
        closeness_all=nx.closeness_centrality(G)
        sum_sub=[]
        sum_all=[]
        for i in closeness_all.keys():
            if i in node:
                sum_sub.append(closeness_all[i])
        CC_core=sum(sum_sub)/len(sum_sub)
        for i in closeness_all.values():
            sum_all.append(i)
        CC_all=sum(sum_all)/len(sum_all)

        return CC_core/CC_all
    average=[]
    degree_sequence = [d for n,d in G.degree()]
    # G_ran=nx.configuration_model(degree_sequence)
    for i in range(1000):
         G_ran = nx.random_degree_sequence_graph(degree_sequence)
         average.append(part(G_ran.to_undirected()))
    avg=sum(average)/len(average)
   # G_ran = ig.Graph.Degree_Sequence(out=degree_sequence)

    # return G_ran
    #return part(G)-part(G_ran.to_undirected())
    return part(G)-avg


def CP_coef_2(G,core):
    def part(G):
        # sub = copy.deepcopy(G)
        # degree_sequence = [d for n, d in sub.degree()]
        # sub=nx.k_core(G,k=None)
        # sub=set
        #是要kcore这些点在原图当中的closene_centrality还是在kcore当中的closeness_centrality
        # closeness_sub=nx.closeness_centrality(sub)
        # g = ig.Graph.from_networkx(G)
        closeness_all=nx.closeness_centrality(G)
        sum_sub=[]
        sum_all=[]
        for i in closeness_all.keys():
            if i in core:
                sum_sub.append(closeness_all[i])
        CC_core=sum(sum_sub)/len(sum_sub)
        for i in closeness_all.values():
            sum_all.append(i)
        CC_all=sum(sum_all)/len(sum_all)

        return CC_core/CC_all
    average=[]
    degree_sequence = [d for n,d in G.degree()]
    # G_ran=nx.configuration_model(degree_sequence)
    for i in range(10):
         G_ran = nx.random_degree_sequence_graph(degree_sequence)
         average.append(part(G_ran.to_undirected()))
    avg=sum(average)/len(average)
   # G_ran = ig.Graph.Degree_Sequence(out=degree_sequence)

    # return G_ran
    #return part(G)-part(G_ran.to_undirected())
    return part(G)-avg

def CP_coef_3(G,core):
    g = ig.Graph.from_networkx(G)
    def part(g):
        # sub = copy.deepcopy(G)
        # degree_sequence = [d for n, d in sub.degree()]
        # sub=nx.k_core(G,k=None)
        # sub=set
        #是要kcore这些点在原图当中的closene_centrality还是在kcore当中的closeness_centrality
        # closeness_sub=nx.closeness_centrality(sub)

        cl=g.closeness()
        bo = [cl[int(i)] for i in core]
        CC_core=sum(bo)/len(bo)
        CC_all=sum(cl)/len(cl)

        return CC_core/CC_all
    average=[]
    degree_sequence = g.degree()
    # G_ran=nx.configuration_model(degree_sequence)
    for i in range(1000):
         g_ran=Graph.Degree_Sequence(degree_sequence)

         average.append(part(g_ran))
    # print(sum(average),len(average))
    avg_np=np.array(average)
    avg=np.nanmean(avg_np)
    # print(avg)
   # G_ran = ig.Graph.Degree_Sequence(out=degree_sequence)

    # return G_ran
    #return part(G)-part(G_ran.to_undirected())
    # print(part(g),avg)
    return part(g)-avg

def core_periphery_coefficient(G):
    k_core = nx.core_number(G)
    k_max = max(k_core.values())
    k_cores = [G.subgraph(n for n, d in k_core.items() if d >= k) for k in range(k_max + 1)]
    cc = [nx.average_clustering(k_cores[i]) for i in range(len(k_cores))]
    ccp = cc[k_max] / sum(cc) - (sum(cc) - cc[k_max]) / (len(cc) - 1) / sum(cc)
    return ccp

# def core_periphery_coefficient_2(G):
#     def cc(G):
#         k_core = nx.core_number(G)
#         k_max = max(k_core.values())
#         k_cores = [G.subgraph(n for n, d in k_core.items() if d >= k) for k in range(k_max + 1)]
#
#         return [nx.average_clustering(k_cores[i]) for i in range(len(k_cores))]
#     ccp = cc[k_max] / sum(cc) - (sum(cc) - cc[k_max]) / (len(cc) - 1) / sum(cc)
#     return ccp