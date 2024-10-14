import copy
import os
from cProfile import Profile

from matplotlib import pyplot as plt
from environment import Environment
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import random
from agent import Agent
from Common import Evaluation, arguements
import igraph as ig

from tqdm import tqdm, trange
import Common.arguements
import time
def train(G):
    cur_n = nx.number_of_nodes(G)
    # graph_dic = {}
    # #seed = 125
    # #graph_one = graph.Graph(graph_type=args.graph_type, cur_n=20, p=0.15,m=4, seed=seed)
    #
    # for graph_ in range(args.graph_nbr):
    #    seed = np.random.seed(120+graph_)
    #    graph_dic[graph_] = graph.Graph(graph_type=args.graph_type, cur_n=args.node, p=args.p, m=args.m, seed=seed)
    TARGET_UPDATE_FREQUENCY = 10
    env = Environment()
    n_state = 128
    n_action = cur_n

    """Generate agents"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agentVecadd = []
    agentVecdlt = []
    for i in range(cur_n):
        addAgent: Agent = Agent(n_input=n_state,
                  n_output=n_action,device=device,
                  mode='train')
        # addAgent.online_net.load_state_dict(torch.load("fig/pre/addpara/addpara{}.pt".format(i)))
        dltAgent: Agent = Agent(n_input=n_state,
                                n_output=n_action, device=device,
                                mode='train')
        # dltAgent.online_net.load_state_dict(torch.load("fig/pre/dltpara/dltpara{}.pt".format(i)))
        # oneNewAgent.setAgentID(agentID)
        agentVecadd.append(addAgent)
        agentVecdlt.append(dltAgent)
        # agentID += 1
        # oneNewAgent.setAgentTimeStep(agentTime)
        # # oneNewAgent.setAgentActionUtilityTable(agentInitialActionUtilityTable)
        #
        # oneNewAgent.setAlpha(parameter1)
        # oneNewAgent.setBeta(parameter2)
    # if  os.path.exists("addpara") :
    #  for agent_i in range(cur_n):
    #     oneAgentadd = agentVecadd[agent_i]
    #     oneAgentadd.online_net.load_state_dict(torch.load("addpara/addpara{}.pt".format(agent_i)))
    #     # oneAgentadd.target_net.load_state_dict(oneAgentadd.online_net.state_dict())
    # #  print(1)
    # # if os.path.exists("dltpara"):
    # #  for agent_i in range(cur_n):
    #     oneAgentdlt = agentVecdlt[agent_i]
    #     oneAgentdlt.online_net.load_state_dict(torch.load('dltpara/dltpara{}.pt'.format(agent_i)))

    """Main Training Loop"""
    n_episode = args.episode
    n_time_step = args.step
    modularity = []
    clustering = []
    shortestpath =[]
    degree = []
    pbar = tqdm(total=n_episode)
    for episode_i in range(args.episode):
        G=env.reset2()
        static_graphs = []
        static_graphs.append(G)
        pbar.set_description('Process')
        n=G.number_of_nodes()
        print("Episode: {}".format(episode_i))
        # episode_reward_add = 0
        # episode_reward_dlt = 0
        # epsilon = max(EPSILON_END, epsilon * epsilon_decay)
        epsilon = 0
        r_add = np.zeros(G.number_of_nodes())
        r_dlt = np.zeros(G.number_of_nodes())
        # subset=random.sample(range(0,100),10)
        # print(subset)
        time_start = time.time()

        for step_i in range(n_time_step):
            dlt_reward = []
            next_G = copy.deepcopy(G)
            observation = env.observe_2(G)
            g = ig.Graph.from_networkx(G)
            s_add = []
            s_dlt = []
            a_add_list = []
            a_dlt_list = []
            node_all = []

            for agent_i in range(cur_n):
               # agent_i = int(agent_ii)
               oneAgentadd = agentVecadd[agent_i]

               nodes=g.neighborhood(agent_i,order=args.ego)
               node_all.append(nodes)

               sum2 = np.zeros(32)
               for i in nodes:
                   sum2 += observation.get(i)
               sum_tensor = torch.as_tensor(sum2, dtype=torch.float32)
               sum_tensor_2=sum_tensor.expand([n,32]).to(device)
               nodes.remove(agent_i)


               x_list = list(observation.values())
               obs_tensor = torch.as_tensor(np.array(x_list), device=device,dtype=torch.float32)
               obs = torch.cat((obs_tensor, sum_tensor_2), dim=1)
               s_add.append(obs.cpu().numpy().tolist())
               a_add_list.append(add_sort(oneAgentadd.online_net.act(nodes,obs)))

               oneAgentdlt = agentVecdlt[agent_i]
               nodes2 = g.neighborhood(agent_i, order=1)
               sum1 = np.zeros(32)
               for i in nodes2:
                    sum1 += observation.get(i)
               sum_tensor = torch.as_tensor(sum1, dtype=torch.float32)
               sum_tensor_2 = sum_tensor.expand([n, 32]).to(device)
               nodes2.remove(agent_i)
               x_list = list(observation.values())
               obs_tensor = torch.as_tensor(np.array(x_list), device=device, dtype=torch.float32)
               obs = torch.cat((obs_tensor, sum_tensor_2), dim=1)
               s_dlt.append(obs.cpu().numpy().tolist())
               a_dlt_list.append(dlt_sort(oneAgentdlt.online_net.act(nodes2, obs)))

            a_add = []
            a_dlt = []

            for agent_i in range(cur_n):
                action_list_add = []
                nodes = g.neighborhood(agent_i, order=args.ego )
                nodes.remove(agent_i)
                random_sample = random.random()
                if a_add_list and nodes:
                 if random_sample <= epsilon:
                     for _ in range(2):
                      a_random=random.choice(nodes)
                      action_list_add.append(a_random)
                      next_G.add_edge(agent_i, a_random)

                 else:
                    for i in a_add_list[agent_i]:
                      if a_add_list[i]:
                            if agent_i in a_add_list[i] and env.get_reward(G, i) > args.add_thresh:
                                action_list_add.append(i)
                                next_G.add_edge(agent_i, i)

                a_add.append(action_list_add)

                action_list_dlt = []
                nodes2 = g.neighborhood(agent_i, order=1)
                if a_dlt_list :
                 if random_sample <= epsilon:
                    a_random = random.choice(nodes2)

                    if next_G.has_edge(agent_i, a_random) :

                        action_list_dlt.append(a_random)
                        next_G.remove_edge(agent_i, a_random)
                 else:
                        for i in a_dlt_list[agent_i]:
                          # print(agent_i, i, a_dlt_list[i])
                          if a_dlt_list[i]:
                            # if agent_i in a_dlt_list[i] and env.get_reward_4(G, i) > args.dlt_thresh:
                            if agent_i in a_dlt_list[i] :
                               dlt_reward.append(env.get_reward_4(G, i))
                               if next_G.has_edge(agent_i, i):
                                    action_list_dlt.append(i)
                                    next_G.remove_edge(agent_i, i)

                a_dlt.append(action_list_dlt)


            for agent_i in range(cur_n):

                mix_cur = env.get_reward(G, agent_i)
                mix_next = env.get_reward(next_G, agent_i)
                r_add[agent_i]=mix_next-mix_cur
                mix_dlt_cur = env.get_reward_4(G, agent_i)
                mix_dlt_next = env.get_reward_4(next_G, agent_i)
                r_dlt[agent_i] = mix_dlt_next - mix_dlt_cur

            observation_next = env.observe_2(next_G)


            for agent_i in range(cur_n):
                oneAgentadd = agentVecadd[agent_i]
                g_next = ig.Graph.from_networkx(next_G)
                nodes_next=g_next.neighborhood(agent_i,order=args.ego)

                sum_next = np.zeros(32)
                for i in nodes_next:
                    sum_next += observation_next.get(i)
                sum_tensor_next = torch.as_tensor(sum_next, device=device,dtype=torch.float32)
                sum_tensor_2_next = sum_tensor_next.expand([n, 32])
                nodes_next.remove(agent_i)


                x_list_next = list(observation_next.values())
                obs_tensor_next = torch.as_tensor(np.array(x_list_next),device=device, dtype=torch.float32)
                obs_next = torch.cat((obs_tensor_next, sum_tensor_2_next), dim=1)
                s_ = obs_next.cpu().numpy().tolist()
                if a_add[agent_i]and s_add[agent_i]:
                    for i in a_add[agent_i]:
                       # i_tensor = torch.as_tensor(np.array(i), device=device, dtype=torch.float32)
                       # r_tensor = torch.as_tensor(np.array(r_add[agent_i]), device=device, dtype=torch.float32)
                       # # print(s_add[agent_i], i_tensor, r_tensor, s_)
                       oneAgentadd.memo.add_memo(s_add[agent_i], i, r_add[agent_i],s_)



                oneAgentdlt = agentVecdlt[agent_i]
                nodes2_next = g_next.neighborhood(agent_i, order=1)
                sum_next = np.zeros(32)
                for i in nodes2_next:
                    sum_next += observation_next.get(i)
                sum_tensor_next = torch.as_tensor(sum_next, device=device, dtype=torch.float32)
                sum_tensor_2_next = sum_tensor_next.expand([n, 32])
                nodes2_next.remove(agent_i)
                x_list_next = list(observation_next.values())
                obs_tensor_next = torch.as_tensor(np.array(x_list_next), device=device, dtype=torch.float32)
                obs_next = torch.cat((obs_tensor_next, sum_tensor_2_next), dim=1)
                s_ = obs_next.cpu().numpy().tolist()
                if a_dlt[agent_i] and s_dlt[agent_i]:
                    for i in a_dlt[agent_i]:
                       # print(i)
                       oneAgentdlt.memo.add_memo(s_dlt[agent_i], i, r_dlt[agent_i],s_)
                # oneAgentdlt.memo.to_csv()

            for agent_i in range(cur_n):

                oneAgentadd = agentVecadd[agent_i]
                # print(oneAgentadd.memo.all_s_)
                # if oneAgentadd.memo.size()>0:
                batch_s, batch_a, batch_r, batch_s_ = oneAgentadd.memo.sample()
                # print(batch_s, batch_a, batch_r, batch_s_)
                # Compute Targets
                if batch_s_.numel() != 0:
                 target_q_values = oneAgentadd.target_net(batch_s_).squeeze(2)

                 max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]  # ?

                 targets = batch_r + oneAgentadd.GAMMA * max_target_q_values
                # Compute Q_values
                 q_values = oneAgentadd.online_net(batch_s)

                 a_q_values = torch.gather(input=q_values.squeeze(2), dim=1, index=batch_a) #
                # Compute Loss

                 loss = nn.functional.smooth_l1_loss(a_q_values, targets)
                # Gradient Descent
                 oneAgentadd.optimizer.zero_grad()
                 loss.backward()
                 oneAgentadd.optimizer.step()


                oneAgentdlt = agentVecdlt[agent_i]
                # if oneAgentdlt.memo.size() > 0:
                batch_s, batch_a, batch_r, batch_s_ = oneAgentdlt.memo.sample()
                # Compute Targets
                if batch_s_.numel()!=0:
                 target_q_values = oneAgentdlt.target_net(batch_s_).squeeze(2)
                 max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]  # ?
                 targets = batch_r + oneAgentdlt.GAMMA * max_target_q_values
                 # Compute Q_values
                 q_values = oneAgentdlt.online_net(batch_s)
                 a_q_values = torch.gather(input=q_values.squeeze(2), dim=1, index=batch_a)  #
                # Compute Loss
                #  tensor_a_q_values =  a_q_values.clone().detach()
                #  tensor_targets = targets.clone().detach()

                 loss = nn.functional.smooth_l1_loss(a_q_values, targets)
                 # Gradient Descent
                 oneAgentdlt.optimizer.zero_grad()
                 loss.backward()
                 oneAgentdlt.optimizer.step()

            G=copy.deepcopy(next_G)
            static_graphs.append(G)


            print(f"Step:{step_i}")
            print(nx.degree_histogram(G))
            print(G.size())
            print(f"Modularity:{Evaluation.Modularity(G)}")
            modularity.append(Evaluation.Modularity(G))
            print(f"Clustering:{Evaluation.globalClustering(G)}")
            clustering.append(Evaluation.globalClustering(G))
            # print(f"ShortestPath:{Evaluation.averageDistance(G)}")
            # shortestpath.append(Evaluation.averageDistance(G))
            # print(f"Degree:{nx.degree_histogram(G)}")
            degree.append(nx.degree_histogram(G))
            # print(nx.degree_histogram(G))
        modularity_cur= Evaluation.Modularity(G)
        # print(modularity_cur)
        modularity.append(modularity_cur)
        clustering_cur= Evaluation.globalClustering(G)
        clustering.append(clustering_cur)
        # shortestpath_cur=Evaluation.averageDistance(G)
        # shortestpath.append(shortestpath_cur)
        # degree_cur = nx.degree_histogram(G)
        # degree.append(degree_cur)
        # print(degree_cur)
        # print(G.size())
        # CP_coef_cur=Evaluation.CP_coef_3(G,subset)
        # CP_coef.append(CP_coef_cur)5
        # print(CP_coef_cur)
        time_end = time.time()
        print(f"Time:{time_end - time_start}")
        # Num(static_graphs)
        if episode_i % TARGET_UPDATE_FREQUENCY == 0:
            for agent_i in range(cur_n):
                oneAgentadd = agentVecadd[agent_i]
                oneAgentadd.target_net.load_state_dict(oneAgentadd.online_net.state_dict())
                oneAgentdlt = agentVecdlt[agent_i]
                oneAgentdlt.target_net.load_state_dict(oneAgentdlt.online_net.state_dict())

        # if (episode_i + 1) % 10 == 0:
        #         data1 = np.array(modularity)
        #         np.savetxt('modularity1.txt', data1)
        #         # nx.write_gexf(G, 'modularityt{}.gexf'.format(episode_i))
        #         # print(degree)
        #         data2 = degree
        #         file_path = "degree.txt"
        #         with open(file_path, "w") as file:
        #             for row in data2:
        #                 row_str = ",".join(map(str, row))
        #                 file.write("[" + row_str + "],\n")
        #         data3 = np.array(clustering)
        #         np.savetxt('ClusteringAddDelete.txt', data3)
        #         data4 = np.array(shortestpath)
        #         np.savetxt('ShortestpathAddDelete.txt', data4)
                # nx.write_gexf(G, 'fig/GWM/node300l=5temp{}.gexf'.format(episode_i))
                # data4 = np.array(REWARD_BUFFER_add)crf
                # np.savetxt('fig/Reward/Reward_add.txt', data4)
                # data5 = np.array(REWARD_BUFFER_dlt)
                # np.savetxt('fig/Reward/Reward_dlt.txt', data5)
        #         if not os.path.exists("addpara"):
        #             os.makedirs("addpara")
        #             os.makedirs("dltpara")
        #         for agent_i in range(cur_n):
        #             oneAgentadd = agentVecadd[agent_i]
        #             torch.save(oneAgentadd.online_net.state_dict(), 'addpara/addpara{}.pt'.format(agent_i))
        #             # oneAgentadd.target_net.load_state_dict(oneAgentadd.online_net.state_dict())
        #             oneAgentdlt = agentVecdlt[agent_i]
        #             torch.save(oneAgentdlt.online_net.state_dict(), 'dltpara/dltpara{}.pt'.format(agent_i))
        #
        pbar.update()

def add_sort(obj):
    e = sorted(obj.items(), key=lambda e: e[1], reverse=True)

    e1 = []
    for i in e:
        e1.append(i[0])
    # print(min(len(e1),args.add_list))
    return e1[0:min(len(e1),args.add_list)]

def dlt_sort(obj):
    e = sorted(obj.items(), key=lambda e: e[1], reverse=True)
    e1 = []
    for i in e:
        e1.append(i[0])
        # min(len(e1), args.dlt_list)
    return e1[0:min(len(e1),args.dlt_list)]

def is_connected(graph):
    return nx.is_connected(graph)

def Num (static_graphs):
    degree_changes = []
    for i in range(len(static_graphs) - 1):
        prev_graph = static_graphs[i]
        next_graph = static_graphs[i + 1]
        degree_change_count = 0
        for node in prev_graph.nodes():
            prev_degree = prev_graph.degree(node)
            next_degree = next_graph.degree(node)
            if prev_degree != next_degree:
                degree_change_count += 1
        degree_changes.append(degree_change_count)
    for i in range(len(static_graphs) - 1):
        degree_changes_counts = {}
        current_graph = static_graphs[i]
        next_graph = static_graphs[i + 1]
        for node in current_graph.nodes():
            current_degree = current_graph.degree(node)
            next_degree = next_graph.degree(node)
            degree_diff = next_degree - current_degree
            if degree_diff in degree_changes_counts:
                degree_changes_counts[degree_diff] += 1
            else:
                degree_changes_counts[degree_diff] = 1
        sorted_degree_changes_counts = dict(sorted(degree_changes_counts.items(), key=lambda item: item[0]))
        # print(degree_changes_counts)
        print(f"Degree Changes Count Between Graph {i + 1} and Graph {i + 2}:")
        for change, count in sorted_degree_changes_counts.items():
            print(f"change: {change} number：{count}")


def selectincreasingnode(static_graph):
    import networkx as nx
    graphs = static_graph
    pageranks = []
    for graph in graphs:
        pagerank = nx.pagerank(graph)
        pageranks.append(pagerank)
    max_increasing_node = None
    max_increase = 0
    for node in pageranks[0].keys():
            increase_amount = pageranks[-1][node] - pageranks[0][node]
            if increase_amount > max_increase:
                max_increase = increase_amount
                max_increasing_node = node

    print(f"max_increasing_node: {max_increasing_node}")
    print(f"max_increase: {max_increase}")



def generate_ring_lattice(n, k):
        G = nx.generators.random_graphs.watts_strogatz_graph(n, k, 0)
        return G

if __name__ == "__main__":
    args = arguements.get_args()
    # profiler = Profile()
    # profiler.runcall(main)
    # profiler.print_stats(sort='tottime')
    main_directory = r"D:\MyCode\NEMARL model\NEMARL model\fig/RoleTrans/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)
    os.chdir(main_directory)
    G = nx.read_gexf(main_directory+'Tar31.gexf')
    train(G)