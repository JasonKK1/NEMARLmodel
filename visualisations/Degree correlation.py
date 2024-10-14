import numpy as np
from matplotlib import pyplot as plt
from Common import arguements
# 设置西文字体为新罗马字体
from matplotlib import rcParams
import networkx as nx
from collections import (defaultdict)
from matplotlib.ticker import MultipleLocator
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    # "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)
plt.figure(figsize=(25, 20), dpi=100)
G = nx.read_gexf("")
# 计算节点的度
n = G.number_of_nodes()

import networkx as nx

# 构造或加载一个networkx图
# G = nx.gnm_random_graph(100, 200)
# 例如，生成一个含有100个节点和200条边的随机图
# 初始化度相关矩阵
degree_correlation_matrix = defaultdict(int)
# 获取图中所有节点的度数
node_degrees = dict(G.degree())
# 遍历图中的每条边
# print(node_degrees)
for u, v in G.edges():
    deg_u = node_degrees[u]
    deg_v = node_degrees[v]
    # 更新度相关矩阵
    degree_correlation_matrix[(deg_u, deg_v)] += 1
# print(G.edges())
    if deg_u != deg_v:
# 对于无向图，(i, j)和(j, i)是相同的
      degree_correlation_matrix[(deg_v, deg_u)] += 1

# 归一化度相关矩阵
total_edges = G.number_of_edges()
# print(degree_correlation_matrix)
for key in degree_correlation_matrix:
    degree_correlation_matrix[key] /= (2*total_edges)
    # 输出结果（部分）
for key in sorted(degree_correlation_matrix):
    print(f"Degree {key}: {degree_correlation_matrix[key]}")

max_degree = max(max(k) for k in degree_correlation_matrix.keys())
matrix_size = max_degree + 1
# 加1因为度数从0开始 # 构建矩阵
degree_matrix = np.zeros((matrix_size, matrix_size))
# 填充矩阵
for (i, j), value in degree_correlation_matrix.items():
    degree_matrix[i, j] = value
    # 生成热力图
plt.imshow(degree_matrix, cmap='hot', interpolation='nearest',origin='lower')
cb = plt.colorbar()

# 使用colorbar对象的ax属性设置色条刻度标签的字体大小
cb.ax.tick_params(labelsize=50)
# 显示颜色条
# plt.title('Degree Correlation Matrix Heatmap')
ax = plt.gca()

# 设置X轴和Y轴的刻度间隔为5
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(5))

plt.xlabel('Degree K1',size = 90)
plt.ylabel('Degree K2',size = 90)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.show()