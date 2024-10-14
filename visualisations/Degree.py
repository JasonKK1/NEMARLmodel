import numpy as np
from matplotlib import pyplot as plt
from Common import arguements
# 设置西文字体为新罗马字体
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    # "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)
plt.figure(figsize=(25, 20), dpi=100)
degree_hist1 = [0, 5, 0, 399, 509, 399, 186, 109, 55, 24, 23, 8, 3, 7, 0, 1, 1]
degree_hist2 = [0, 2, 18, 311, 606, 458, 227, 43, 34,20,7,2,1]
# degree_hist3 = [0, 40, 120, 58, 37, 18, 10, 8, 1, 1, 1, 1, 1, 1]
degree_hist1 = np.array(degree_hist1, dtype=float)
degree_prob1 = degree_hist1 / sum(degree_hist1)
degree_hist2 = np.array(degree_hist2, dtype=float)
degree_prob2 = degree_hist2 / sum(degree_hist2)
# degree_hist3 = np.array(degree_hist3, dtype=float)
# degree_prob3 = degree_hist3 / sum(degree_hist3)
# print(sum(degree_hist1))

plt.xlim((1, len(degree_hist1)))
# plt.xscale('log')
p1, = plt.plot(np.arange(degree_prob1.shape[0]), degree_prob1, '#8B0000',linewidth=20.0)
p2, = plt.plot(np.arange(degree_prob2.shape[0]), degree_prob2, '#4b74b2',linewidth=20.0)
# p3, = plt.plot(np.arange(degree_prob3.shape[0]), degree_prob3, '#f7ac53',linewidth=15.0)
# plt.legend([p1,p2,p3], ["Network size = 100","Network size = 300","Network size = 500"], loc='upper right',fontsize=60)
plt.legend([p1,p2], ["Real Data","Model Simulation"], loc='upper right',fontsize=60)
# with open('Degree.txt', 'w') as f:
# plt.xlabel('Node degree k',size='90')
plt.xlabel('K',size='90')
plt.ylabel('p(k)',size='90')
# plt.ylabel('Degree Probability p(k)',size='90')
plt.title('Degree Distribution',size='90')
plt.tick_params(labelsize=40)
# plt.loglog(x, degree_hist1, '.')
plt.grid(linestyle='-.',linewidth=3.0)
plt.show()