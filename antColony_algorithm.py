# using ant colony algorithm to solve TSP problem
import numpy as np
import matplotlib.pyplot as plt

class AntColonyTSP_Solver(object):
    def __init__(self, data, num_ant=31, rho= 0.1, Q=1):
        self.data = data
        self.num = len(data)        # 城市个数
        self.num_ant = num_ant      # 蚁群个数

        self.matrix_distance = self.matrix_dis() 
        self.eta = 1 / self.matrix_distance    # 启发函数矩阵
        
        self.Tau_info = np.ones((self.num, self.num))    # 初始信息素浓度矩阵
        self.path = np.array([0]*self.num*self.num_ant).reshape(self.num_ant,self.num)
        self.rho = rho  # 信息素的挥发程度
        self.Q = Q      # 常系数

    def matrix_dis(self):
        """计算城市间的距离"""
        res = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in range(i+1, self.num):
                res[i, j] = np.linalg.norm(self.data[i, :] - self.data[j, :])
                res[j, i] = res[i, j]
        return res + 0.0001

    def comp_fit(self, a_path):
        """计算单个染色体的路径距离值, 用于更新self.fittness"""
        res = 0
        for i in range(self.num-1):
            res += self.matrix_distance[a_path[i], a_path[i+1]]
        res += self.matrix_distance[a_path[-1], a_path[0]]
        return res

    def display_path(self, a_path):
        """可视化路径"""
        res = str(a_path[0] + 1) + '-->'
        for i in range(1, self.num):
            res += str(a_path[i] + 1) + '-->'
        res += str(a_path[0] + 1) + '\n'
        print(res)

    def initial_path(self):
        """清空蚂蚁路径距离"""
        self.path = np.array([0] * self.num * self.num_ant).reshape(self.num_ant, self.num)

    def rand_chrom(self):
        """初始化每个蚂蚁的初始位置"""
        for i in range(self.num_ant):
            self.path[i, 0] = np.random.randint(self.num)

    def update_info(self, fit):
        """
        更新信息素浓度
        fit:蚂蚁一条路径的长度
        """
        delta = sum([self.Q / fit[i] for i in range(self.num_ant)])
        Delta_Tau = np.zeros((self.num, self.num))
        for i in range(self.num_ant):
            for j in range(self.num-1):
                Delta_Tau[self.path[i, j], self.path[i, j+1]] += self.Q / fit[i]
            Delta_Tau[self.path[i, 0], self.path[i, -1]] += self.Q / fit[i]
        self.Tau_info = (1 - self.rho) * self.Tau_info + Delta_Tau

def main(data):
    """主函数, 模拟蚁群算法的TSP问题求解"""
    num_ant = 10
    alpha = 1   # 信息素重要程度
    beta = 8    # 启发函数
    rho = 0.1   # 信息素挥发
    Q = 1       # 常系数

    iter_0 = 0
    Max_iter = 201
    n = len(data)   # 城市个数

    # 蚁群算法
    Path_short = AntColonyTSP_Solver(data, num_ant=num_ant, rho=rho, Q=Q)

    # 城市集合
    city_index = np.array(range(n))

    best_path = []
    best_fit = []       # 存储每一步的最优路径

    # 绘制初始化路径图
    fig, ax = plt.subplots()
    x = data[:, 0]
    y = data[:, 1]
    ax.scatter(x, y, linewidths=0.5)
    for i, txt in enumerate(range(1, len(data)+1)):
        ax.annotate(txt, (x[i], y[i]))
    res0 = Path_short.path[0]
    x0 = x[res0]
    y0 = y[res0]

    for i in range(len(data)-1):
        plt.quiver(x0[i], y0[i], x0[i+1]-x0[i], y0[i+1]-y0[i], color='c', width=0.005, angles='xy', scale=1, scale_units='xy')
    plt.quiver(x0[-1], y0[-1], x0[0]-x0[-1], y0[0]-y0[-1], color='c', width=0.005, angles='xy', scale=1, scale_units='xy')
    plt.show()

    while iter_0 <= Max_iter:
        Path_short.initial_path()   # 清空蚂蚁路径
        Path_short.rand_chrom()     # 随机安排蚂蚁的初始位置

        # 更新每一个蚂蚁的行走路径
        for i in range(num_ant):
            for j in range(1, n):
                pathed = Path_short.path[i, :j]     # 蚂蚁i已走过的路径
                # 蚂蚁i下个可访问的城市
                allow_city = city_index[~np.isin(city_index, pathed)]

                # 轮盘赌法根据概率选择下一城市
                prob = np.zeros(len(allow_city))
                for k in range(len(allow_city)):
                    prob[k] = (Path_short.Tau_info[pathed[-1], allow_city[k]])**(alpha)\
                        *((Path_short.eta[pathed[-1], allow_city[k]])**beta)
                prob = prob / sum(prob)
                cumsum = np.cumsum(prob)
                pick = np.random.rand()
                for r in range(len(prob)):
                    if cumsum[r] >= pick:
                        Path_short.path[i, j] = allow_city[r]
                        break
        # 计算每个蚂蚁经过的路径距离
        fit = np.zeros(num_ant)
        for i in range(num_ant):
            fit[i] = Path_short.comp_fit(Path_short.path[i, :])
        
        # 存储当前迭代所有蚂蚁的最优路径
        min_index = np.argmin(fit)
        if iter_0 == 0:
            best_path.append(Path_short.path[min_index, :])
            best_fit.append(fit[min_index])
        else:
            if fit[min_index] < best_fit[-1]:
                best_path.append(Path_short.path[min_index, :])
                best_fit.append(fit[min_index])
            else:
                best_path.append(best_path[-1])
                best_fit.append(best_fit[-1])
        
        # 更新信息素
        Path_short.update_info(fit)

        if iter_0 % 20 == 0:
            print("第"+str(iter_0)+"步后的最短距离: " + str(best_fit[-1]))
            print("第"+str(iter_0)+"步后的最优路径: ")
            Path_short.display_path(best_path[-1])  # 显示每一步的最优路径
        iter_0 += 1
    
    Path_short.best_path = best_path
    Path_short.best_fit = best_fit

    return Path_short

if __name__ == '__main__':
    np.random.seed(10)
    data = np.random.rand(50,2)*20

    test1 = main(data)