# using particle group algorithm to solve TSP problem
import numpy as np
import matplotlib.pyplot as plt

class PgTSP_Solver(object):
    def __init__(self, data, num_pop=200):
        self.num_pop = num_pop      # 群体个数
        self.data = data            # 城市坐标
        self.num = len(data)        # 城市数量
    
        self.chrom = np.array([0] * self.num_pop * self.num).reshape(self.num_pop, self.num)  # 群体初始化
        # self.fittness = [0] * self.num_pop  # 路径初始化
        self.fittness = np.zeros(self.num_pop)
        self.matrix_distance = self.matrix_dis()    # 距离矩阵

    def matrix_dis(self):
        """计算城市间的距离"""
        res = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in range(i+1, self.num):
                res[i, j] = np.linalg.norm(self.data[i, :] - self.data[j, :])
                res[j, i] = res[i, j]
        return res

    def rand_chrom(self):
        """随机初始化群体"""
        rand_ch = np.array(range(self.num))
        for i in range(self.num_pop):
            np.random.shuffle(rand_ch)
            self.chrom[i, :] = rand_ch
            self.fittness[i] = self.comp_fit(rand_ch)
    
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

    def corss(self, path, best_path):
        """两条路径的交叉"""
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r2 == r1:
            r2 = np.random.randint(self.num)

        left, right = min(r1, r2), max(r1, r2)
        cross = best_path[left:right+1]
        for i in range(right-left+1):
            for k in range(self.num):
                if path[k] == cross[i]:
                    path[k:self.num-1] = path[k+1:self.num]
                    path[-1] = 0
        path[self.num-right+left-1:self.num] = cross
        return path

    def mutation(self, path):
        """变异"""
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r2 == r1:
            r2 = np.random.randint(self.num)
        path[r1], path[r2] = path[r2], path[r1]
        return path

def main(data, max_n=501, num_pop=200):
    """主函数, 模拟粒子群算法的TSP问题求解"""
    Path_short = PgTSP_Solver(data, num_pop=num_pop)         # 根据位置坐标, 生成一个粒子群算法类
    Path_short.rand_chrom()                                  # 初始化种群

    # 绘制初始化路径图
    fig, ax = plt.subplots()
    x = data[:, 0]
    y = data[:, 1]
    ax.scatter(x, y, linewidths=0.5)
    for i, txt in enumerate(range(1, len(data)+1)):
        ax.annotate(txt, (x[i], y[i]))
    res0 = Path_short.chrom[0]
    x0 = x[res0]
    y0 = y[res0]

    for i in range(len(data)-1):
        plt.quiver(x0[i], y0[i], x0[i+1]-x0[i], y0[i+1]-y0[i], color='c', width=0.005, angles='xy', scale=1, scale_units='xy')
    plt.quiver(x0[-1], y0[-1], x0[0]-x0[-1], y0[0]-y0[-1], color='c', width=0.005, angles='xy', scale=1, scale_units='xy')
    plt.title("Roadmap")
    plt.show()

    print("初始距离: " + str(Path_short.fittness[0]))

    # 存储个体最优的路径和距离
    best_P_chrom = Path_short.chrom.copy()
    best_P_fit = Path_short.fittness.copy()
    min_index = np.argmin(Path_short.fittness)

    # 存储当前种群最优的路径和距离
    best_G_chrom = Path_short.chrom[min_index, :]
    best_G_fit = Path_short.fittness[min_index]

    # 存储每一步迭代后的最优路径与距离
    best_chrom = [best_G_chrom]
    best_fit = [best_G_fit]

    # 交叉变异
    x_new = Path_short.chrom.copy()

    for i in range(max_n):
        # 更新当前个体极值
        for j in range(num_pop):
            if Path_short.fittness[j] < best_P_fit[j]:
                best_P_fit[j] = Path_short.fittness[j]
                best_P_chrom[j, :] = Path_short.chrom[j, :]

        # 更新当前种群的群体极值
        min_index = np.argmin(Path_short.fittness)
        best_G_chrom = Path_short.chrom[min_index, :]
        best_G_fit = Path_short.fittness[min_index]

        # 更新每一步迭代的全局最优路径和最优解
        if best_G_fit < best_fit[-1]:
            best_fit.append(best_G_fit)
            best_chrom.append(best_G_chrom)
        else:
            best_fit.append(best_fit[-1])
            best_chrom.append(best_chrom[-1])
        
        # 将每个个体与个体极值和当前的群体极值交叉
        for j in range(num_pop):
            # 与个体极值交叉
            x_new[j, :] = Path_short.corss(x_new[j, :], best_P_chrom[j, :])
            fit = Path_short.comp_fit(x_new[j, :])

            # 判断是否保留
            if fit < Path_short.fittness[j]:
                Path_short.chrom[j, :] = x_new[j, :]
                Path_short.fittness[j] = fit

            # 与当前极值交叉
            x_new[j, :] = Path_short.corss(x_new[j, :], best_G_chrom)
            fit = Path_short.comp_fit(x_new[j, :])
            if fit < Path_short.fittness[j]:
                Path_short.chrom[j, :] = x_new[j, :]
                Path_short.fittness[j] = fit

            # 变异
            x_new[j, :] = Path_short.mutation(x_new[j, :])
            fit = Path_short.comp_fit(x_new[j, :])
            if fit <= Path_short.fittness[j]:
                Path_short.chrom[j] = x_new[j, :]
                Path_short.fittness[j] = fit

        index = Path_short.fittness.argmin()
        if (i+1) % 50 == 0:

            # 绘制路径图
            fig, ax = plt.subplots()
            x = data[:, 0]
            y = data[:, 1]
            ax.scatter(x, y, linewidths=0.2)
            for k, txt in enumerate(range(1, len(data)+1)):
                ax.annotate(txt, (x[k], y[k]))
            res = Path_short.chrom[index, :]
            x1 = x[res]
            y1 = y[res]

            for k in range(len(data)-1):
                plt.quiver(x1[k], y1[k], x1[k+1]-x1[k], y1[k+1]-y1[k], color='c', width=0.005, angles='xy', scale=1, scale_units='xy')
            plt.quiver(x1[-1], y1[-1], x1[0]-x1[-1], y1[0]-y1[-1], color='c', width=0.005, angles='xy', scale=1, scale_units='xy')
            plt.title("Roadmap")
            plt.show()

            print("第"+str(i+1)+"步后的最短距离: " + str(Path_short.fittness[min_index]))
            print("第"+str(i+1)+"步后的最优路径: ")
            Path_short.display_path(Path_short.chrom[min_index, :])     # 显示每一步的最优路径
    
    Path_short.best_chrom = best_chrom
    Path_short.best_fit = best_fit
    return Path_short

if __name__ == '__main__':
    np.random.seed(1210)
    data = np.random.rand(50, 2)*20

    test1 = main(data)

    fit = test1.best_fit

    np.save('./res/particle.npy', fit)

    plt.plot(range(len(fit)), fit)
    plt.title("Particle Group Algorithm")
    plt.xlabel("number of iterations")
    plt.ylabel("distance")
    plt.show()