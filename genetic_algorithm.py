# using genetic algorithm to solve TSP problem
from math import floor
import numpy as np
import matplotlib.pyplot as plt

class GenaTSP_Solver(object):
    def __init__(self, data, maxgen=501, size_pop=200, cross_prob=0.9, pmuta_prob=0.01, select_prob=0.8):
        self.maxgen = maxgen            # 最大迭代次数
        self.size_pop = size_pop        # 群体规模
        self.cross_prob = cross_prob    # 交叉概率
        self.pmuta_prob = pmuta_prob    # 变异概率
        self.select_prob = select_prob  # 选择概率

        self.data = data                # 城市数据
        self.num = len(data)            # 城市数量

        self.matrix_distance = self.matrix_dis()    # 距离矩阵, 第[i,j]个元素表示城市i到j距离

        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)     # 通过选择概率确定子代的选择个数

        self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop, self.num)  # 父代群体初始化
        self.sub_sel = np.array([0] * self.select_num * self.num).reshape(self.select_num, self.num)    # 子代群体初始化

        self.fittness = np.zeros(self.size_pop)  # 存储群体中每条染色体的路径总长度, 对应单个染色体的适应度为其倒数

        self.best_fit = []      # 存储每一步群体的最优路径
        self.best_path = []     # 存储每一步群体的最优距离

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
        for i in range(self.size_pop):
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

    def select_sub(self):
        """子代选取, 根据选中概率与对应的适应度函数, 采用随机遍历选择方法"""
        fit = 1. / (self.fittness)      # 适应度函数
        cumsum_fit = np.cumsum(fit)
        pick = cumsum_fit[-1] / self.select_num * (np.random.rand() + np.array(range(self.select_num)))
        i, j = 0, 0
        index = []
        while i < self.size_pop and j < self.select_num:
            if cumsum_fit[i] > pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index, :]
    
    def cross_sub(self):
        """交叉, 依概率对子代进行交叉操作"""
        if self.select_num % 2 == 0:
            num = range(0, self.select_num, 2)
        else:
            num = range(0, self.select_num-1, 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i, :], self.sub_sel[i+1, :] = self.intercross(self.sub_sel[i, :], self.sub_sel[i+1, :])

    def intercross(self, ind_a, ind_b):
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r2 == r1:
            r2 = np.random.randint(self.num)
        left, right = min(r1, r2), max(r1, r2)
        ind_a1 = ind_a.copy()
        ind_b1 = ind_b.copy()

        for i in range(left, right+1):
            ind_a2 = ind_a.copy()
            ind_b2 = ind_b.copy()
            ind_a[i] = ind_b1[i]
            ind_b[i] = ind_a1[i]
            x = np.argwhere(ind_a == ind_a[i])
            y = np.argwhere(ind_b == ind_b[i])
            if len(x) == 2:
                ind_a[x[x!=i]] = ind_a2[i]
            if len(y) == 2:
                ind_b[y[y!=i]] = ind_b2[i]
        return ind_a, ind_b

    def mutation_sub(self):
        """变异"""
        for i in range(self.select_num):
            if np.random.rand() <= self.cross_prob:
                r1 = np.random.randint(self.num)
                r2 = np.random.randint(self.num)
                while r2 == r1:
                    r2 = np.random.randint(self.num)
                self.sub_sel[i, [r1, r2]] = self.sub_sel[i, [r2, r1]]
    
    def reverse_sub(self):
        """进化逆转"""
        for i in range(self.select_num):
            r1 = np.random.randint(self.num)
            r2 = np.random.randint(self.num)
            while r2 == r1:
                r2 = np.random.randint(self.num)
            left, right = min(r1, r2), max(r1, r2)
            sel = self.sub_sel[i, :].copy()

            sel[left:right+1] = self.sub_sel[i, left:right+1][::-1]
            if self.comp_fit(sel) < self.comp_fit(self.sub_sel[i, :]):
                self.sub_sel[i, :] = sel

    def reins(self):
        """子代插入父代, 得到相同规模的新群体"""
        index = np.argsort(self.fittness)[::-1]
        self.chrom[index[:self.select_num], :] = self.sub_sel

def main(data):
    """主函数, 模拟遗传算法的TSP问题求解"""
    Path_short = GenaTSP_Solver(data)       # 根据位置坐标, 生成一个遗传算法类
    Path_short.rand_chrom()                 # 初始化父类

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

    # 进行遗传迭代计算
    for i in range(Path_short.maxgen):
        Path_short.select_sub()     # 选择子代
        Path_short.cross_sub()      # 交叉
        Path_short.mutation_sub()   # 变异
        Path_short.reverse_sub()    # 进化逆转
        Path_short.reins()          # 子代插入

        # 重新计算新群体的距离
        for j in range(Path_short.size_pop):
            Path_short.fittness[j] = Path_short.comp_fit(Path_short.chrom[j, :])
        
        # 显示当前群体的最优距离
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


            print("第"+str(i+1)+"步后的最短距离: " + str(Path_short.fittness[index]))
            print("第"+str(i+1)+"步后的最优路径: ")
            Path_short.display_path(Path_short.chrom[index, :])     # 显示每一步的最优路径
        
        # 存储每一步的最优路径及距离
        Path_short.best_fit.append(Path_short.fittness[index])
        Path_short.best_path.append(Path_short.chrom[index, :])
    return Path_short

if __name__ == '__main__':
    np.random.seed(1210)
    data = np.random.rand(50, 2)*20

    test1 = main(data)

    fit = test1.best_fit

    np.save('./res/genetic.npy', fit) 

    plt.plot(range(len(fit)), fit)
    plt.title("Genetic Algorithm")
    plt.xlabel("number of iterations")
    plt.ylabel("distance")
    plt.show()