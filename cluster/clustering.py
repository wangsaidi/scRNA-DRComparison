import community as community_louvain
import numpy as np
import networkx as nx
import random


'''
聚类及指标计算
'''

# Compute cluster centroids, which is the mean of all points in one cluster.
def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])

class louvain:
    def __init__(self, level):
        self.level = level
        return

    def updateLabels(self, level):
        # Louvain algorithm labels community at different level (with dendrogram).
        # Here we want the community labels at a given level.
        level = int((len(self.dendrogram) - 1) * level)
        partition = community_louvain.partition_at_level(self.dendrogram, level)
        # Convert dictionary to numpy array
        self.labels = np.array(list(partition.values()))
        return

    def update(self, inputs, adj_mat=None):
        """Return the partition of the nodes at the given level.

        A dendrogram is a tree and each level is a partition of the graph nodes.
        Level 0 is the first partition, which contains the smallest communities,
        and the best is len(dendrogram) - 1.
        Higher the level is, bigger the communities are.
        """
        self.graph = nx.from_numpy_matrix(adj_mat)
        self.dendrogram = community_louvain.generate_dendrogram(self.graph)
        self.updateLabels(self.level)
        self.centroids = computeCentroids(inputs, self.labels)
        return



# 调用scikit-learn的其他两种聚类
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

def clustering(h, n_cluster, k=15, f="louvain"):
    '''
    :param h: 输入数据
    :param n_cluster: 真实簇的个数
    :param f: 使用什么聚类方法
    :return: 预估标签
    这里的random_state=0,设置了固定的种子，每次聚类结果不变；
    可以将 random_state 设置为当前时间戳，运行多次取平均来计算指标；
    louvain是数据驱动的聚类，算法自动确定簇的数量
    '''
    from preprocessing import get_adj
    from clustering import louvain
    adj, adj_n = get_adj(h, k=k, pca=False)  # 邻接矩阵和归一化的邻接矩阵
    if f == "louvain":
        # cl_model = louvain(level=0.5)
        # cl_model.update(h, adj_mat=adj)
        # labels = cl_model.labels
        best_labels = None
        best_modularity = -1
        for i in range(50):  # 运行50次
            np.random.seed(i)
            random.seed(i)
            cl_model = louvain(level=0.5)
            cl_model.update(h, adj_mat=adj)
            current_labels = cl_model.labels
            partition = dict(enumerate(current_labels))
            modularity = community_louvain.modularity(partition, nx.from_numpy_matrix(adj))
            if modularity > best_modularity:
                best_modularity = modularity
                best_labels = current_labels
        labels = best_labels
    elif f == "spectral":
        labels = SpectralClustering(n_clusters=n_cluster, affinity="precomputed", assign_labels="discretize",
                                    random_state=0,n_init=50).fit_predict(adj)   # 这里的random_state=0,设置了固定的种子，每次聚类结果不变，保证结果的可复现性
    elif f == "kmeans":
        labels = KMeans(n_clusters=n_cluster, random_state=0,n_init=50).fit(h).labels_    # 这里的random_state=0,设置了固定的种子，每次聚类结果不变
    return labels

'''
spectral聚类先做谱嵌入，然后做kmeans聚类，所以对于kmeans与spectral，都有参数n_init
当n_init=20时，kmeans会用20组不同的初始质心各跑一次完整的聚类流程，然后选取inertia（样本到簇中心的平方距离总和）最小的那一组作为最终结果；
当设置n_init > 1，也就是运行多次时（比如20），仍然可以设置random_state为固定值来保证结果的可复现性，运行多次会随机选择20个不同的初始质心，random_state保证的是每次运行这20次选取的质心序列是一致的，结果仍然可复现；

总结：kmeans与spectral可以通过n_init参数重复多次取最好的结果，通过random_state保证结果的可复现性
对于louvain，没有这两个参数，只能手动运行多次取最好的（循环、模块度最高的），若想结果可复现，只能设置全局种子(np.random.seed(0))

使用：对于目前的程序，对于kmeans与spectral，只需要加入参数n_init=50,就会自动运行多次，返回最好的结果
对于louvain，将上述分支部分改为下面的代码即可运行多次、选取模块度最高的结果；

if f == "louvain":
    best_labels = None
    best_modularity = -1
    for i in range(50):  # 运行50次
        np.random.seed(i)
        random.seed(i)
        cl_model = louvain(level=0.5)
        cl_model.update(h, adj_mat=adj)
        current_labels = cl_model.labels
        partition = dict(enumerate(current_labels))
        modularity = community_louvain.modularity(partition, nx.from_numpy_matrix(adj))
        if modularity > best_modularity:
            best_modularity = modularity
            best_labels = current_labels
    labels = best_labels


调用时不变，仍然是 labels = clustering(embedding.values, n_clusters, f=method)
'''

# 指标计算，调用scikit-learn库的实现
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, silhouette_score
def measure(true, pred, x):
    '''
    :param true: 真实标签
    :param pred: 预估标签
    :return: 指标
    NMI:[0, 1]    衡量聚类结果与真实标签之间的相似性
    RAND:[-1, 1]    衡量聚类结果与真实标签之间的相似性，考虑了随机聚类的可能性
    HOMO:[0,1]    每个聚类是否只包含一个类别的样本；评估聚类结果是否“纯净”
    COMP:[0,1]    每个类别的样本是否被分配到同一个聚类中；评估聚类结果是否“完整”
    都是越接近 1 结果越好
    '''
    NMI = round(normalized_mutual_info_score(true, pred), 2)
    RAND = round(adjusted_rand_score(true, pred), 2)
    HOMO = round(homogeneity_score(true, pred), 2)
    COMP = round(completeness_score(true, pred), 2)
    SIL = round(silhouette_score(x, pred), 2)
    return [NMI, RAND, HOMO, COMP, SIL]



def get_centers_louvain(Y, adj):
    from clustering import louvain
    cl_model = louvain(level=0.5)
    cl_model.update(Y, adj_mat=model.adj)
    labels = cl_model.labels
    centers = computeCentroids(Y, labels)
    return centers, labels

def get_centers_spectral(Y, adj):
    from sklearn.cluster import SpectralClustering
    l = SpectralClustering(n_clusters=10,affinity="precomputed", assign_labels="discretize",random_state=0).fit_predict(adj)
    centers = computeCentroids(Y, l)
    return centers, l