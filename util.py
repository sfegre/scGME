import torch
import umap
import platform
import cluster as cs
import numpy as np
import scanpy as sc
import pandas as pd
from torch.nn.parameter import Parameter
from anndata import AnnData
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn import mixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
import sys
from typing import Literal, Tuple

import anndata as ad
import cv2
import numpy as np
import scanpy as sc
import squidpy as sq
from loguru import logger
from sklearn.impute import SimpleImputer



def detect_device():
    """Automatically detects if you have a cuda enabled GPU"""
    if torch.cuda.is_available():#是否可以使用GPU
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def judge_system():
    """
    Since results of ADClust slightly change under different operating environments,
    we take different activation functions for them.

    We used Ubuntu operating system with 16.04 version
    :return:
    """

    if platform.system() == "Windows":
        return False

    sys="linux"
    version="1.0"
    try:
        import lsb_release_ex as lsb
        info=lsb.get_lsb_information()
        sys=info['ID']
        version=info['RELEASE']
    except Exception as e:
        return False

    return (sys, version) == ('Ubuntu','16.04')


def encode_batchwise(dataloader, model, device):
    """ Utility function for embedding the whole data set in a mini-batch fashion
    """

    embeddings = []
    for batch in dataloader:
        batch_data = batch.to(device)
        embeddings.append(model.encode(batch_data).detach().cpu())

        #append()在数组后加上相应的元素
        #detach()的作用是返回一个Tensor，它和原张量的数据相同，但得到的张量不会具有梯度。
        #记detach()得到的张量为de，后续基于de继续计算，那么反向传播过程中，遇到调用了detach() 方法的张量就会终止 (强调: de没有梯度)，不会继续向后计算梯度。
        #将数据的处理设备从其他设备（如.cuda()拿到cpu上），不会改变变量类型，转换后仍然是Tensor变量。
    return torch.cat(embeddings, dim=0).numpy()
        #将张量（tensor）拼接在一起，dim=0（竖着拼），dim=1（横着拼）





def get_center_labels(X):

    lowest_bic = np.infty  # infty是无穷大
    bic = []
    n_components_range = range(1, 100)  # 从1到100
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components)
        # 建立高斯混合成分为n_components的高斯混合模型
        gmm.fit(X)
        # fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，
        # 如果有验证集的话，也包含了验证集的这些指标变化情况
        bic.append(gmm.bic(X))
        # bic()计算gmm的bic值
        # append()将计算的bic值附加到之前计算的bic值之后
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
        # 若bic比lowest_bic小，则将bic赋值给lowest_bic，相应地将gmm赋值给best_gmm
    bic = np.array(bic)  # 将[所有bic值]转化为array类型
    clustering = best_gmm  # 使用聚类性能最好的高斯混合模型，即bic值最低的模型
    y_pred = clustering.predict(X)
    features = pd.DataFrame(X, index=np.arange(0, X.shape[0]))
    Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
    Mergefeature = pd.concat([features, Group], axis=1)
    # 沿着指定的轴将多个dataframe或者series拼接到一起，axis（默认上下堆叠，等于1时左右堆叠）
    # 将嵌入数据和obs，拼接到一起，便于分组

    init_centroid = np.asarray(Mergefeature.groupby("Group").mean())
    n_clusters = init_centroid.shape[0]
    return init_centroid, y_pred



def get_center_labels2(features):
    '''
    resolution: Value of the resolution parameter, use a value above
          (below) 1.0 if you want to obtain a larger (smaller) number
          of communities.
    '''

    print("\nInitializing cluster centroids using the louvain method.")

    kmeans = KMeans(n_clusters=30, max_iter=300, n_init=10, init="k-means++", random_state=0)
    y_pred = kmeans.fit_predict(features)


    features = pd.DataFrame(features, index=np.arange(0, features.shape[0]))
    Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
    Mergefeature = pd.concat([features, Group], axis=1)
    #沿着指定的轴将多个dataframe或者series拼接到一起，axis（默认上下堆叠，等于1时左右堆叠）
    #将嵌入数据和obs，拼接到一起，便于分组


    init_centroid = np.asarray(Mergefeature.groupby("Group").mean())
    #Mergefeature.groupby("Group")将数据集按Group字段划分
    #将相同组别的按列取均值，不同组别不做处理，输出最终分类后数据
    n_clusters = init_centroid.shape[0]
    #输出louvain的类别，即初始聚类数量

    #print("\n " + str(n_clusters) + " micro-clusters detected. \n")
    return init_centroid, y_pred





def gaussian_kernel(x, y, sigma=1.0):
    distance = np.linalg.norm(x - y)
    similarity = np.exp(-distance ** 2 / (2 * (sigma ** 2)))
    return similarity

def compute_similarity_matrix(X, sigma=1.0):
    n = X.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = gaussian_kernel(X[i], X[j], sigma)
    return S

def myKNN(S, k=5, sigma=1.0):
    N = len(S)
    W = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours

        for j in neighbours_id: # xj is xi's neighbour
            W[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            W[j][i] = W[i][j] # mutually

    return W

def calLaplacianMatrix(W):
    degreeMatrix = np.sum(W, axis=1)  # 按照行对W矩阵进行求和
    L = np.diag(degreeMatrix) - W  # 计算对应的对角矩阵减去w
    # 拉普拉斯矩阵标准化，就是选择Ncut切图
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))  # D^(-1/2)
    L_sym = np.dot(np.dot(sqrtDegreeMatrix, L), sqrtDegreeMatrix)  # D^(-1/2) L D^(-1/2)
    return L_sym

def normalization(matrix): # 归一化
    sum = np.sqrt(np.sum(matrix**2,axis=1,keepdims=True)) # 求数组的正平方根
    nor_matrix = matrix/sum # 求平均
    return nor_matrix


def get_center_labels2(features, k=5, sigma=1.0):
    S=compute_similarity_matrix(features, sigma=1.0)
    W = myKNN(S, k=k, sigma=sigma)  # 计算邻接矩阵
    L_sym = calLaplacianMatrix(W)  # 依据W计算标准化拉普拉斯矩阵
    lam, H = np.linalg.eig(L_sym)  # 特征值分解

    t = np.argsort(lam)  # 将lam中的元素进行排序，返回排序后的下标
    H = np.c_[H[:, t[0]], H[:, t[1]]]  # 0和1类的两个矩阵按行连接，就是把两矩阵左右相加，要求行数相等。
    H = normalization(H)  # 归一化处理

    model = KMeans(n_clusters=15)  # 新建20簇的Kmeans模型
    model.fit(np.real(H))  # 训练
    y_pred = model.labels_  #
    features = pd.DataFrame(features, index=np.arange(0, features.shape[0]))
    Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
    Mergefeature = pd.concat([features, Group], axis=1)


    init_centroid = np.asarray(Mergefeature.groupby("Group").mean())
    n_clusters = init_centroid.shape[0]

    return init_centroid, y_pred



def get_center_labels1(features, resolution=3.0):
    '''
    resolution: Value of the resolution parameter, use a value above
          (below) 1.0 if you want to obtain a larger (smaller) number
          of communities.
    '''

    print("\nInitializing cluster centroids using the louvain method.")

    adata0 = AnnData(features)#将嵌入数据转化为anndata形式，因为下面两个函数要求为anndata形式
    sc.pp.neighbors(adata0, n_neighbors=15, use_rep="X") #计算邻域图
    adata0 = sc.tl.louvain(adata0, resolution=resolution, random_state=0, copy=True)
    #分辨率越高就会找到更多更小的集群
    #sc.tl.louvain实际上为adata0添加obs，obs的名称是louvain，即一个简单的分类
    y_pred = adata0.obs['louvain']
    y_pred = np.asarray(y_pred, dtype=int)
    #将obs提取出来

    features = pd.DataFrame(adata0.X, index=np.arange(0, adata0.shape[0]))
    Group = pd.Series(y_pred, index=np.arange(0, adata0.shape[0]), name="Group")
    Mergefeature = pd.concat([features, Group], axis=1)
    #沿着指定的轴将多个dataframe或者series拼接到一起，axis（默认上下堆叠，等于1时左右堆叠）
    #将嵌入数据和obs，拼接到一起，便于分组

    init_centroid = np.asarray(Mergefeature.groupby("Group").mean())
    #Mergefeature.groupby("Group")将数据集按Group字段划分
    #将相同组别的按列取均值，不同组别不做处理，输出最终分类后数据
    n_clusters = init_centroid.shape[0]
    #输出louvain的类别，即初始聚类数量

    #print("\n " + str(n_clusters) + " micro-clusters detected. \n")
    return init_centroid, y_pred


def get_nearest_points_to_optimal_centers(X, optimal_centers, embedded_data):
    best_center_points = np.argmin(cdist(optimal_centers, embedded_data), axis=1)
    #给出指定方向（默认水平方向）最小值的下标，axis=1指纵轴；#cdist() 计算距离
    #寻找和25个嵌入中心距离最近的25个下标     25*1
    centers_cpu = X[best_center_points, :]  #data（1600*2000）数据中和嵌入中心最近的25个细胞
    embedded_centers_cpu = embedded_data[best_center_points, :]  #嵌入数据（1600*10）中和嵌入中心最近的25个细胞
    return centers_cpu, embedded_centers_cpu

def get_nearest_points(points_in_larger_cluster, center, size_smaller_cluster, max_cluster_size_diff_factor,
                        min_sample_size):

    distances = cdist(points_in_larger_cluster, [center])
    nearest_points = np.argsort(distances, axis=0) #返回的是元素值从小到大排序后的索引值的数组
    # Check if more points should be taken because the other cluster is too small
    sample_size = size_smaller_cluster * max_cluster_size_diff_factor
    if size_smaller_cluster + sample_size < min_sample_size:
        sample_size = min(min_sample_size - size_smaller_cluster, len(points_in_larger_cluster))
    subset_all_points = points_in_larger_cluster[nearest_points[:sample_size, 0]]
    #返回和中心距离较小的细胞，同时减少大簇中的细胞数
    return subset_all_points


def squared_euclidean_distance(centers, embedded, weights=None):
    ta = centers.unsqueeze(0)    #1*25*10
    tb = embedded.unsqueeze(1)     #25*1*10
    squared_diffs = (ta - tb)        #25*25*10
    if weights is not None:
        weights_unsqueezed = weights.unsqueeze(0).unsqueeze(1)
        squared_diffs = squared_diffs * weights_unsqueezed
    squared_diffs = squared_diffs.pow(2).sum(2)    #pow(n) n次方；#sum(2)对张量的第三维度求和
    return squared_diffs

def int_to_one_hot(label_tensor, n_labels):
    print("label_tensor",type(label_tensor))
    onehot = torch.zeros([label_tensor.shape[0], n_labels], dtype=torch.float, device=label_tensor.device)
    #返回一个形状为[label_tensor.shape[0], n_labels],类型为dtype，里面的每一个值都是0的tensor
    print(onehot.shape)
    print(label_tensor.unsqueeze(1).shape)
    onehot.scatter_(1, label_tensor.unsqueeze(1).long(), 1.0)
    #scatter_(dim, index, src)将src中数据根据index中的索引按照dim的方向进行填充,同时在原来的基础上对Tensor进行修改。
    #unsqueeze()函数起升维的作用,参数表示在哪个地方加一个维度
    return onehot

def dip_pval(data_dip, n_points):
    N, SIG, CV = _dip_table_values()
    i1 = N.searchsorted(n_points, side='left')
    #np.searchsorted(a, v, side='left', sorter=None)
    #在数组a中插入数组v（并不执行插入操作），返回一个下标列表，这个列表指明了v中对应元素应该插入在a中那个位置上;
    #当为left时，将返回第一个符合条件的元素下标；
    #当为right时，将返回最后一个符合条件的元素下标，如果没有符合的元素，将返回0或者N（a的长度）
    i0 = i1 - 1
    # if n falls outside the range of tabulated sample sizes, use the
    # critical values for the nearest tabulated n (i.e. treat them as
    # 'asymptotic')
    i0 = max(0, i0)
    i1 = min(N.shape[0] - 1, i1) #shape[0]读取矩阵行数
    # interpolate on sqrt(n)
    if i0 == i1 and i0 == N.shape[0] - 1:
        i0 = i1-1

    n0, n1 = N[[i0, i1]]#同时选取两个参数要用两个中括号嵌套
    fn = float(n_points - n0) / (n1 - n0)#float函数可以将一个十进制整数、十进制浮点数字符串或布尔值转化为十进制浮点数 #转化的是整除后的数
    y0 = np.sqrt(n0) * CV[i0]#numpy.sqrt(arr,out=None) arr:输入数组；out:如果给定了out，结果将储存在out中，out应与arr形状相同。此函数返回输入数组中每个元素的平方根数组。
    y1 = np.sqrt(n1) * CV[i1]
    sD = np.sqrt(n_points) * data_dip
    pval = 1. - np.interp(sD, y0 + fn * (y1 - y0), SIG)#通过给定横坐标xp(y0 + fn * (y1 - y0)),纵坐标yp(SIG),输出插入横坐标(sD)的纵坐标
    return pval


def _dip_table_values():
    N = np.array([4, 5, 6, 7, 8, 9, 10, 15, 20,
                  30, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
                  20000, 40000, 72000])

    SIG = np.array([0., 0.01, 0.02, 0.05, 0.1, 0.2,
                    0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                    0.9, 0.95, 0.98, 0.99, 0.995, 0.998,
                    0.999, 0.9995, 0.9998, 0.9999, 0.99995, 0.99998,
                    0.99999, 1.])

    #  table of critical values from https://github.com/alimuldal/diptest
    # ,and https://github.com/tatome/dip_test
    # [len(N), len(SIG)] table of critical values
    CV = np.array([[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.132559548782689,
                    0.157497369040235, 0.187401878807559, 0.20726978858736, 0.223755804629222, 0.231796258864192,
                    0.237263743826779, 0.241992892688593, 0.244369839049632, 0.245966625504691, 0.247439597233262,
                    0.248230659656638, 0.248754269146416, 0.249302039974259, 0.249459652323225, 0.24974836247845],
                   [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.108720593576329, 0.121563798026414, 0.134318918697053,
                    0.147298798976252, 0.161085025702604, 0.176811998476076, 0.186391796027944, 0.19361253363045,
                    0.196509139798845, 0.198159967287576, 0.199244279362433, 0.199617527406166, 0.199800941499028,
                    0.199917081834271, 0.199959029093075, 0.199978395376082, 0.199993151405815, 0.199995525025673,
                    0.199999835639211],
                   [0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333,
                    0.0833333333333333, 0.0924514470941933, 0.103913431059949, 0.113885220640212, 0.123071187137781,
                    0.13186973390253, 0.140564796497941, 0.14941924112913, 0.159137064572627, 0.164769608513302,
                    0.179176547392782, 0.191862827995563, 0.202101971042968, 0.213015781111186, 0.219518627282415,
                    0.224339047394446, 0.229449332154241, 0.232714530449602, 0.236548128358969, 0.2390887911995,
                    0.240103566436295, 0.244672883617768],
                   [0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.0725717816250742,
                    0.0817315478539489, 0.09405901819225269, 0.103244490800871, 0.110964599995697,
                    0.117807846504335, 0.124216086833531, 0.130409013968317, 0.136639642123068, 0.144240669035124,
                    0.159903395678336, 0.175196553271223, 0.184118659121501, 0.191014396174306, 0.198216795232182,
                    0.202341010748261, 0.205377566346832, 0.208306562526874, 0.209866047852379, 0.210967576933451,
                    0.212233348558702, 0.212661038312506, 0.21353618608817],
                   [0.0625, 0.0625, 0.06569119945032829, 0.07386511360717619, 0.0820045917762512,
                    0.0922700601131892, 0.09967371895993631, 0.105733531802737, 0.111035129847705,
                    0.115920055749988, 0.120561479262465, 0.125558759034845, 0.141841067033899, 0.153978303998561,
                    0.16597856724751, 0.172988528276759, 0.179010413496374, 0.186504388711178, 0.19448404115794,
                    0.200864297005026, 0.208849997050229, 0.212556040406219, 0.217149174137299, 0.221700076404503,
                    0.225000835357532, 0.233772919687683],
                   [0.0555555555555556, 0.0613018090298924, 0.0658615858179315, 0.0732651142535317,
                    0.0803941629593475, 0.0890432420913848, 0.0950811420297928, 0.09993808978110461,
                    0.104153560075868, 0.108007802361932, 0.112512617124951, 0.122915033480817, 0.136412639387084,
                    0.146603784954019, 0.157084065653166, 0.164164643657217, 0.172821674582338, 0.182555283567818,
                    0.188658833121906, 0.194089120768246, 0.19915700809389, 0.202881598436558, 0.205979795735129,
                    0.21054115498898, 0.21180033095039, 0.215379914317625],
                   [0.05, 0.0610132555623269, 0.0651627333214016, 0.0718321619656165, 0.077966212182459,
                    0.08528353598345639, 0.09032041737070989, 0.0943334983745117, 0.0977817630384725,
                    0.102180866696628, 0.109960948142951, 0.118844767211587, 0.130462149644819, 0.139611395137099,
                    0.150961728882481, 0.159684158858235, 0.16719524735674, 0.175419540856082, 0.180611195797351,
                    0.185286416050396, 0.191203083905044, 0.195805159339184, 0.20029398089673, 0.205651089646219,
                    0.209682048785853, 0.221530282182963],
                   [0.0341378172277919, 0.0546284208048975, 0.0572191260231815, 0.0610087367689692,
                    0.06426571373304441, 0.06922341079895911, 0.0745462114365167, 0.07920308789817621,
                    0.083621033469191, 0.08811984822029049, 0.093124666680253, 0.0996694393390689,
                    0.110087496900906, 0.118760769203664, 0.128890475210055, 0.13598356863636, 0.142452483681277,
                    0.150172816530742, 0.155456133696328, 0.160896499106958, 0.166979407946248, 0.17111793515551,
                    0.175900505704432, 0.181856676013166, 0.185743454151004, 0.192240563330562],
                   [0.033718563622065, 0.0474333740698401, 0.0490891387627092, 0.052719998201553,
                    0.0567795509056742, 0.0620134674468181, 0.06601638720690479, 0.06965060750664009,
                    0.07334377405927139, 0.07764606628802539, 0.0824558407118372, 0.08834462700173699,
                    0.09723460181229029, 0.105130218270636, 0.114309704281253, 0.120624043335821, 0.126552378036739,
                    0.13360135382395, 0.138569903791767, 0.14336916123968, 0.148940116394883, 0.152832538183622,
                    0.156010163618971, 0.161319225839345, 0.165568255916749, 0.175834459522789],
                   [0.0262674485075642, 0.0395871890405749, 0.0414574606741673, 0.0444462614069956,
                    0.0473998525042686, 0.0516677370374349, 0.0551037519001622, 0.058265005347493,
                    0.0614510857304343, 0.0649164408053978, 0.0689178762425442, 0.0739249074078291,
                    0.08147913793901269, 0.0881689143126666, 0.0960564383013644, 0.101478558893837,
                    0.10650487144103, 0.112724636524262, 0.117164140184417, 0.121425859908987, 0.126733051889401,
                    0.131198578897542, 0.133691739483444, 0.137831637950694, 0.141557509624351, 0.163833046059817],
                   [0.0218544781364545, 0.0314400501999916, 0.0329008160470834, 0.0353023819040016,
                    0.0377279973102482, 0.0410699984399582, 0.0437704598622665, 0.0462925642671299,
                    0.048851155289608, 0.0516145897865757, 0.0548121932066019, 0.0588230482851366,
                    0.06491363240467669, 0.0702737877191269, 0.07670958860791791, 0.0811998415355918,
                    0.0852854646662134, 0.09048478274902939, 0.0940930106666244, 0.0974904344916743,
                    0.102284204283997, 0.104680624334611, 0.107496694235039, 0.11140887547015, 0.113536607717411,
                    0.117886716865312],
                   [0.0164852597438403, 0.022831985803043, 0.0238917486442849, 0.0256559537977579,
                    0.0273987414570948, 0.0298109370830153, 0.0317771496530253, 0.0336073821590387,
                    0.0354621760592113, 0.0374805844550272, 0.0398046179116599, 0.0427283846799166,
                    0.047152783315718, 0.0511279442868827, 0.0558022052195208, 0.059024132304226,
                    0.0620425065165146, 0.06580160114660991, 0.0684479731118028, 0.0709169443994193,
                    0.0741183486081263, 0.0762579402903838, 0.0785735967934979, 0.08134583568891331,
                    0.0832963013755522, 0.09267804230967371],
                   [0.0111236388849688, 0.0165017735429825, 0.0172594157992489, 0.0185259426032926,
                    0.0197917612637521, 0.0215233745778454, 0.0229259769870428, 0.024243848341112,
                    0.025584358256487, 0.0270252129816288, 0.0286920262150517, 0.0308006766341406,
                    0.0339967814293504, 0.0368418413878307, 0.0402729850316397, 0.0426864799777448,
                    0.044958959158761, 0.0477643873749449, 0.0497198001867437, 0.0516114611801451,
                    0.0540543978864652, 0.0558704526182638, 0.0573877056330228, 0.0593365901653878,
                    0.0607646310473911, 0.0705309107882395],
                   [0.00755488597576196, 0.0106403461127515, 0.0111255573208294, 0.0119353655328931,
                    0.0127411306411808, 0.0138524542751814, 0.0147536004288476, 0.0155963185751048,
                    0.0164519238025286, 0.017383057902553, 0.0184503949887735, 0.0198162679782071,
                    0.0218781313182203, 0.0237294742633411, 0.025919578977657, 0.0274518022761997,
                    0.0288986369564301, 0.0306813505050163, 0.0320170996823189, 0.0332452747332959,
                    0.0348335698576168, 0.0359832389317461, 0.0369051995840645, 0.0387221159256424,
                    0.03993025905765, 0.0431448163617178],
                   [0.00541658127872122, 0.00760286745300187, 0.007949878346447991, 0.008521651834359399,
                    0.00909775605533253, 0.009889245210140779, 0.0105309297090482, 0.0111322726797384,
                    0.0117439009052552, 0.012405033293814, 0.0131684179320803, 0.0141377942603047,
                    0.0156148055023058, 0.0169343970067564, 0.018513067368104, 0.0196080260483234,
                    0.0206489568587364, 0.0219285176765082, 0.0228689168972669, 0.023738710122235,
                    0.0248334158891432, 0.0256126573433596, 0.0265491336936829, 0.027578430100536, 0.0284430733108,
                    0.0313640941982108],
                   [0.00390439997450557, 0.00541664181796583, 0.00566171386252323, 0.00607120971135229,
                    0.0064762535755248, 0.00703573098590029, 0.00749421254589299, 0.007920878896017331,
                    0.008355737247680061, 0.00882439333812351, 0.00936785820717061, 0.01005604603884,
                    0.0111019116837591, 0.0120380990328341, 0.0131721010552576, 0.0139655122281969,
                    0.0146889122204488, 0.0156076779647454, 0.0162685615996248, 0.0168874937789415,
                    0.0176505093388153, 0.0181944265400504, 0.0186226037818523, 0.0193001796565433,
                    0.0196241518040617, 0.0213081254074584],
                   [0.00245657785440433, 0.00344809282233326, 0.00360473943713036, 0.00386326548010849,
                    0.00412089506752692, 0.00447640050137479, 0.00476555693102276, 0.00503704029750072,
                    0.00531239247408213, 0.00560929919359959, 0.00595352728377949, 0.00639092280563517,
                    0.00705566126234625, 0.0076506368153935, 0.00836821687047215, 0.008863578928549141,
                    0.00934162787186159, 0.009932186363240289, 0.0103498795291629, 0.0107780907076862,
                    0.0113184316868283, 0.0117329446468571, 0.0119995948968375, 0.0124410052027886,
                    0.0129467396733128, 0.014396063834027],
                   [0.00174954269199566, 0.00244595133885302, 0.00255710802275612, 0.00273990955227265,
                    0.0029225480567908, 0.00317374638422465, 0.00338072258533527, 0.00357243876535982,
                    0.00376734715752209, 0.00397885007249132, 0.00422430013176233, 0.00453437508148542,
                    0.00500178808402368, 0.00542372242836395, 0.00592656681022859, 0.00628034732880374,
                    0.00661030641550873, 0.00702254699967648, 0.00731822628156458, 0.0076065423418208,
                    0.00795640367207482, 0.008227052458435399, 0.00852240989786251, 0.00892863905540303,
                    0.009138539330002131, 0.009522345795667729],
                   [0.00119458814106091, 0.00173435346896287, 0.00181194434584681, 0.00194259470485893,
                    0.00207173719623868, 0.00224993202086955, 0.00239520831473419, 0.00253036792824665,
                    0.00266863168718114, 0.0028181999035216, 0.00299137548142077, 0.00321024899920135,
                    0.00354362220314155, 0.00384330190244679, 0.00420258799378253, 0.00445774902155711,
                    0.00469461513212743, 0.00499416069129168, 0.00520917757743218, 0.00540396235924372,
                    0.00564540201704594, 0.00580460792299214, 0.00599774739593151, 0.00633099254378114,
                    0.00656987109386762, 0.00685829448046227],
                   [0.000852415648011777, 0.00122883479310665, 0.00128469304457018, 0.00137617650525553,
                    0.00146751502006323, 0.00159376453672466, 0.00169668445506151, 0.00179253418337906,
                    0.00189061261635977, 0.00199645471886179, 0.00211929748381704, 0.00227457698703581,
                    0.00250999080890397, 0.00272375073486223, 0.00298072958568387, 0.00315942194040388,
                    0.0033273652798148, 0.00353988965698579, 0.00369400045486625, 0.00383345715372182,
                    0.00400793469634696, 0.00414892737222885, 0.0042839159079761, 0.00441870104432879,
                    0.00450818604569179, 0.00513477467565583],
                   [0.000644400053256997, 0.000916872204484283, 0.000957932946765532, 0.00102641863872347,
                    0.00109495154218002, 0.00118904090369415, 0.00126575197699874, 0.00133750966361506,
                    0.00141049709228472, 0.00148936709298802, 0.00158027541945626, 0.00169651643860074,
                    0.00187306184725826, 0.00203178401610555, 0.00222356097506054, 0.00235782814777627,
                    0.00248343580127067, 0.00264210826339498, 0.0027524322157581, 0.0028608570740143,
                    0.00298695044508003, 0.00309340092038059, 0.00319932767198801, 0.00332688234611187,
                    0.00339316094477355, 0.00376331697005859]])
    return N, SIG, CV


def plot_umap(embeddings, labels, title="umap", fontsize = 10):

    fit = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2, metric='cosine', random_state=123)
    embeddings = fit.fit_transform(embeddings)
    formatting = AnnData(embeddings)
    formatting.obs["cell_type"] = labels.astype(str)
    sc.pp.neighbors(formatting, n_neighbors=25, use_rep='X', n_pcs=40)
    sc.tl.umap(formatting)
    sc.pl.umap(formatting, color=["cell_type"],
               legend_fontsize=fontsize,
               title=title,
               )


def find_resolution(adata_, n_clusters, random=666):
    """A function to find the louvain resolution tjat corresponds to a prespecified number of clusters, if it exists.


    Arguments:
    ------------------------------------------------------------------
    - adata_: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to low dimension features.
    - n_clusters: `int`, Number of clusters.
    - random: `int`, The random seed.

    Returns:
    ------------------------------------------------------------------
    - resolution: `float`, The resolution that gives n_clusters after running louvain's clustering algorithm.
    """

    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]

    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions) / 2
        adata = sc.tl.louvain(adata_, resolution=current_res, random_state=random, copy=True)
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))

        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res

        iteration = iteration + 1

    return current_res


def load_PBMC3k(min_genes: int = 200, min_cells: int = 3) -> ad.AnnData:
    """s
    Load the PBMC3k dataset as an example.

    Parameters
    ----------
    min_genes: int
        Minimum number of genes expressed required for a cell to pass filtering.
    min_cells: int
        Minimum number of cells expressed required for a gene to pass filtering.

    Returns
    -------
    adata : ad.AnnData
        The PBMC3k dataset as an `AnnData` object.
    """
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata.X = adata.X.toarray()
    return adata


def load_simulated_data(n_genes: int = 15000, n_celltype: int = 5, n_observations: int = 1000) -> ad.AnnData:
    """
    Gaussian Blobs.

    Parameters
    ----------
    n_genes
        The number of genes.
    n_celltype
        The number of cell types.
    n_observations
        The number of cells.
    Returns
    -------
    adata
        A simulated `AnnData` object.
    """
    adata = sc.datasets.blobs(n_variables=n_genes, n_centers=n_celltype, n_observations=n_observations)
    adata.X[adata.X < 0] = 0
    adata.X[adata.X > 5] = 0
    adata.X = np.round(adata.X, decimals=0)
    adata.obs.rename(columns={'blobs': 'celltype'}, inplace=True)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    return adata


def load_mouse_brain(min_genes: int = 200, min_spots: int = 3) -> Tuple[ad.AnnData, np.ndarray]:
    adata = sq.datasets.visium('V1_Adult_Mouse_Brain', include_hires_tiff=True)
    adata.var_names_make_unique()
    img = cv2.imread(adata.uns['spatial']['V1_Adult_Mouse_Brain']['metadata']['source_image_path'])
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_spots)
    adata.X = adata.X.toarray()
    return adata, img


def set_logger(verbosity: Literal[0, 1, 2] = 1):
    """
    Set the verbosity level.

    Parameters
    ----------
    verbosity
        0 (only print warnings and errors), 1 (also print info), 2 (also print debug messages)
    """
    def formatter(record: dict):
        if record['level'].name in ('DEBUG', 'INFO'):
            return "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " \
                   "<level>{level: <5}</level> | " \
                   "<level>{message}\n</level>"
        else:
            return "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " \
                   "<level>{level: <8}</level> | " \
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}\n</level>"

    level_dict = {0: 'WARNING', 1: 'INFO', 2: 'DEBUG'}
    logger.remove()
    logger.add(sys.stdout, colorize=True, level=level_dict[verbosity], format=formatter)


