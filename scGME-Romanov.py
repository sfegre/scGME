import numpy as np
import torch
import torch.nn as nn
import os
import random
import platform
import ctypes
import argparse
import pyreadr
import torch.backends.cudnn as cudnn #相当于为torch.backends.cudnn定义别名cudnn
import preprocessing as pro
from mimimodel import scClust_Autoencoder
from util import *
from torch.autograd import Variable
from layer import ZINBLoss, MeanAct, DispAct
from torch.optim import Adam
from torch.nn.parameter import Parameter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari_score
from evaluation import cluster_acc
import h5py



# fix random seeds
seed=666
cudnn.deterministic = True
#设置为True，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
cudnn.benchmark = True
#cuDNN是英伟达专门为深度神经网络所开发出来的GPU加速库，针对卷积、池化等等常见操作做了非常多的底层优化，比一般的GPU程序要快很多。在使用cuDNN的时候，默认为False。
#设置为True将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。适用场景是网络结构固定，网络输入形状不变（即一般情况下都适用）。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
random.seed(seed)#设置随机种子以此让每次产生的随机数相同
torch.manual_seed(seed)#为CPU设置随机种子
torch.cuda.manual_seed_all(seed)#为所有GPU设置随机种子
np.random.seed(seed)#生成随机数


C_DIP_FILE = None

def dip_test(X, is_data_sorted=False, debug=False):#def function name (parameter):
    n_points = X.shape[0]#function body  #shape[0]读取矩阵行数;shape[1]读取矩阵列数
    data_dip = dip(X, just_dip=True, is_data_sorted=is_data_sorted, debug=debug)
    pval = dip_pval(data_dip, n_points)
    return data_dip, pval#return


def dip(X, just_dip=False, is_data_sorted=False, debug=False):
    assert X.ndim == 1, "Data must be 1-dimensional for the dip-test. Your shape:{0}".format(X.shape)
    #assert X.ndim == 1 ,"X的维度==1"   #这句的意思：如果X的维度==1，程序正常往下运行
    N = len(X)#len ()是python的内置函数，用于返回字符串、列表、字典、元组等对象的长度，即元素的个数
    if not is_data_sorted:#如果not后面的表达式为False,则执行冒号后面的语句
        X = np.sort(X)#对数组进行排序，详见浏览器收藏
    if N < 4 or X[0] == X[-1]:#[0]第一项;[-1]最后一项
        d = 0.0
        return d if just_dip else (d, None, None)
    #if just_dip:
    #    return:d
    #    else:
    #    return:(d, None, None)

    # Prepare data to match C data types
    if C_DIP_FILE is None:
        load_c_dip_file()
    X = np.asarray(X, dtype=np.float64) #将输入数据（列表的列表，元组的元组，元组的列表等）转换为矩阵形式
    X_c = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    N_c = np.array([N]).ctypes.data_as(ctypes.POINTER(ctypes.c_int)) #np.array()把列表转化为数组
    dip_value = np.zeros(1, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)) #包含5个元素的零矩阵
    low_high = np.zeros(4).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    modal_triangle = np.zeros(3).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    gcm = np.zeros(N).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    lcm = np.zeros(N).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    mn = np.zeros(N).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    mj = np.zeros(N).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    debug_c = np.array([1 if debug else 0]).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Execute C dip test
    _ = C_DIP_FILE.diptst(X_c, N_c, dip_value, low_high, modal_triangle, gcm, lcm, mn, mj, debug_c)
    dip_value = dip_value[0]
    if just_dip:
        return dip_value
    else:
        low_high = (low_high[0], low_high[1], low_high[2], low_high[3])
        modal_triangle = (modal_triangle[0], modal_triangle[1], modal_triangle[2])
        return dip_value, low_high, modal_triangle

def load_c_dip_file(): # 寻找当前文件夹所处的路径，并在当前路径调用dll文件，定义digtst函数的参数和返回值的类型
    global C_DIP_FILE  #Python中定义函数时，若想在函数内部对函数外的变量进行操作，就需要在函数内部声明其为global
    files_path = os.path.dirname(__file__)#输出去掉文件名的当前路径
    if platform.system() == "Windows":  #platform.system()返回当前操作系统的名字
        dip_compiled = files_path + "/dip.dll"
    else:
        dip_compiled = "dip.so"

    print(dip_compiled)
    if os.path.isfile(dip_compiled):  #如果路径中存在此文件，则输出True，否则为False
        # load c file
        try:

            C_DIP_FILE = ctypes.CDLL(dip_compiled)  #python下调用C库前加载dll文件，这里采用了cdecl调用规定
            C_DIP_FILE.diptst.restype = None   #定义diptst函数的返回类型,,使用None表示void，即不返回任何结果的函数
            C_DIP_FILE.diptst.argtypes = [ctypes.POINTER(ctypes.c_double),#ctypes.POINTER()定义某个类型的指针
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int)]
            #参数类型用关键字argtypes定义，argtypes必须是一个序列，如tuple或list,否则会报错 返回类型用 restype 定义,使用None表示void，即不返回任何结果的函数
        except Exception as e:
            print("[WARNING] Error while loading the C compiled dip file.")
            raise e
        #try:
        #需要检测异常的代码
        #except Exception as e:
        # e 就是异常实例
        #except as  检验一段可能有错误的代码；raise 抛出错误
    else:
        raise Exception("C compiled dip file can not be found.\n"
                        "On Linux execute: gcc -fPIC -shared -o dip.so dip.c\n"
                        "Or Please ensure the dip.so was added in your LD_LIBRARY_PATH correctly by executing \n"
                        "(export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./dip.so)   in the current directory of the ADClust folder. \n")


def get_trained_autoencoder(trainloader, learning_rate, n_epochs, input_dim, embedding_size,
                           optimizer_class, device, loss_fn, autoencoder_class):

    if embedding_size > input_dim:
        print(
            "WARNING: embedding_size is larger than the dimensionality of the input dataset. Setting embedding_size to",
            input_dim)
        embedding_size = input_dim

    if judge_system():
        act_fn = torch.nn.ReLU  #一个函数，函数公式见收藏
    else:
        act_fn = torch.nn.LeakyReLU   #一个函数，函数公式见收藏

    # Pretrain Autoencoder
    print(input_dim)
    autoencoder = autoencoder_class(input_dim=input_dim, embedding_size=embedding_size,
                                    act_fn=act_fn).to(device)

    optimizer = optimizer_class(autoencoder.parameters(), lr=learning_rate)
    autoencoder.start_training(trainloader, n_epochs, device, optimizer, loss_fn)

    return autoencoder

def _get_dip_matrix(data, dip_centers, dip_labels, n_clusters, max_cluster_size_diff_factor=3, min_sample_size=100):
    dip_matrix = np.zeros((n_clusters, n_clusters))

    # Loop over all combinations of centers
    for i in range(0, n_clusters - 1):  #计数
        for j in range(i + 1, n_clusters):
            center_diff = dip_centers[i] - dip_centers[j]
            points_in_i = data[dip_labels == i]
            points_in_j = data[dip_labels == j]
            points_in_i_or_j = np.append(points_in_i, points_in_j, axis=0)
            proj_points = np.dot(points_in_i_or_j, center_diff)  #获取两个元素的乘积
            _, dip_p_value = dip_test(proj_points)

            # Check if clusters sizes differ heavily
            if points_in_i.shape[0] > points_in_j.shape[0] * max_cluster_size_diff_factor or \
                    points_in_j.shape[0] > points_in_i.shape[0] * max_cluster_size_diff_factor:
                if points_in_i.shape[0] > points_in_j.shape[0] * max_cluster_size_diff_factor:
                    points_in_i = get_nearest_points(points_in_i, dip_centers[j], points_in_j.shape[0],
                                                      max_cluster_size_diff_factor, min_sample_size)
                    #get_nearest_points留下和嵌入中心 j 距离小的points_in_i细胞，因为这里计算的是dip-score,即检验i簇和j簇的相似性，所以选择距离小的
                elif points_in_j.shape[0] > points_in_i.shape[0] * max_cluster_size_diff_factor:
                    points_in_j = get_nearest_points(points_in_j, dip_centers[i], points_in_i.shape[0],
                                                      max_cluster_size_diff_factor, min_sample_size)
                points_in_i_or_j = np.append(points_in_i, points_in_j, axis=0)  #将一个数组附加到另一个数组的尾部
                proj_points = np.dot(points_in_i_or_j, center_diff)
                _, dip_p_value_2 = dip_test(proj_points)
                dip_p_value = min(dip_p_value, dip_p_value_2)
                #最终要将dip-score和阈值作比较，大于阈值才能合并，故这里选择较小值

            # Add pval to dip matrix
            dip_matrix[i][j] = dip_p_value
            dip_matrix[j][i] = dip_p_value

    return dip_matrix

def _merge_by_dip_value(X, embedded_data, cluster_labels_cpu, dip_argmax, n_clusters_current, centers_cpu, embedded_centers_cpu):

    # Get points in clusters
    #dip_argmax即大于dip阈值的dip-score在dip矩阵中的坐标，横纵坐标为生成dip-score的两个聚类标签;cluster_labels_cpu即1600个细胞的聚类标签
    points_in_center_1 = len(cluster_labels_cpu[cluster_labels_cpu == dip_argmax[0]])
    #dip_argmax横坐标代表的聚类标签中细胞的数量
    points_in_center_2 = len(cluster_labels_cpu[cluster_labels_cpu == dip_argmax[1]])
    #dip_argmax纵坐标代表的聚类标签中细胞的数量

    # update labels
    for j, l in enumerate(cluster_labels_cpu):
        #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        if l == dip_argmax[0] or l == dip_argmax[1]:
            cluster_labels_cpu[j] = n_clusters_current - 1
        elif l < dip_argmax[0] and l < dip_argmax[1]:
            cluster_labels_cpu[j] = l
        elif l > dip_argmax[0] and l > dip_argmax[1]:
            cluster_labels_cpu[j] = l - 2
        else:
            cluster_labels_cpu[j] = l - 1

    # Find new center position
    optimal_new_center = (embedded_centers_cpu[dip_argmax[0]] * points_in_center_1 +
                          embedded_centers_cpu[dip_argmax[1]] * points_in_center_2) / (
                                 points_in_center_1 + points_in_center_2)
    #将超过阈值的dip-score关联的两个微簇求平均值，即合并成为一个微簇，即两个超过阈值中心合并为一个新中心     10
    new_center_cpu, new_embedded_center_cpu = get_nearest_points_to_optimal_centers(X, [optimal_new_center],
                                                                                     embedded_data)
    #和新中心距离最近的data、embedded-data中的一个细胞数据

    # Remove the two old centers and add the new one
    centers_cpu_tmp = np.delete(centers_cpu, dip_argmax, axis=0)   #删除
    #在25个中心中去掉超过dip-score的两个中心
    centers_cpu = np.append(centers_cpu_tmp, new_center_cpu, axis=0)  #将一个数组附加到另一个数组的尾部
    #将计算出的新中心放到centers_cpu_tmp中
    embedded_centers_cpu_tmp = np.delete(embedded_centers_cpu, dip_argmax, axis=0)
    ##在25个中心中去掉超过dip-score的两个中心
    embedded_centers_cpu = np.append(embedded_centers_cpu_tmp, new_embedded_center_cpu, axis=0)
    ##将计算出的新中心放到embedded_centers_cpu_tmp中

    # Update dip values
    dip_matrix_cpu = _get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu, n_clusters_current)
    #计算新的dip矩阵
    return cluster_labels_cpu, centers_cpu, embedded_centers_cpu, dip_matrix_cpu


def _scclust_training(X, X_raw, sf, n_clusters_current, dip_merge_threshold, batch_size, centers_cpu,
                      cluster_labels_cpu,dip_matrix_cpu,n_clusters_max, n_clusters_min, dedc_epochs, optimizer,
                      loss_fn, lambda_1, lambda_2, autoencoder, device, trainloader, testloader):

    i = 0
    while i < dedc_epochs:
        print("------------------------------")
        print("i",i)
        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                *(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(sf), torch.Tensor(cluster_labels_cpu))),
            # torch.from_numpy把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变.float()
            # torch.arange返回一维张量，其值介于区间[start,end),以step为步长等间隔取值
            # torch.Tensor(X), torch.Tensor(X_raw.todense()), torch.Tensor(sf)
            batch_size=batch_size,  # 每个batch加载多少个样本
            # sample random mini-batches from the data
            shuffle=True,  # 设置为True时会在每个epoch重新打乱数据
            drop_last=False)


        #cluster_labels_torch = torch.from_numpy(cluster_labels_cpu).long().to(device)
        print("cluster_labels_cpu",cluster_labels_cpu)
        #把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变
        centers_torch = torch.from_numpy(centers_cpu).float().to(device)
        dip_matrix_torch = torch.from_numpy(dip_matrix_cpu).float().to(device)

        # Get dip costs matrix
        dip_matrix_eye = dip_matrix_torch + torch.eye(n_clusters_current, device=device)
        #torch.eye()创建一个二维矩阵m × n，对角全为1，其它都为0
        dip_matrix_final = dip_matrix_eye / dip_matrix_eye.sum(1).reshape((-1, 1))
        #reshape函数用于在不更改数据的情况下为数组赋予新形状
        # X.sum(1)求数组每一行的和
        #对每一行数据求均值进行标准化

        # Iterate over batches
        for batch_idx, (x_batch, x_raw_batch, sf_batch, labels_batch) in enumerate(trainloader):
            print("batch_idx",batch_idx)
            batch_data = x_batch.to(device)
            x_raw_tensor = x_raw_batch.to(device)
            sf_tensor = sf_batch.to(device)
            labels_batch=labels_batch.long().to(device)
            print("labels_batch",labels_batch)
            #z, mean, disp, pi = self.forward(x_tensor)
            #loss = self.zinb_loss(x=x_raw_tensor, mean=mean, disp=disp, pi=pi, scale_factor=sf_tensor)
        #for batch, ids in trainloader:
            # ids可能是每一批次中细胞在data中的排列顺序

            #batch_data = batch.to(device)  #?
            print(batch_data.shape)
            embedded = autoencoder.encode(batch_data)
            reconstruction = autoencoder.decode(embedded)
            embedded_centers_torch = autoencoder.encode(centers_torch)

            # Reconstruction Loss
            #ae_loss = loss_fn(reconstruction, batch_data)

            # Get distances between points and centers. Get nearest center
            squared_diffs = squared_euclidean_distance(embedded_centers_torch, embedded)

            # Update labels? Pause is needed, so cluster labels can adjust to the new structure

            if i != 0:
                # Update labels
                current_labels = squared_diffs.argmin(1)
                # cluster_labels_torch[ids] = current_labels
                # 每一批次的squared_diffs是128*25，在128的每一行中寻找最小值的下标即为新的聚类标签
            else:
                current_labels = labels_batch
                # 当i=0时取原始聚类标签


            print("current_labels",current_labels)
            print("n_clusters_current",n_clusters_current)
            onehot_labels = int_to_one_hot(current_labels, n_clusters_current).float()
            # 将每一批次中128个细胞的聚类标签放到一个128*25的矩阵中。
            cluster_relationships = torch.matmul(onehot_labels, dip_matrix_final)
            # tensor的乘法，输入可以是高维的
            escaped_diffs = cluster_relationships * squared_diffs

            # Normalize loss by cluster distances
            squared_center_diffs = squared_euclidean_distance(embedded_centers_torch, embedded_centers_torch)

            # Ignore zero values (diagonal)
            mask = torch.where(squared_center_diffs != 0)
            # 寻找squared_center_diffs中不为0的下标，结果为两组张量，下标的行在第一组，下标的列在第二组
            # 根据条件，返回从x,y中选择元素所组成的张量。即如果满足条件，则返回x中元素。若不满足，返回y中元素。
            masked_center_diffs = squared_center_diffs[mask[0], mask[1]]
            sqrt_masked_center_diffs = masked_center_diffs.sqrt()
            masked_center_diffs_std = sqrt_masked_center_diffs.std() if len(sqrt_masked_center_diffs) > 2 else 0

            # Loss function
            cluster_loss = escaped_diffs.sum(1).mean() * (
                    1 + masked_center_diffs_std) / sqrt_masked_center_diffs.mean()
            #论文中Lclu的公式，escaped_diffs.sum(1)即对各簇求和，escaped_diffs.sum(1).mean()进一步对一批次中各细胞求和再除以总细胞数，即求均值
            cluster_loss *= 1
            #loss = ae_loss * ae_weight_loss + cluster_loss

            x_tensor = Variable(batch_data)
            x_raw_tensor = Variable(batch_data)
            print("adata.obs",adata.obs)
            #adata.obs['size_factors'] = adata.obs.nCount_RNA / np.median(adata.obs.n_counts)
            #nCount_RNA、n_counts
            sf_tensor = sf_batch.to(device)
            #sf_tensor = Variable(torch.Tensor(adata.obs.size_factors))
            z, mean, disp, pi = autoencoder.forward(x_tensor)
            all_weights = dict()
            all_weights['Coef'] = Parameter(1.0e-4 * torch.ones(size=(len(batch_data),len(batch_data))))
            weights = all_weights
            Coef = weights['Coef']
            z_c = torch.matmul(Coef, z)
            loss_reconst = 1 / 2 * torch.sum(torch.pow((x_tensor - mean), 2))
            # loss_reconst = 1 / 2 * torch.sum(torch.pow((x_tensor - x_bar), 2))
            loss_reg = torch.sum(torch.pow(Coef, 2))
            loss_selfexpress = 1 / 2 * torch.sum(torch.pow((z - z_c), 2))
            loss_zinb = ZINBLoss().forward(x=x_raw_tensor, mean=mean, disp=disp, pi=pi, scale_factor=sf_tensor)
            loss = (0.2 * loss_reconst + lambda_1 * loss_reg + lambda_2 * loss_selfexpress) ** 1 / 10 + loss_zinb +cluster_loss
            #lambda_1=1.0, lambda_2=0.5

            # Backward pass
            optimizer.zero_grad() #将梯度置零
            loss.backward() #见收藏
            optimizer.step() #见收藏

        # Update centers
        embedded_data = encode_batchwise(testloader, autoencoder, device)  #1600*10
        embedded_centers_cpu = autoencoder.encode(centers_torch).detach().cpu().numpy()  #25*10
        cluster_labels_cpu = np.argmin(cdist(embedded_centers_cpu, embedded_data), axis=0)
        #cdist()计算马氏距离             25*1600---->1*1600
        optimal_centers = np.array([np.mean(embedded_data[cluster_labels_cpu == cluster_id], axis=0) for cluster_id in
                                    range(n_clusters_current)])   #25*10
        #np.mean()计算给定数组沿指定轴的算术平均值,axis = 0：压缩行，对各列求均值，返回1*n的矩阵
        #embedded_data[cluster_labels_cpu == cluster_id]寻找聚类标签等于i(i=0...24)的嵌入数据
        #np.mean(embedded_data[cluster_labels_cpu == cluster_id], axis=0)求聚类标签等于i的嵌入数据每列的均值，得到1*10的结果
        centers_cpu, embedded_centers_cpu = get_nearest_points_to_optimal_centers(X, optimal_centers, embedded_data)

        # Update Dips
        dip_matrix_cpu = _get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu, n_clusters_current)
        #embedded_centers_cpu、cluster_labels_cpu与上一次的dip-maxtrix计算不同


        # i is increased here. Else next iteration will start with i = 1 instead of 0 after a merge
        i += 1

        # Start merging procedure
        dip_argmax = np.unravel_index(np.argmax(dip_matrix_cpu, axis=None), dip_matrix_cpu.shape)
        #np.unravel_index()获取一个一维组int类型的索引值在一个多维数组中的位置(此一维组和多维组是同一个组的不同维度）

        # Is merge possible?
        if i != 0:
            while dip_matrix_cpu[dip_argmax] >= dip_merge_threshold and n_clusters_current > n_clusters_min:
                # Reset iteration and reduce number of cluster
                i = 0
                n_clusters_current -= 1
                cluster_labels_cpu, centers_cpu, embedded_centers_cpu, dip_matrix_cpu = \
                    _merge_by_dip_value(X, embedded_data, cluster_labels_cpu, dip_argmax, n_clusters_current,
                                        centers_cpu,  embedded_centers_cpu)
                #将超过阈值的dip-score关联的两个微簇合并，得到新的聚类标签，新的data中心，新的embedded中心，以及新的dip矩阵
                dip_argmax = np.unravel_index(np.argmax(dip_matrix_cpu, axis=None), dip_matrix_cpu.shape)
                #在新的dip矩阵中寻找最大的dip-score

        if n_clusters_current == 1:
            print("Only one cluster left")
            break

    return cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder


def _scclust(X, X_raw, sf, dip_merge_threshold, batch_size, learning_rate, pretrain_epochs, embedding_size, dedc_epochs,lambda_1, lambda_2, n_clusters_max, n_clusters_min,optimizer_class=torch.optim.Adam, loss_fn=torch.nn.MSELoss()):
    #torch.optim.Adam更新参数


    device = detect_device()


    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(sf))),
        #torch.from_numpy把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变.float()
        #torch.arange返回一维张量，其值介于区间[start,end),以step为步长等间隔取值
        #torch.Tensor(X), torch.Tensor(X_raw.todense()), torch.Tensor(sf)
        batch_size=batch_size,#每个batch加载多少个样本
        # sample random mini-batches from the data
        shuffle=True,#设置为True时会在每个epoch重新打乱数据
        drop_last=False)
        #如果数据集大小不能被batch_size整除，则设置为True后可删除最后一个不完整的batch。
        # 如果设为False并且数据集的大小不能被batch_size整除，则最后一个batch将更小。

    # create a Dataloader to test the autoencoder in mini-batch fashion
    testloader = torch.utils.data.DataLoader(torch.from_numpy(X).float(),
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False)


    autoencoder = get_trained_autoencoder(trainloader, learning_rate, pretrain_epochs, X.shape[1], embedding_size,
                                          optimizer_class, device, loss_fn, scClust_Autoencoder)

    #get_trained_autoencoder(trainloader, learning_rate, n_epochs, device, optimizer_class, loss_fn,
                            #input_dim, embedding_size, autoencoder_class):
    #(X, dip_merge_threshold, batch_size, learning_rate, pretrain_epochs, dedc_epochs,
             #lambda_1, lambda_2, denoise, sigma,n_clusters_max, n_clusters_min,
              #debug, optimizer_class=torch.optim.Adam, loss_fn=layer.ZINBLoss()):
    #利用训练集数据得到自编码器模型


    embedded_data = encode_batchwise(testloader, autoencoder, device)
    #针对上一步得到的模型，使用测试集小批量输入到编码器中得到测试数据的嵌入数据
    print("embedded_data",type(embedded_data))

    # Execute Louvain algorithm to get initial micro-clusters in embedded space
    init_centers, cluster_labels_cpu = get_center_labels(embedded_data)
    #29,,,1600个
    #输出louvain分类后数据和类别所在的列



    n_clusters_start=len(np.unique(cluster_labels_cpu))
    # 去除其中重复的元素 ，并按元素由小到大返回一个新的无元素重复的元组或者列表
    print("\n "  "Initialize " + str(n_clusters_start) + "  mirco_clusters \n")

    # Get nearest points to optimal centers
    centers_cpu, embedded_centers_cpu = get_nearest_points_to_optimal_centers(X, init_centers, embedded_data)
    #找和中心最近的点
    # Initial dip values
    dip_matrix_cpu = _get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu, n_clusters_start)

    # Reduce learning_rate from pretraining by a magnitude of 10
    dedc_learning_rate = learning_rate * 0.1
    optimizer = optimizer_class(autoencoder.parameters(), lr=dedc_learning_rate)




    # Start clustering training
    cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder = _scclust_training(X, X_raw, sf, n_clusters_start,
                                                                                          dip_merge_threshold,
                                                                                          batch_size,
                                                                                          centers_cpu,
                                                                                          cluster_labels_cpu,
                                                                                          dip_matrix_cpu,
                                                                                          n_clusters_max,
                                                                                          n_clusters_min,
                                                                                          dedc_epochs,
                                                                                          optimizer,
                                                                                          loss_fn,
                                                                                          lambda_1,
                                                                                          lambda_2,
                                                                                         autoencoder,
                                                                                          device,
                                                                                          trainloader,
                                                                                          testloader)
    #(X, n_clusters_current, dip_merge_threshold, centers_cpu,
     #cluster_labels_cpu, dip_matrix_cpu, n_clusters_max, n_clusters_min, dedc_epochs, optimizer,
     #loss_fn, lambda_1, lambda_2, autoencoder, device, trainloader, testloader, debug):

    return cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder



class SCClust():

    def __init__(self, dip_merge_threshold=0.9, batch_size=256,learning_rate=0.001, pretrain_epochs=50, embedding_size=47,dedc_epochs=10, lambda_1=1.0, lambda_2=0.5, n_clusters_max=np.inf, n_clusters_min=3):

        ########alt_lr=0.001, sigma=2.0,denoise=False,

        self.dip_merge_threshold = dip_merge_threshold
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.embedding_size = embedding_size
        self.dedc_epochs = dedc_epochs
        self.lambda_1=lambda_1
        self.lambda_2=lambda_2
        #self.denoise = denoise
        self.n_clusters_max=n_clusters_max
        self.n_clusters_min=n_clusters_min


    def fit(self, X, X_raw, sf):
        labels, n_clusters, centers, autoencoder = _scclust(X, X_raw, sf,
                                                               self.dip_merge_threshold,
                                                               self.batch_size,
                                                               self.learning_rate,
                                                               self.pretrain_epochs,
                                                               self.embedding_size,
                                                               self.dedc_epochs,
                                                               self.lambda_1,
                                                               self.lambda_2,
                                                               self.n_clusters_max,
                                                               self.n_clusters_min)
        # (self, dip_merge_threshold=0.9, batch_size=128,
        #                  learning_rate=1e-4, pretrain_epochs=100, dedc_epochs=50, lambda_1=1.0, lambda_2=0.5 ,
        #                  alt_lr=0.001, denoise=False, sigma=2.0, data_size=10000, n_clusters_max=np.inf,
        #                  n_clusters_min=3, debug=False):


        self.labels_ = labels
        self.n_clusters_ = n_clusters
        self.cluster_centers_ = centers
        self.autoencoder = autoencoder

        return labels, n_clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #ArgumentDefaultsHelpFormatter，打印出了default默认值
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--name', type=str, default='Romanov')
    #Xin_normalized、CITESeq_pbmc_spector_all
    args = parser.parse_args()
    print(args.name)

    # Read_data  cell x gene

    #adata = sc.read_h5ad(args.data_path + "/" + args.name + ".h5ad")
    #adata.obs['size_factors'] = adata.obs.nCount_RNA / np.median(adata.obs.nCount_RNA)
    #print(adata.obs)
    #print("type",type(adata))
    #data = adata.X.A
    #print(data)
    #y=get_center_labels(data)
    #print(y)

    # data=np.transpose(data)

    # data preprocessing
    # data=pro.preprocessing(data)
    # print(data.shape)
    # data=pro.Selecting_highly_variable_genes(data, 2000)
    # print(data.shape)
    # data = np.transpose(data)
    # adata = pro.Selecting_highly_variable_genes(adata, 2000)
    # print(probabillity)


    data_mat = h5py.File(args.data_path + "/" + args.name +  "/" + args.name + ".h5")
    print(data_mat)
    data = np.array(data_mat['X'])
    data=data.T
    print("data",data.shape,data)
    #x2 = np.array(data_mat['X2'])
    y = np.array(data_mat['Y'])
    print(y)
    data_mat.close()

    # 随机森林
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(data, y)
    # feature importance
    importances = model.feature_importances_
    print("importances", importances)
    sorted_indices = np.argsort(importances)[::-1]
    top_indices = sorted_indices[:25000]
    print("top_indices", top_indices, type(top_indices))
    X_selected = data[:, top_indices]
    print("X_selected", X_selected.shape, X_selected)

    # scGeneClust
    adata_org = sc.AnnData(data)

    genes_fast = pro.scGeneClust(adata_org, n_var_clusters=200, version='fast')
    print("genes_fast", genes_fast.shape, genes_fast, type(genes_fast))
    genes_fast_int = genes_fast.astype(int)
    X_geneclust = data[:, genes_fast_int]
    print("X_geneclust", X_geneclust.shape, X_geneclust)

    # 合并过程
    merged_array = np.hstack((X_selected, X_geneclust))

    # 对每一列去重，保留唯一的值
    #unique_cols = np.unique(combined_arr.T, axis=0)

    # 转置回正常的形状
    #merged_array = unique_cols.T

    print("合并后的数组形状:", merged_array.shape, merged_array)

    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(merged_array)
    print("adata.obs", adata.obs)
    adata.obs['Group'] = y


    adata = pro.read_dataset(adata,
                          transpose=False,
                          test_split=False,
                          copy=True)

    adata.X = adata.X.astype(float)
    adata = pro.normalize(adata,
                       size_factors=True,
                       normalize_input=True,
                       logtrans_input=True)
    print(type(adata.X))
    print("adata.X",adata.X)



    if 'CellType' in adata.obs:
        if type(adata.obs['CellType'].values[0]) == str:
            labels=adata.obs['CellType'].astype('category').cat.codes.astype('long').values
            #astype()函数可将dateframe某一列的str类型转为int; category是一种特殊的数据类型，表示一个类别，比如性别，血型，分类，级别等等。
            #cat.codes实现对整数的映射
            #.values只保留数值
            print("labels1",labels)
        else:
            labels=adata.obs['CellType'].values.astype(np.int32)
            print("labels2", labels)
    else:
        labels=y

    # Training

    cluster_labels, estimated_cluster_numbers = SCClust().fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors.values,)
    #.A
    print(cluster_labels)

    # === Print results ===
    print("The estimated number of clusters:", estimated_cluster_numbers)

    #if 'CellType' in adata.obs:
    print("ARI: ", ari_score(labels, cluster_labels))
    print("NMI:", nmi(labels, cluster_labels))
    print("CA: ", cluster_acc(labels, cluster_labels))

    if not os.path.exists("output"):
        # os.path.exists()判断括号里的文件是否存在的意思，括号内的可以是文件路径
        os.mkdir("output")
        # os.mkdir()函数创建目录
    pd.DataFrame({"cluster_labels": cluster_labels}, index=list(adata.obs_names)).to_csv(
        "output/"  + args.name + "_predssv")
    # index即行标签
    # DataFrame文件导出成.csv文件




