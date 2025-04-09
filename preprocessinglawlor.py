import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
import scipy as sp
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import warnings
import numpy.matlib
import cvxpy as cp
import scipy.sparse as ss
#import SNMF_inner as SI
import matplotlib.pyplot as plt
import os
from typing import Literal, Optional, Union, Tuple
import time

import anndata as ad
import numpy as np
from loguru import logger
from scipy.sparse import issparse

import scGeneClust.pp as pp
import scGeneClust.tl as tl
from util import set_logger
from _validation import check_args, check_all_genes_selected
import time

import numpy as np
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix

import triku as tk




eps = 2.22044604925031e-16

def scGeneClust(
        raw_adata: ad.AnnData,
        image: np.ndarray = None,
        n_var_clusters: int = None,
        n_obs_clusters: int = None,
        n_components: int = 10,
        relevant_gene_pct: int = 20,
        post_hoc_filtering: bool = True,
        version: Literal['fast', 'ps'] = 'fast',
        modality: Literal['sc', 'st'] = 'sc',
        shape: Literal['hexagon', 'square'] = 'hexagon',
        return_info: bool = False,
        subset: bool = False,
        max_workers: int = os.cpu_count() - 1,
        verbosity: Literal[0, 1, 2] = 1,
        random_state: int = 0
) -> Optional[Union[Tuple[ad.AnnData, np.ndarray], np.ndarray]]:
    # check arguments
    check_args(raw_adata, image, version, n_var_clusters, n_obs_clusters, n_components, relevant_gene_pct,
               post_hoc_filtering, modality, shape, return_info, subset, max_workers, verbosity, random_state)
    # set log level
    set_logger(verbosity)
    # feature selection starts
    logger.opt(colors=True).info(
        f"Performing <magenta>GeneClust-{version}</magenta> "
        f"on <magenta>{'scRNA-seq' if modality == 'sc' else 'SRT'}</magenta> data, "
        f"with <yellow>{max_workers}</yellow> workers."
    )
    copied_adata = raw_adata.copy()
    copied_adata.X = raw_adata.X.toarray() if issparse(raw_adata.X) else raw_adata.X

    # preprocessing
    pp.normalize(copied_adata, modality)
    pp.reduce_dim(copied_adata, version, random_state)
    # gene clustering
    tl.cluster_genes(copied_adata, image, version, modality, shape, n_var_clusters, n_obs_clusters, n_components,
                     relevant_gene_pct, max_workers, random_state)
    # select features from gene clusters
    selected_genes = tl.select_from_clusters(copied_adata, version, post_hoc_filtering, random_state)
    check_all_genes_selected(raw_adata, selected_genes)

    if subset:
        raw_adata._inplace_subset_var(selected_genes)
        logger.opt(colors=True).info(f"<magenta>GeneClust-{version}</magenta> finished.")
        return None

    logger.opt(colors=True).info(f"GeneClust-{version} finished.")
    if return_info:
        return copied_adata, selected_genes
    else:
        return selected_genes


def read_data(filename, data_type):

    if data_type == '10X':
        y = pd.read_csv("./datasets/10X/label.csv",index_col=0,header=0,sep=',')
        data = mmread("./datasets/10X/matrix.mtx")
        a = data.todense()  #.todense()将稀疏矩阵转为稠密矩阵
        a = a.transpose()  #transpose()函数的作用就是调换数组的行列值的索引值，类似于求矩阵的转置
        X = np.array(a).astype('float32')
    if data_type == 'csv':
        data_path = filename + "/data.csv"
        label_path = filename + "/label.csv"
        X = pd.read_csv(data_path,index_col=0,header=None,sep=',',encoding='utf-8', engine='c')
        y = pd.read_csv(label_path,index_col=0,header=0, sep=',',encoding='utf-8', engine='c')
    return X, y


#def read_data(data_path): #scSO
    #Initial_Data = pd.read_csv(filepath_or_buffer=data_path,
                                        #sep=',',
                                        #header=0,
                                        #index_col=0)
# 读取数据
    #geneName = Initial_Data.index
# 读取基因名称（行名）
    #if np.size(geneName) > np.size(np.unique(geneName)):
        # 判断行名是否重复
           #geneName = np.unique(geneName)
        # 将重复的行名去掉，并按从小到大的顺序排列
           #Initial_Data = Initial_Data.groupby(level=0).sum()
        # 将Initial_Data中按不同的行名分组，重复的行名求和
    #Data = Initial_Data.values
    #return Initial_Data



#def filter_genes(Initial_Data,bound_low=0.1, bound_up=8.5): #scCO
    #A_ing = Initial_Data.copy()  # copy()深层复制  #90*20214
    #A_ing[A_ing > 0] = 1
    #A_ing[A_ing <= 0] = 0  # 把A_ing中的正值如：2/3/4变为1，负值和0变为0
    #Row_mean = np.mean(A_ing, axis=1)  # 对每个基因（每行)取均值  #90*1
    #Data = Initial_Data[(Row_mean > 0) &
                                    #(Row_mean < 1), :].copy()
    # 在矩阵中去掉所有表达都为负值和0和大于等于1的基因（行）  #82*20214
    # 其实就是去掉了普遍存在的基因和罕见的基因
    #Row_mean = np.mean(Data, axis=1)  # 对每个基因（每行)取均值  #82*1
    #rho = np.mean(Row_mean)  # 对所有值取均值  1*1
    #Data = Data[(Row_mean > bound_low * rho) &
                          #(Row_mean < bound_up * rho), :]  # 70*20214
    # 留下0.1*总均值<表达均值<8.5*总均值的基因   #公式
    #return Data

def LogNormalize(Data,scale_factor=10000, b_log1p=True):  #scCO
    Data = np.dot(Data,
                  np.diag(scale_factor / np.sum(Data, axis=0)))
    # np.sum(self.Data, axis=0)求每列的和，即每个细胞的和
    # numpy.diag()函数是以一维数组的形式返回方阵的对角线（或非对角线）元素，或将一维数组转换成方阵（非对角线元素为0）
    Data = np.log10(Data + 1)
    return Data

def preprocessing(Data): #scCAEs
    #Data=filter_genes(data1)
    data=LogNormalize(Data, scale_factor=10000, b_log1p=True)
    return data


def Selecting_highly_variable_genes(adata, highly_genes): #scCAEs
    adata.var_names_make_unique()   #通过将数字字符串追加到每个重复的索引元素，使索引唯一： “1”、“2”等
    # sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)   #根据细胞数量或计数过滤基因
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)   #对每个细胞的count进行标准化，使得每个细胞标准化后有相同的count
    sc.pp.log1p(adata)  #对数化
    adata.raw = adata   #储存X和Var的原始版本
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2000)
    #该函数用于确定高变基因；高变异基因就是highly variable features（HVGs），就是在细胞与细胞间进行比较，选择表达量差别最大的基因
    adata = adata[:, adata.var['highly_variable']].copy()
    # sc.pp.scale(adata, max_value=3)
    return adata


def read_dataset(adata, transpose=False, test_split=False, copy=False):
    if isinstance(adata, sc.AnnData):  # isinstance判断一个对象是否是一个已知的类型
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    # assert X Y，如果X为true,则程序正常运行，否则报错。

    # 判断adata是否为anndata类型以及adata.obs中有'n_count'

    #if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
        #if sp.sparse.issparse(adata.X):
            # 判断x是否为稀疏数组或稀疏矩阵类型
            #assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
            # nnz(X)返回矩阵X中的非零元素的数目
            # 判断adata.X是否为整数类型

        #else:
            #assert np.all(adata.X.astype(int) == adata.X), norm_error
            # all()函数用于判断整个数组中的元素的值是否全部满足条件

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))
    # 49*3762

    return adata


def clr_normalize_each_cell(adata):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    adata.raw = adata.copy()
    sc.pp.normalize_per_cell(adata)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.raw.X.A if scipy.sparse.issparse(adata.raw.X) else adata.raw.X)
    )
    return adata


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)  # 过滤基因  该函数用于保留在至少 min_cells 个细胞中出现的基因，或者保留在至多 max_cells 个细胞中出现的基因；
        sc.pp.filter_cells(adata, min_counts=1)  # 过滤细胞  该函数保留至少有 min_genes 个基因（某个基因表达非0可判断存在该基因）的细胞，或者保留至多有 max_genes 个基因的细胞；

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        print("type", type(adata.X))
        adata.X = adata.X.astype(float)
        sc.pp.normalize_per_cell(adata)  # 归一化操作
        # 通过所有基因的总计数对每个细胞进行归一化，使每个细胞归一化后的总数相同
        print("type", type(adata.X))
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        #adata.obs.n_counts 未标准化前的行和，即细胞基因表达的UMI求和，只要通过sc.pp.normalize_per_cell处理就会出现这个属性
        #print("type", type(adata.X))
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)  # 对数化

    if normalize_input:
        sc.pp.scale(adata)  # 将数据归一化到mean=0，var=1

    return adata


def feature_selection(data,adata, adata_raw,y,n_RandomForest):

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X=adata.X
    print("X",X.shape,X)
    print("y",type(y),y)


    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y)
    print(clf.feature_importances_)

    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print(X_new.shape)


    # 随机森林
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X, y)
    # feature importance
    importances = model.feature_importances_
    print("importances",importances)
    sorted_indices = np.argsort(importances)[::-1]
    top_indices = sorted_indices[:n_RandomForest]
    print("top_indices",top_indices,type(top_indices))
    X_selected = X[:, top_indices]
    print("X_selected",X_selected.shape,X_selected)

    #scGeneClust
    adata_org=sc.AnnData(data)

    genes_fast = scGeneClust(adata_org, n_var_clusters=200, version='fast')
    print("genes_fast",genes_fast.shape,genes_fast,type(genes_fast))
    genes_fast_int = genes_fast.astype(int)
    X_geneclust=X[:,genes_fast_int]
    print("X_geneclust",X_geneclust.shape, X_geneclust)



    #合并过程
    combined_arr = np.hstack((X_selected, X_geneclust))

    # 对每一列去重，保留唯一的值
    unique_cols = np.unique(combined_arr.T, axis=0)

    # 转置回正常的形状
    merged_array = unique_cols.T

    print("合并后的数组形状:", merged_array.shape, merged_array)

    #
    sc.pp.neighbors(adata)
    adata = adata.copy()


    tk.triku(adata)
    print("adata",adata.var["highly_variable"],adata.var["highly_variable"].shape)
    highly_gene= adata.var["highly_variable"]
    print(highly_gene)
    X_triku=adata.X[:,highly_gene]
    print("X_triku",X_triku.shape,X_triku)


    return merged_array










