import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string


def loadData(embedded_data):
    f = embedded_data
    vector_dict = {}
    edge_dict = {}
    for line in f.readlines():
        lines = line.strip().split("   ")
        for i in range(2):
            if lines[i] not in vector_dict:
                vector_dict[lines[i]] = int(lines[i])
                edge_list = []
                if len(lines) == 3:
                    edge_list.append(lines[1 - i] + ":" + lines[2])
                else:
                    edge_list.append(lines[1 - i] + ":" + "1")
                edge_dict[lines[i]] = edge_list
            else:
                edge_list = edge_dict[lines[i]]
                if len(lines) == 3:
                    edge_list.append(lines[1 - i] + ":" + lines[2])
                else:
                    edge_list.append(lines[1 - i] + ":" + "1")
                edge_dict[lines[i]] = edge_list
    return vector_dict, edge_dict


def get_max_community_label(vector_dict, adjacency_node_list):
    label_dict = {}
    for node in adjacency_node_list:
        node_id_weight = node.strip().split(":")
        node_id = node_id_weight[0]
        node_weight = int(node_id_weight[1])

        # 按照label为group维度，统计每个label的weight累加和
        if vector_dict[node_id] not in label_dict:
            label_dict[vector_dict[node_id]] = node_weight
        else:
            label_dict[vector_dict[node_id]] += node_weight

    sort_list = sorted(label_dict.items(), key=lambda d: d[1], reverse=True)
    return sort_list[0][0]


def check(vector_dict, edge_dict):
    for node in vector_dict.keys():
        adjacency_node_list = edge_dict[node]   # 获取该节点的邻居节点
        node_label = vector_dict[node]          # 获取该节点当前label
        label = get_max_community_label(vector_dict, adjacency_node_list)   # 从邻居节点列表中选择weight累加和最大的label
        if node_label >= label:
            continue
        else:
            return 0    #  找到weight权重累加和更大的label
    return 1


def label_propagation(vector_dict, edge_dict):
    t = 0
    print('First Label: ')
    while True:
        if (check(vector_dict, edge_dict) == 0):
            t = t + 1
            print('iteration: ', t)
            # 每轮迭代都更新一遍所有节点的社区label
            for node in vector_dict.keys():
                adjacency_node_list = edge_dict[node]
                vector_dict[node] = get_max_community_label(vector_dict, adjacency_node_list)
        else:
            break
    return vector_dict