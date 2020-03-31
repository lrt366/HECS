# -*- coding: utf-8 -*-
import tensorflow as tf


def eval_metric(qids, cids, preds, labels):

    pre_dict = {}
    dic = {}

    # 首先把相同qid的用字典的方式聚集起来~
    for qid, cid, pred, label in zip(qids, cids, preds, labels):
        #字典中包含若给定键，则返回该键对应的值，否则返回设置的值
        pre_dict.setdefault(qid, [])
        pre_dict[qid].append([cid, pred, label])

    #查找键值
    for qid in pre_dict.keys():
        # 按照pred排序 对list排序,已经对分数排名
        dic[qid] = sorted(pre_dict[qid], key=lambda x: x[1], reverse=True)
        # 得到rank
        cid2rank = {cid: [label, rank] for (rank, (cid, pred, label)) in enumerate(dic[qid])}
        #  排名
        dic[qid] = cid2rank

        #{’Q0':{'C00':[0,0],'C01':[0,1],'C02':[1,2],'C03':[0,3]}....,’Q1':}

    recall_2 = 0.0
    mrr = 0.0
    dic_length = 0



    for qid in dic.keys():
        #判断字典的长度
        dic_length += 1
        # 按dic[qid]里面的 rank排序 ,从小到大（label,rank) C02是正确样本
        sort_rank = sorted(dic[qid].items(), key=lambda x:x[1][1], reverse=False)
        #[('C00', [0, 0]), ('C01', [0, 1]), ('C02', [1, 2]), ('C03', [0, 3])]

        #循环遍历最佳候选位置
        for i in range(len(sort_rank)):

            if sort_rank[i][1][0] == 1:
                # 计算 MRR值
                mrr += 1.0/float(i+1)

                if i <= 1:
                    # 计算recall@5的值
                    recall_2 += 1.0
    import itertools
    list2d = [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
    merged = list(itertools.chain(*list2d))

    recall_2/= dic_length

    mrr /= dic_length

    ave_frank /= dic_length


    return recall_2, mrr

def eval_metric_rec1(qids, cids, preds, labels):

    pre_dict = {}
    dic = {}

    # 首先把相同qid的用字典的方式聚集起来~
    for qid, cid, pred, label in zip(qids, cids, preds, labels):
        #字典中包含若给定键，则返回该键对应的值，否则返回设置的值
        pre_dict.setdefault(qid, [])
        pre_dict[qid].append([cid, pred, label])

    #查找键值
    for qid in pre_dict.keys():
        # 按照pred排序 对list排序,已经对分数排名
        dic[qid] = sorted(pre_dict[qid], key=lambda x: x[1], reverse=True)
        # 得到rank
        cid2rank = {cid: [label, rank] for (rank, (cid, pred, label)) in enumerate(dic[qid])}
        #  排名
        dic[qid] = cid2rank

        #{’Q0':{'C00':[0,0],'C01':[0,1],'C02':[1,2],'C03':[0,3]}....,’Q1':}

    recall_1 = 0.0
    mrr = 0.0
    dic_length = 0

    for qid in dic.keys():
        #判断字典的长度
        dic_length += 1
        # 按dic[qid]里面的 rank排序 ,从小到大（label,rank) C02是正确样本
        sort_rank = sorted(dic[qid].items(), key=lambda x:x[1][1], reverse=False)
        #[('C00', [0, 0]), ('C01', [0, 1]), ('C02', [1, 2]), ('C03', [0, 3])]

        #循环遍历最佳候选位置
        for i in range(len(sort_rank)):

            if sort_rank[i][1][0] == 1:
                # 计算 MRR值
                mrr += 1.0/float(i+1)
                print('mrr'+str(qid))
                print('mrr' + str(mrr))

                if i == 0:
                    # 计算recall@1的值
                    recall_1 += 1.0
                    print('recall' + str(qid))
                    print('mrr' + str(recall_1))

    recall_1/= dic_length

    mrr /= dic_length


    return recall_1, mrr

def count_parameters():

    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_params = 1
        for dim in shape:
            variable_params *= int(dim)
        total_params += variable_params

    return total_params


