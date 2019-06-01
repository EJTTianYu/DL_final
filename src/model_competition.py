# coding=utf-8

from memory_profiler import memory_usage, profile
import time
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import Levenshtein
import pandas as pd
from sklearn.externals import joblib


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable


# 用于填写模型训练代码,深度学习
@profile
def train_model_LR(train_path, test_path, result_path):
    with open(train_path, "r") as train, open(test_path, "r") as test, open(result_path, "w") as result:
        train_data = csv.reader(train)
        test_data = csv.reader(test)
        wirter = csv.writer(result)
        # random模型以及全0模型尝试
        # wirter.writerow("1")
        # for row_num, row in enumerate(test_data):
        #     # print("%s,%s,%s" % (row[0], row[2], random.random()))
        #     process_data = []
        #     process_data.append(row[0])
        #     process_data.append(row[2])
        #     # process_data.append(random.random())
        #     process_data.append(0)
        #     wirter.writerow(process_data)
        # query_index = 1
        query_index = 2548
        query_title_list = []
        for row_num, row in enumerate(test_data):
            if int(row[0]) != query_index:
                process_data(query_title_list)

                query_title_list = []
                query_title_list.append(row[1])
                query_title_list.append(row[3])
                query_index = int(row[0])

            else:
                if row_num == 0:
                    query_title_list.append(row[1])
                query_title_list.append(row[3])


def process_data(query_title_list):
    with open(r"/Users/tianyu/PycharmProjects/competition/res/median1.csv", 'a') as median:
        writer = csv.writer(median)
        tfidf2 = TfidfVectorizer(max_features=100)
        features = tfidf2.fit_transform(query_title_list)
        features = features.todense()
        features_numpy = np.array(features)
        # print(features.shape)
        for i in range(1, features.shape[0]):
            row = []
            dist1 = np.linalg.norm(features[0] - features[i])
            dist2 = (cosine_similarity(features[0], features[i])[0]).tolist()[0]
            per_div = len(query_title_list[0]) / len(query_title_list[i])
            dist3 = Levenshtein.distance(query_title_list[0], query_title_list[i])
            # product = features[0] * features[i].T
            row.append(dist1)
            row.append(dist2)
            row.append(per_div)
            row.append(dist3)
            # row.append(product)
            # print(row)
            writer.writerow(row)


def concat(train_path, median_path, result_path):
    with open(train_path, "r") as train, open(median_path, "r") as median, open(result_path, "w") as result:
        train_read = csv.reader(train)
        median_read = csv.reader(median)
        writer = csv.writer(result)
        for (row, line) in zip(train_read, median_read):
            line.append(row[-1])
            writer.writerow(line)


def model_LR():
    train_data = pd.read_csv('/Users/tianyu/PycharmProjects/competition/res/median_final.csv', lineterminator='\n',
                             header=-1)
    # test_data = pd.read_csv('/Users/tianyu/PycharmProjects/competition/res/sample_test.csv', lineterminator='\n',
    #                         header=-1)
    lr = LogisticRegression()
    lr.fit(train_data.iloc[:, 0:4], train_data.iloc[:, 4:5])
    joblib.dump(lr, "train_model.m")


# mode
def predict_lr():
    lr = joblib.load("train_model.m")
    train_data = pd.read_csv('/Users/tianyu/PycharmProjects/competition/res/median1.csv', lineterminator='\n',
                             header=-1)
    # test_data = pd.read_csv('/Users/tianyu/PycharmProjects/competition/res/sample_test.csv', lineterminator='\n',
    #                         header=-1)
    lr_y_predit = lr.predict_proba(train_data.iloc[:, 0:4])
    # output = pd.DataFrame(data={test_data["ID"].tolist(), "Pred": pred_list})
    with open('/Users/tianyu/PycharmProjects/competition/res/sample_test.csv', "r") as test, open(
            "/Users/tianyu/PycharmProjects/competition/res/result_lr.csv", "w") as output:
        data_test = csv.reader(test)
        writer = csv.writer(output)
        for data_test, y_predict in zip(data_test, lr_y_predit):
            line = []
            line.append(data_test[0])
            line.append(data_test[2])
            line.append(y_predict[1])
            writer.writerow(line)


if __name__ == "__main__":
    time_before = time.time()
    # train_model_LR("/Users/tianyu/PycharmProjects/competition/res/sample_train.csv",
    #                "/Users/tianyu/PycharmProjects/competition/res/sample_test.csv",
    #                "/Users/tianyu/PycharmProjects/competition/res/result.csv")
    # concat("/Users/tianyu/PycharmProjects/competition/res/sample_train.csv",
    #        "/Users/tianyu/PycharmProjects/competition/res/median.csv",
    #        "/Users/tianyu/PycharmProjects/competition/res/median_final.csv")
    # model_LR()
    predict_lr()
    time_aft = time.time()
    print(time_aft - time_before)
