# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def cal_Auc(test_path, result_path):
    # y_true = [0, 0, 1, 1]
    # y_scores = [0.1, 0.4, 0.35, 0.8]
    df_test = pd.read_csv(test_path, encoding='utf-8', header=-1)
    df_result = pd.read_csv(result_path, encoding='utf-8', header=-1)
    query_num = list(set(df_test.iloc[:, 0]))
    # print(len(query_num))
    query_total = len(query_num)
    auc_sum = 0
    # columns = df_test.columns.values.tolist()
    # print(columns)
    # print(df_test[df_test.iloc[:, 0]==2717])
    for query in query_num:
        one_query_test = list(df_test[df_test.iloc[:, 0] == query].iloc[:, -1])
        one_query_result = list(df_result[df_test.iloc[:, 0] == query].iloc[:, -1])
        try:
            auc_sum += roc_auc_score(one_query_test, one_query_result)
            # print(roc_auc_score(one_query_test, one_query_result))
        except:
            auc_sum += 0.5

    score = auc_sum / query_total
    print(score)


if __name__ == "__main__":
    cal_Auc("/Users/tianyu/PycharmProjects/competition/res/sample_test.csv",
            "/Users/tianyu/PycharmProjects/competition/res/result_lr.csv")
