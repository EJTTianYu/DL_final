# coding=utf-8

import csv


def sample_div(input_path, output1_path, output2_path):
    with open(input_path, "r") as input, open(output1_path, "w") as output1, open(output2_path, "w") as output2:
        input_data = csv.reader(input)
        writer = csv.writer(output1)
        writer2 = csv.writer(output2)
        for row_num, row in enumerate(input_data):
            if row_num < 17000:
                writer.writerow(row)
            else:
                writer2.writerow(row)


if __name__ == '__main__':
    sample_div("/Users/tianyu/PycharmProjects/competition/res/train_data.sample",
               "/Users/tianyu/PycharmProjects/competition/res/sample_train.csv",
               "/Users/tianyu/PycharmProjects/competition/res/sample_test.csv")
