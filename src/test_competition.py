# coding=utf-8
from memory_profiler import memory_usage, profile
import psutil


@profile
def test():
    for i in range(1, 10000):
        print(i)


if __name__ == "__main__":
    test()
