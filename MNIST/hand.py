import load_data
import math


def mula(arr1: list, arr2: list):
    if len(arr1[0]) != len(arr2):
        raise Exception("Can't mula!")
    ret = []
    for i in range(len(arr1)):
        ret.append([])
        for j in range(len(arr2[0])):
            ret[i].append(0)
            for k in range(len(arr2)):
                ret[i][j] += arr1[i][k] * arr2[k][j]
    return ret


def op(arr1: list, arr2: list, op='+'):
    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
        raise Exception("Can't op" + op)
    ret = arr1.copy()
    for i in range(len(arr1)):
        for j in range(len(arr1[0])):
            ret[i][j] = eval(str(arr1[i][j]) + ' ' +
                             op + ' ' + str(arr2[i][j]))
    return ret


def traversal(arr: list, func):
    return list(map(lambda x: list(map(func, x)), arr))


def modle(data: list, label: list, hidden: tuple, e_num: int, lr: float):
    weights = tuple([] for _ in range(len(hidden)))
    bais = tuple([] for _ in range(len(hidden)))
    for i in range(len(hidden)):
        for j in range(load_data.height * load_data.width):
            weights[i].append([1] * len(hidden))
            bais[i].append([0.1] * len(hidden))
    for i in range(e_num):
        y = data
        for j in range(len(weights)):
            y = traversal(
                op(mula(y, weights[j]), bais[j]), lambda x: 1 / (1 + math.e**(-x)))
        # softmax
        sum_e = list(map(lambda x: sum(map(lambda i: math.e**i)), y))
        for i in range(len(sum_e)):
            for j in range(len(y[0])):
                y[i][j] = math.e**y[i][j] / sum_e[i]
        # Cross-entropy loss


print(mula([[1], [2]], [[3, 4]]))
print(mula([[1, 0, -1], [2, 1, 0], [3, 2, -1]], [[1, 0], [3, 1], [0, 2]]))
# print(op([[1, 0, -1], [2, 1, 0], [3, 2, -1]], [[1, 0], [3, 1], [0, 2]]))
print(traversal([[1, 0, -1], [2, 1, 0], [3, 2, -1]], lambda x: x + 1))
