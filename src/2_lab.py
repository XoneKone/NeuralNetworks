import numpy as np
import matplotlib.pyplot as plt


def task_1():
    def relu(x):
        return np.maximum(x, 0)

    def go(C):
        x = np.array([C[0], C[1], 1])
        w1 = [1, 1, 0]
        w2 = [1, 1, -1]
        w_hidden = np.array([w1, w2])
        w_out = np.array([1, -2, 0])

        sum = np.dot(w_hidden, x)
        out = [relu(x) for x in sum]
        out.append(1)
        out = np.array(out)

        sum = np.dot(w_out, out)
        y = relu(sum)

        return y

    C1 = [(1, 0), (0, 1)]
    C2 = [(0, 0), (1, 1)]

    print(f'(0, 0) = {go(C2[0])}')
    print(f'(1, 0) = {go(C1[0])}')
    print(f'(0, 1) = {go(C1[1])}')
    print(f'(1, 1) = {go(C2[1])}')


if __name__ == '__main__':
    task_1()
