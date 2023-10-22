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


def task_2():
    import random
    random.seed(20)

    def generate_marked_dots(n=20):
        dots = [[round(random.random(), 5), round(random.random(), 5)] for i in range(n)]
        for dot in dots:
            if dot[0] < dot[1]:
                dot.append(0)
            if dot[0] > dot[1]:
                dot.append(1)
        return dots

    def act_adaline(x):
        if x < 0:
            return -1
        if x > 0:
            return 1
    def df_relu(x):
        return 0 if x < 0 else 1

    def act_perceptron(x):

    def df_sig(x):
        return 0.5 * (1 + x) * (1 - x)

    def go_forward_adaline(input_vectors, W):
        sum = np.dot(W, input_vectors)
        y = relu(sum)
        return y

    def go_forward_perceptron(input_vectors, W):
        sum = np.dot(W, input_vectors)
        y = relu(sum)
        return y

    def train(epoch, W):
        lmd = 0.01
        N = 1000
        count = len(epoch)
        for k in range(N):
            x = epoch[np.random.randint(0, count)]
            y = go_forward(x[:2], W)
            e = y - x[-1]
            delta = e * df_relu(y)
            W = W - lmd * delta * np.array(x[:2])

    W = np.array([round(np.random.uniform(-1, 1), 2), round(np.random.uniform(-1, 1), 2)])
    train_marked_dots = generate_marked_dots()
    train(train_marked_dots, W)

    for x in train_marked_dots:
        y = go_forward(x[:2], W)
        print(f"Выходное значение НС: {y} => {x[-1]}")


def task_3():
    def relu(x):
        return np.maximum(x, 0.0)

    def df_relu(x):
        return 0 if x < 0 else 1

    def sig(x):
        return 2 / (1 + np.exp(-x)) - 1

    def df_sig(x):
        return 0.5 * (1 + x) * (1 - x)

    def go_forward(input_vectors, W):
        sum = np.dot(W, input_vectors)
        y = relu(sum)
        return y

    def train(epoch, W):
        lmd = 0.01
        N = 1000
        count = len(epoch)
        for k in range(N):
            x = epoch[np.random.randint(0, count)]
            y = go_forward(x[:2], W)
            e = y - x[-1]
            delta = e * df_relu(y)
            W = W - lmd * delta * np.array(x[:2])

    W = np.array([round(np.random.uniform(-1, 1), 2), round(np.random.uniform(-1, 1), 2)])
    train_marked_dots = generate_marked_dots()
    train(train_marked_dots, W)

    for x in train_marked_dots:
        y = go_forward(x[:2], W)
        print(f"Выходное значение НС: {y} => {x[-1]}")


if __name__ == '__main__':
    # task_1()
    task_2()
