import numpy as np
import matplotlib.pyplot as plt
import random


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
    def generate_marked_dots(n=20):
        dots = [[round(random.random(), 5), round(random.random(), 5)] for i in range(n)]
        for dot in dots:
            if dot[0] < dot[1]:
                dot.append(0)
            if dot[0] > dot[1]:
                dot.append(1)
        return dots

    def act_adaline(u):
        if u < 0:
            return -1
        if u > 0:
            return 1

    def act_perceptron(u):
        if u < 0:
            return 0
        if u > 0:
            return 1

    def go_forward_adaline(input_vectors, W):
        u = np.dot(input_vectors, W)
        y = act_adaline(u)
        return y

    def go_forward_perceptron(input_vectors, W):
        u = np.dot(input_vectors, W)
        y = act_perceptron(u)
        return y

    def clarification_weights_perceptron(y, x, W):
        if y == 0 and x[-1] == 1:
            W = W + x[:2]
        if y == 1 and x[-1] == 0:
            W = W - x[:2]
        return W

    def clarification_weights_adaline(y, x, W):
        nu = np.float64(0.5)
        e = x[-1] - np.dot(x[:2], W)
        delta = np.dot(nu * e, x[:2])
        W = W + delta
        return W

    def train(epoch, W, go_forward_func, clarification_weights_func):
        count = len(epoch)
        for k in range(count):
            x = epoch[k]
            y = go_forward_func(x[:2], W)
            W = clarification_weights_func(y, x, W)
        return W

    random.seed(127)
    W1 = np.array([round(np.random.uniform(-1, 1), 2),
                   round(np.random.uniform(-1, 1), 2)])
    W2 = W1.copy()

    train_marked_dots = generate_marked_dots()

    W1 = train(train_marked_dots,
               W1,
               go_forward_perceptron,
               clarification_weights_perceptron)

    W2 = train(train_marked_dots,
               W2,
               go_forward_adaline,
               clarification_weights_adaline)

    test_marked_dots = generate_marked_dots(1000)

    mistakes = 0
    print("Персептрон")
    for x in test_marked_dots:
        y = go_forward_perceptron(x[:2], W1)
        if y != x[-1]:
            mistakes += 1

    print(f"Всего предсказаний => {len(test_marked_dots)}")
    print(f"Количество верных предсказаний => {len(test_marked_dots) - mistakes}")
    print(f"Количество ошибок => {mistakes}")
    print(f"Процент правильных ответов => {(len(test_marked_dots) - mistakes) / len(test_marked_dots)}")

    print("-" * 100)
    print("Адалайн")
    mistakes = 0
    for x in test_marked_dots:
        y = go_forward_adaline(x[:2], W2)
        if (y == -1 and x[-1] == 1) or (y == 1 and x[-1] == 0):
            mistakes += 1

    print(f"Всего предсказаний => {len(test_marked_dots)}")
    print(f"Количество верных предсказаний => {len(test_marked_dots) - mistakes}")
    print(f"Количество ошибок => {mistakes}")
    print(f"Процент правильных ответов => {(len(test_marked_dots) - mistakes) / len(test_marked_dots)}")


def task_3():
    pass

# train_marked_dots = generate_marked_dots()
#  train(train_marked_dots, W)
#
#  for x in train_marked_dots:
#      y = go_forward(x[:2], W)
#      print(f"Выходное значение НС: {y} => {x[-1]}")


if __name__ == '__main__':
    # task_1()
    #task_2()
    task_3()
