from enum import Enum
import numpy as np


class Letter(Enum):
    X = [0, 0]
    Y = [0, 1]
    I = [1, 0]
    L = [1, 1]


input_vectors = [[[1, 0, 1,
                   0, 1, 0,
                   1, 0, 1], Letter.X],
                 [[1, 0, 1,
                   0, 0, 0,
                   1, 0, 1], Letter.X],
                 [[0, 0, 1,
                   0, 1, 0,
                   1, 0, 1], Letter.X],
                 [[1, 0, 0,
                   0, 1, 0,
                   1, 0, 1], Letter.X],
                 [[1, 0, 1,
                   0, 1, 0,
                   0, 1, 0], Letter.Y],
                 [[0, 0, 1,
                   0, 1, 0,
                   0, 1, 0], Letter.Y],
                 [[1, 0, 0,
                   0, 1, 0,
                   0, 1, 0], Letter.Y],
                 [[1, 0, 1,
                   0, 1, 0,
                   0, 0, 0], Letter.Y],
                 [[0, 1, 0,
                   0, 1, 0,
                   0, 1, 0], Letter.I],
                 [[0, 0, 0,
                   0, 1, 0,
                   0, 1, 0], Letter.I],
                 [[0, 1, 0,
                   0, 0, 0,
                   0, 1, 0], Letter.I],
                 [[0, 1, 0,
                   0, 1, 0,
                   0, 0, 0], Letter.I],
                 [[1, 0, 0,
                   1, 0, 0,
                   1, 1, 1], Letter.L],
                 [[0, 0, 0,
                   1, 0, 0,
                   1, 1, 1], Letter.L],
                 [[1, 0, 0,
                   0, 0, 0,
                   1, 1, 1], Letter.L],
                 [[1, 0, 0,
                   1, 0, 0,
                   0, 1, 1], Letter.L],
                 [[1, 0, 0,
                   1, 0, 0,
                   1, 0, 1], Letter.L],
                 [[1, 0, 0,
                   1, 0, 0,
                   1, 1, 0], Letter.L]
                 ]

W1 = np.array([
    [round(np.random.uniform(-1, 1), 2) for _ in range(9)],
    [round(np.random.uniform(-1, 1), 2) for _ in range(9)],
    [round(np.random.uniform(-1, 1), 2) for _ in range(9)],
])

W2 = np.array([
    [round(np.random.uniform(-1, 1), 2) for _ in range(3)],
    [round(np.random.uniform(-1, 1), 2) for _ in range(3)],
])

W1 = np.append(W1, np.array([[1], [1], [1]]), axis=1)
W2 = np.append(W2, np.array([[1], [1]]), axis=1)


def relu(x):
    return np.maximum(x, 0.0)


def df_relu(x):
    return 0 if x < 0 else 1


def go_forward(inp):
    sum = np.dot(W1, inp)
    out = np.array([relu(x) for x in sum])

    out = np.append(out, 1)
    sum = np.dot(W2, out)
    y = np.array([relu(x) for x in sum])
    return y, out


def train(epoch):
    lmd = 0.01
    N = 10_000
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]
        y, out = go_forward(x[0])
        e = y - x[-1].value
        delta = e * [df_relu(yi) for yi in y]

        W2[0] = W2[0] - lmd * delta[0] * out
        W2[1] = W2[1] - lmd * delta[1] * out

        delta2 = np.array([delta[0] * W2[0, 0] + delta[1] * W2[1, 0],
                           delta[0] * W2[0, 1] + delta[1] * W2[1, 1],
                           delta[0] * W2[0, 2] + delta[1] * W2[1, 2]
                           ])

        W1[0, :] = W1[0, :] - lmd * delta2[0] * np.array(x[0])
        W1[1, :] = W1[1, :] - lmd * delta2[1] * np.array(x[0])
        W1[2, :] = W1[2, :] - lmd * delta2[2] * np.array(x[0])


if __name__ == '__main__':
    for x in input_vectors:
        x[0].append(1)

    train(input_vectors)

    for x in input_vectors:
        y, out = go_forward(x[0])
        e = y - x[-1].value
        print("-" * 100)
        print(f"Буква: {x[-1].name} {x[-1].value}")
        print(f'Входной массив: {x[0]}')
        print(f'Выходное значение НС: {y}')
        print(f'Ошибка: {e}')
