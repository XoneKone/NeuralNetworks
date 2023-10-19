import numpy as np
import matplotlib.pyplot as plt

np.random.seed(127)


class NeuralNetworkWTA:
    def __init__(self, x):
        self.winner = None
        self.layer = np.array([])
        self.input_vectors = x
        self.weights = np.random.rand(self.input_vectors.shape[1], 4)
        self.initial_weights = self.weights.copy()
        self.output = np.zeros(4)
        self.nu = 0.5

    def feedforward(self):
        self.weights = self.initial_weights.copy()
        self.output = np.zeros(4)
        for vector in self.input_vectors:
            self.layer = np.dot(vector, self.weights)
            self.winner = np.argmax(self.layer)
            self.output[self.winner] += 1.0
            delta = (self.weights[:, self.winner] +
                     self.nu * (np.transpose(vector) - self.weights[:, self.winner]))
            self.weights[:, self.winner] = delta
            self.show_statistics(vector)

    def feedforward_with_penalties(self):
        self.weights = self.initial_weights.copy()
        self.output = np.zeros(4)
        for vector in self.input_vectors:
            self.layer = np.dot(vector, self.weights)
            self.layer = self.layer - np.array([np.exp(i) if i > 0.0 else 0.0 for i in self.output])
            self.winner = np.argmax(self.layer)
            self.output[self.winner] += 1.0
            delta = (self.weights[:, self.winner] +
                     self.nu * (np.transpose(vector) - self.weights[:, self.winner]))
            self.weights[:, self.winner] = delta
            self.show_statistics(str(vector))

    def show_statistics(self, vector):
        print("-" * 100)
        print("Входной вектор:")
        print(vector)
        print("*" * 10)

        print("Значения выходных значений нейронов:")
        print(self.layer)
        print("*" * 10)

        print(f"Победитель нейрон №{self.winner + 1}")
        print("*" * 10)

        print(f"Новые веса нейрона № {self.winner + 1}")
        print(self.weights[:, self.winner])
        print("*" * 10)

        print("Изначальные веса:")
        print(self.initial_weights)
        print("*" * 10)

        print("Веса нейронов:")
        print(self.weights)
        print("*" * 10)

        print("Количество побед нейронов:")
        print(self.output)


if __name__ == '__main__':
    input_vectors = np.array([[0.97, 0.20],
                              [1.0, 0],
                              [-0.72, 0.7],
                              [-0.67, 0.74],
                              [-0.8, 0.6],
                              [0, -1.0],
                              [0.2, -0.97],
                              [-0.3, -0.95]
                              ])
    wta = NeuralNetworkWTA(input_vectors)
    print("@" * 100)
    print("Обычный метод")
    print("@" * 100)
    wta.feedforward()
    print("@" * 100)
    print("Метод со штрафами")
    print("@" * 100)
    wta.feedforward_with_penalties()
