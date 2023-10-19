import numpy as np


class NeuralNetworkWTA:
    def __init__(self, x):
        self.winner = None
        self.layer = np.array([])
        self.input_vectors = x
        self.weights = np.random.rand(self.input_vectors.shape[1], 4)
        self.output = np.zeros(4)
        self.nu = 0.5
        print("Изначальные веса - ")
        print(self.weights)

    def feedforward(self):
        for vector in self.input_vectors:
            self.layer = np.dot(vector, self.weights)
            self.winner = np.argmax(self.layer)
            self.output[self.winner] += 1.0
            delta = (self.weights[:, self.winner] +
                     self.nu * (np.transpose(vector) - self.weights[:, self.winner]))
            self.weights[:, self.winner] = delta
            self.show_statistics(str(vector))

    def backforward(self):
        self.weights = self.weights

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

        print("Веса нейронов:")
        print(self.weights)
        print("*" * 10)

        print("Количество побед нейронов:")
        print(self.output)
