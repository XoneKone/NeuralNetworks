import numpy as np

np.random.seed(127)


class NeuralNetworkHebb:
    def __init__(self, x):
        self.winner = None
        self.layer = np.array([])
        self.input_vectors = x
        self.weights = np.random.rand(self.input_vectors.shape[1], 2)
        self.initial_weights = self.weights.copy()
        self.output = np.zeros(1)
        self.nu = 0.5
        self.yu = self.nu / 3

    def feedforward(self):
        self.weights = self.initial_weights.copy()
        self.output = np.zeros(2)
        for vector in self.input_vectors:
            self.layer = np.dot(self.weights, vector)
            self.weights = self.weights * (1 - self.yu) + self.nu * vector * self.layer
            self.show_statistics(vector)

    def show_statistics(self, vector):
        print("-" * 100)
        print("Входной вектор:")
        print(vector)
        print("*" * 10)

        print("Значения выходных значений нейронов:")
        print(self.layer)
        print("*" * 10)

        print("Изначальные веса:")
        print(self.initial_weights)
        print("*" * 10)

        print("Веса нейронов:")
        print(self.weights)
        print("*" * 10)


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
    n = NeuralNetworkHebb(input_vectors)
    n.feedforward()
