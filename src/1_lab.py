import numpy as np
from neural_network import NeuralNetworkWTA
import matplotlib.pyplot as plt

N = 4


def act(x):
    pass


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
print("@"*100)
print("Обычный метод")
print("@"*100)
wta.feedforward()
print("@"*100)
print("Метод со штрафами")
print("@"*100)
wta.feedforward_with_penalties()
