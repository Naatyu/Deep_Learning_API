from NeuralNets import *
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

X, y = spiral_data(samples=100, classes=3) # produce 2 variables, x and y (coordinates)

dense1 = Layers.Dense(2,3)
activation1 = Activations.ReLU()
dense2 = Layers.Dense(3,3)
activation2 = Activations.Softmax()

dense1.forward_pass(X)
activation1.forward_pass(dense1.output)
dense2.forward_pass(activation1.output)
activation2.forward_pass(dense2.output)

print(activation2.output[:5])

loss_function = Loss_functions.Categorical_crossentropy()
loss = loss_function.calculate(activation2.output,y)
print(loss)