from Personnal_framework import *
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

X, y = spiral_data(samples=100, classes=3) # produce 2 variables, x and y (coordinates)

#plt.scatter(X[:,0],X[:,1],c=y)
#plt.show()

dense1 = Layers.Dense(2,64)
activation1 = Activations.ReLU()
dense2 = Layers.Dense(64,3)

loss_activation = Loss_functions.Softmax_plus_Categorical_crossentropy()

optimizer = Optimizers.Adam(learning_rate = 0.05, decay=5e-7)

for epoch in range(10001):

	dense1.forward_pass(X)
	activation1.forward_pass(dense1.output)
	dense2.forward_pass(activation1.output)
	loss = loss_activation.forward_pass(dense2.output, y)

	predictions = np.argmax(loss_activation.output, axis=1)
	if len(y.shape)==2:
		y = np.argmax(y, axis=1)
	accuracy = np.mean(predictions==y)

	if not epoch % 100:
		print(f'Epoch : {epoch}, ' +
			  f'Accuracy : {accuracy:.3f}, ' +
			  f'Loss : {loss:.3f}, ' +
			  f'Learning_rate : {optimizer.current_learning_rate}')

	loss_activation.backward_pass(loss_activation.output, y)
	dense2.backward_pass(loss_activation.dinputs)
	activation1.backward_pass(dense2.dinputs)
	dense1.backward_pass(activation1.dinputs)

	optimizer.pre_update()
	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
	optimizer.post_update()