import numpy as np

# Layers

class Layers :

	def __init__(self):
		self.dense = self.Dense()

	class Dense :

		def __init__(self,inputs,neurons):
			self.weights = 0.01*np.random.randn(inputs,neurons)
			self.biases = np.zeros((1,neurons))

		def forward_pass(self,inputs):
			self.output = np.dot(inputs,self.weights) + self.biases


# Activations functions

class Activations:

	def __init__(self):
		self.ReLU = self.ReLU()
		self.Softmax = self.Softmax()

	class ReLU:

		def forward_pass(self, inputs):
			self.output = np.maximum(0,inputs)

	class Softmax:

		def forward_pass(self, inputs):
			exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
			probabilities = exps / np.sum(exps, axis=1, keepdims=True)
			self.output = probabilities


# Loss functions

class Loss_functions:

	def __init__(self):
		self.Loss = self.Loss()

	class Loss:

		def calculate(self, output, y):
			samples_loss = self.forward(output, y)
			data_loss = np.mean(samples_loss)
			return data_loss

	class Categorical_crossentropy(Loss):

		def forward(self, y_pred, y_true):
			samples = len(y_pred)
			y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
			if len(y_true.shape) == 1: 
				correct_confidences = y_pred_clipped[range(samples), y_true]
			elif len(y_true.shape) == 2: 
				correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
			negative_log_likelihoods = -np.log(correct_confidences)
			return negative_log_likelihoods
