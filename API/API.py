import numpy as np

# Layers

class Layers :

	class Dense :

		def __init__(self,inputs, neurons):
			self.weights = 0.01*np.random.randn(inputs,neurons)
			self.biases = np.zeros((1,neurons))

		def forward_pass(self, inputs):
			self.inputs = inputs
			self.output = np.dot(inputs,self.weights) + self.biases

		def backward_pass(self, dvalues):
			self.dweights = np.dot(self.inputs.T, dvalues)
			self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
			self.dinputs = np.dot(dvalues, self.weights.T)


# Activations functions

class Activations:

	class ReLU:

		def forward_pass(self, inputs):
			self.inputs = inputs
			self.output = np.maximum(0,inputs)

		def backward_pass(self, dvalues):
			self.dinputs = dvalues.copy()
			self.dinputs[self.inputs <= 0] = 0

	class Softmax:

		def forward_pass(self, inputs):
			self.inputs = inputs
			exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
			probabilities = exps / np.sum(exps, axis=1, keepdims=True)
			self.output = probabilities

		def backward_pass(self, dvalues):
			self.dinputs = np.empty_like(dvalues)
			for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
				single_output = single_output.reshape(-1, 1)
				jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
				self.dinputs[index] = np.dot(jacobian, single_dvalues)


# Loss functions

class Loss_functions:

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

		def backward_pass(self, dvalues, y_true):
			samples = len(dvalues)
			labels = len(dvalues[0])
			if len(y_true.shape) == 1:
				y_true = np.eye(labels)[y_true]
			self.dinputs = -y_true / dvalues
			self.dinputs = self.dinputs / samples 

	class Softmax_plus_Categorical_crossentropy(): # Combination of softmax activation and catcrossentropy loss -> 7 times faster than calculating both gradients separately

		def __init__(self):
			self.activation = Activations.Softmax()
			self.loss = Loss_functions.Categorical_crossentropy()

		def forward_pass(self, inputs, y_true):
			self.activation.forward_pass(inputs)
			self.output = self.activation.output
			return self.loss.calculate(self.output, y_true)

		def backward_pass(self, dvalues, y_true):
			samples = len(dvalues)
			if len(y_true.shape) == 2:
				y_true = np.argmax(y_true, axis=1)
			self.dinputs = dvalues.copy()
			self.dinputs[range(samples), y_true] -= 1
			self.dinputs = self.dinputs / samples


# Optimizers

class Optimizers:

	class SGD:

		def __init__(self, learning_rate=1., decay=0., momentum=0.):
			self.learning_rate = learning_rate
			self.current_learning_rate = learning_rate
			self.decay = decay
			self.iterations = 0
			self.momentum = momentum

		def pre_update(self):
			if self.decay:
				self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

		def update_params(self, layer):
			if self.momentum:
				if not hasattr(layer, 'weight_momentums'):
					layer.weight_momentums = np.zeros_like(layer.weights)
					layer.bias_momentums = np.zeros_like(layer.biases)

				weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
				layer.weight_momentums = weight_updates

				bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dweights
				layer.bias_momentums = bias_updates

			else:
				weight_updates = -self.current_learning_rate * layer.dweights
				bias_updates = -self.current_learning_rate * layer.dbiases

			layer.weights += weights_updates
			layer.biases += bias_updates

		def post_update(self):
			self.iterations += 1

	class Adagrad:

		def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
			self.learning_rate = learning_rate
			self.current_learning_rate = learning_rate
			self.decay = decay
			self.iterations = 0
			self.epsilon

		def pre_update(self):
			if self.decay:
				self.current_learning_rate = self.learning_rate * (1. / (1. + self.learning_rate * self.iterations))

		def update_params(self, layer):
			if not hasattr(layer, 'weight_cache'):
				layer.weight_cache = np.zeros_like(layer.weights)
				layer.bias_cache = np.zeros_like(layers.biases)

			layer.weight_cache += layer.dweights**2
			layer.bias_cache += layer.dbiases**2

			layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
			layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.weight_cache) + self.epsilon)

		def post_update(self):
			self.iterations += 1

	class RMSprop:

		def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
			self.learning_rate = learning_rate
			self.current_learning_rate = learning_rate
			self.decay = decay
			self.iterations = 0
			self.epsilon = epsilon
			self.rho = rho

		def pre_update(self):
			if self.decay:
				self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

		def update_params(self, layer):
			if not hasattr(layer, 'weight_cache'):
				layer.weight_cache = np.zeros_like(layer.weights)
				layer.bias_cache = np.zeros_like(layer.biases)

			layer.weight_cache = self.rho * self.layer.weight_cache + (1 - self.rho) * layer.dweights**2
			layer.bias_cache = self.rho * self.layer.bias_cache + (1 - self.rho) * layer.dbiases**2

			layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
			layer.biases += -self.current_learning_rate * layer.biases / (np.sqrt(layer.bias_cache) + self.epsilon)

		def post_update(self):
			self.iterations += 1

	class Adam:

		def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
			self.learning_rate = learning_rate
			self.current_learning_rate = learning_rate
			self.decay = decay
			self.iterations = 0
			self.epsilon = epsilon
			self.beta_1 = beta_1
			self.beta_2 = beta_2

		def pre_update(self):
			if self.decay:
				self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

		def update_params(self, layer):
			if not hasattr(layer, 'weight_cache'):
				layer.weight_momentums = np.zeros_like(layer.weights)
				layer.bias_momentums = np.zeros_like(layer.biases)
				layer.weight_cache = np.zeros_like(layer.weights)
				layer.bias_cache = np.zeros_like(layer.biases)

			layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
			layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

			weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
			bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

			layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
			layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

			weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
			bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

			layer.weights += -self.learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
			layer.biases += -self.learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

		def post_update(self):
			self.iterations += 1

