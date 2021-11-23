import numpy as np

# Layers

class Layers :

	class Dense :

		def __init__(self,inputs, neurons, weight_regularizer_L1=0, weight_regularizer_L2=0, bias_regularizer_L1=0, bias_regularizer_L2=0): #L1 and L2 regularization to penalize large weights and biases (meaning it is attempting to momeorize a data element)
			self.weights = np.random.randn(inputs,neurons) / np.sqrt(inputs/2) # modified Xavier Initialisation due to ReLU (He et al. 2015)
			self.biases = np.zeros((1,neurons))
			self.weight_regularizer_L1 = weight_regularizer_L1
			self.weight_regularizer_L2 = weight_regularizer_L2
			self.bias_regularizer_L1 = bias_regularizer_L1
			self.bias_regularizer_L2 = bias_regularizer_L2

		def forward_pass(self, inputs, training):
			self.inputs = inputs
			self.output = np.dot(inputs,self.weights) + self.biases

		def backward_pass(self, dvalues):
			self.dweights = np.dot(self.inputs.T, dvalues)
			self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

			if self.weight_regularizer_L1 > 0:
				dL1 = np.ones_like(self.weights)
				dL1[self.weights < 0] = -1
				self.dweights += self.weight_regularizer_L1 * dL1

			if self.weight_regularizer_L2 > 0:
				self.dweights += 2 * self.weight_regularizer_L2 * self.weights

			if self.bias_regularizer_L1 > 0:
				dL1 = np.ones_like(slf.biases)
				dL1[self.biases < 0] = -1
				self.dbiases += self.bias_regularizer_L1 * dL1

			if self.bias_regularizer_L2 > 0:
				self.biases += 2 * self.bias_regularizer_L2 * self.biases

			self.dinputs = np.dot(dvalues, self.weights.T)

	class Dropout:

		def __init__(self, rate):
			self.rate = 1 - rate

		def forward_pass(self, inputs, training):
			self.inputs = inputs

			if not training:
				self.output = inputs.copy()
				return

			self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate #divided by rate to upscale and keep the wight sum during prediction without dropout
			self.output = inputs * self.binary_mask

		def backward_pass(self, dvalues):
			self.dinputs = dvalues * self.binary_mask

	class Input():

		def forward_pass(self, inputs, training):
			self.output = inputs

# Activations functions

class Activations:

	class ReLU:

		def forward_pass(self, inputs, training):
			self.inputs = inputs
			self.output = np.maximum(0,inputs)

		def backward_pass(self, dvalues):
			self.dinputs = dvalues.copy()
			self.dinputs[self.inputs <= 0] = 0

		def predictions(self, outputs):
			return outputs

	class Softmax:

		def forward_pass(self, inputs, training):
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

		def predictions(self, outputs):
			return np.argmax(outputs, axis=1)

	class Sigmoid:

		def forward_pass(self, inputs, training):
			self.inputs = inputs
			self.output = 1 / (1 + np.exp(-inputs))

		def backward_pass(self, dvalues):
			self.dinputs = dvalues * (1 - self.output) * self.output

		def predicitons(self, outputs):
			return (outputs > 0.5) * 1

	class Linear:

		def forward_pass(self, inputs, training):
			self.inputs = inputs
			self.output = inputs

		def backward_pass(self, dvalues):
			self.dinputs = dvalues.copy()

		def predictions(self, outputs):
			return outputs


# Loss functions

class Loss_functions:

	class Loss:

		def regularization_loss(self):
			regularization_loss = 0

			for layer in self.trainable_Layers:

				if layer.weight_regularizer_L1 > 0:
					regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))

				if layer.weight_regularizer_L2 > 0:
					regularization_loss += layer.weight_regularizer_L2 * np.sum(layer.weights * layer.weights)

				if layer.bias_regularizer_L1 > 0:
					regularization_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))

				if layer.bias_regularizer_L2 > 0:
					regularization_loss += layer.bias_regularizer_L2 * np.sum(layer.biases * layer.biases)

			return regularization_loss

		def remember_trainable_layers(self, trainable_Layers):
			self.trainable_Layers = trainable_Layers

		def calculate(self, output, y, *, include_regularization=False):
			samples_loss = self.forward_pass(output, y)
			data_loss = np.mean(samples_loss)
			if not include_regularization:
				return data_loss

			return data_loss, self.regularization_loss()

	class Categorical_Crossentropy(Loss):

		def forward_pass(self, y_pred, y_true):
			samples = len(y_pred)
			y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # to avoid division by 0, clipped to both side to keep mean
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

	class Softmax_plus_Categorical_Crossentropy(): # Combination of softmax activation and catcrossentropy loss -> 7 times faster than calculating both gradients separately

		def backward_pass(self, dvalues, y_true):
			samples = len(dvalues)
			if len(y_true.shape) == 2:
				y_true = np.argmax(y_true, axis=1)
			self.dinputs = dvalues.copy()
			self.dinputs[range(samples), y_true] -= 1
			self.dinputs = self.dinputs / samples

	class Binary_Crossentropy(Loss):

		def forward_pass(self, y_pred, y_true):
			y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
			sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
			sample_losses = np.mean(sample_losses, axis=-1)
			return sample_losses

		def backward_pass(self, dvalues, y_true):
			samples = len(dvalues)
			outputs = len(dvalues[0])
			clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
			self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
			self.dinputs = self.dinputs / samples

	class Mean_Squared_Error(Loss):

		def forward_pass(self, y_pred, y_true):
			sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
			return sample_losses

		def backward_pass(self, dvalues, y_true):
			samples = len(dvalues)
			outputs = len(dvalues[0])
			self.dinputs = -2 * (y_true - dvalues) / outputs
			self.dinputs = self.dinputs / samples

	class Mean_Absolute_error(Loss):

		def forward_pass(self, y_pred, y_true):
			sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
			return sample_losses

		def backward_pass(self, dvalues, y_true):
			samples = len(dvalues)
			outputs = len(dvalues[0])
			self.dinputs = np.sign(y_true - dvalues) / outputs
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


# Model, accuracy, train

class Utilities:

	class Accuracy:

		def calculate(self, predictions, y):
			comparisons = self.compare(predictions, y)
			accuracy = np.mean(comparisons)
			return accuracy

	class Accuracy_Categorical(Accuracy):

		def init(self, y):
			pass

		def compare(self, predictions, y):
			if len(y.shape) == 2:
				y = np.argmax(y, axis=1)
			return predictions == y

	class Accuracy_Regression(Accuracy):

		def __init__(self):
			self.precision = None

		def init(self, y, reinit=False):
			if self.precision is None or reinit:
				self.precision = np.std(y) / 250

		def compare(self, predictions, y):
			return np.abs(predictions - y) < self.precision

	class Model:

		def __init__(self):
			self.layers = []
			self.softmax_classifier_output = None

		def add(self, layer):
			self.layers.append(layer)

		def set(self, *, loss, optimizer, accuracy):
			self.loss = loss
			self.optimizer = optimizer
			self.accuracy = accuracy

		def finalize(self):
			self.input_layer = Layers.Input()
			layer_count = len(self.layers)
			self.trainable_Layers = []

			for i in range(layer_count):
				if i == 0:
					self.layers[i].prev = self.input_layer
					self.layers[i].next = self.layers[i+1]
				elif i < layer_count - 1:
					self.layers[i].prev = self.layers[i-1]
					self.layers[i].next = self.layers[i+1]
				else:
					self.layers[i].prev = self.layers[i-1]
					self.layers[i].next = self.loss
					self.output_layer_activation = self.layers[i]

				if hasattr(self.layers[i], 'weights'):
					self.trainable_Layers.append(self.layers[i])

				self.loss.remember_trainable_layers(self.trainable_Layers)

				if isinstance(self.layers[-1], Activations.Softmax) and isinstance(self.loss, Loss_functions.Categorical_Crossentropy):
					self.softmax_classifier_output = Loss_functions.Softmax_plus_Categorical_Crossentropy()

		def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
			self.accuracy.init(y)
			for epoch in range(1, epochs+1):
				output = self.forward_pass(X, training=True)

				data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
				loss = data_loss + regularization_loss
				predictions = self.output_layer_activation.predictions(output)
				accuracy = self.accuracy.calculate(predictions, y)
				self.backward_pass(output, y)

				self.optimizer.pre_update()
				for layer in self.trainable_Layers:
					self.optimizer.update_params(layer)
				self.optimizer.post_update()

				if not epoch % print_every:
					print(f'Epoch : {epoch}, ' +
						  f'Accuracy : {accuracy:.3f}, ' +
						  f'Loss : {loss:.3f}, ' +
						  f'Data Loss : {data_loss:.3f}, ' +
						  f'Regularization Loss : {regularization_loss:.3f}, ' +
						  f'Learning rate : {self.optimizer.current_learning_rate}')

			if validation_data is not None:
				X_val, y_val = validation_data
				output = self.forward_pass(X_val, training=False)
				loss = self.loss.calculate(output, y_val)
				predictions = self.output_layer_activation.predictions(output)
				accuracy = self.accuracy.calculate(predictions, y_val)

				if not epoch % print_every:
					print(f'Validation, ' +
						  f'Accuracy : {accuracy:.3f}, ' +
						  f'Loss : {loss:.3f}')

		def forward_pass(self, X, training):
			self.input_layer.forward_pass(X, training)
			for layer in self.layers:
				layer.forward_pass(layer.prev.output, training)
			return layer.output

		def backward_pass(self, output, y):
			if self.softmax_classifier_output is not None:
				self.softmax_classifier_output.backward_pass(output, y)
				self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

				for layer in reversed(self.layers[:-1]):
					layer.backward_pass(layer.next.dinputs)
				return

			self.loss.backward_pass(output, y)
			for layer in reversed(self.layers):
				layer.backward_pass(layer.next.dinputs)