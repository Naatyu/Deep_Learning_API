from Personnal_framework import *
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Utilities.Model()
model.add(Layers.Dense(2, 512)) #weight_regularizer_L2=5e-4, bias_regularizer_L2=5e-4))
model.add(Activations.ReLU())
model.add(Layers.Dropout(0.1))
model.add(Layers.Dense(512, 3))
model.add(Activations.Softmax())

model.set(
	loss = Loss_functions.Categorical_Crossentropy(),
	optimizer = Optimizers.Adam(learning_rate=0.05, decay=5e-5),
	accuracy = Utilities.Accuracy_Categorical()
)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)