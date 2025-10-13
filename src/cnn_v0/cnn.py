import numpy as np

from conv import Conv2D
from pooling import MaxPool2D
from ReLU import ReLU
from loss import SoftmaxCrossEntropy
from flatten import Flatten
from dense import Dense

class CNN:
    def __init__(self):

        # mnist model 
        # self.layers = [
        #     Conv2D(1, 32, 3, stride=1, padding=1),
        #     ReLU(),
        #     MaxPool2D(2, stride=2),
        #     Conv2D(32, 64, 3, stride=1, padding=1),
        #     ReLU(),
        #     MaxPool2D(2, stride=2),
        #     Flatten(),
        #     Dense(7*7*64, 128),  
        #     ReLU(),
        #     Dense(128, 10)
        # ]

        # cifar-10 v0
        self.layers = [
            Conv2D(3, 32, 3, stride=1, padding=1),
            ReLU(),
            # Conv2D(32, 32, 3, stride=1, padding=1),
            # ReLU(),
            MaxPool2D(2, stride=2),
            Conv2D(32, 64, 3, stride=1, padding=1),
            ReLU(),
            # Conv2D(64, 64, 3, stride=1, padding=1),
            # ReLU(),
            MaxPool2D(2, stride=2),
            Conv2D(64, 128, 3, stride=1, padding=1),
            ReLU(),
            # Conv2D(128, 128, 3, stride=1, padding=1),
            # ReLU(),
            MaxPool2D(2, stride=2),
            Flatten(),
            Dense(128*4*4, 128),
            ReLU(),
            Dense(128, 10),
        ]

        # cifar-10 v1
        # self.layers = [
        #     Conv2D(3, 32, 3, stride=1, padding=1),
        #     ReLU(),
        #     MaxPool2D(2, stride=2),
        #     Flatten(),
        #     Dense(32*16*16, 10),
        # ]

        #oxford-flowers-17
        # self.layers = [

        #     Conv2D(3, 32, 3, stride=1, padding=1),
        #     ReLU(),
        #     MaxPool2D(2, stride=2),
        #     Conv2D(32, 64, 3, stride=1, padding=1),
        #     ReLU(),
        #     MaxPool2D(2, stride=2),
        #     Flatten(),
        #     Dense(64*8*8, 128),
        #     ReLU(),
        #     Dense(128, 17)
        # ]
        
        self.loss_fn = SoftmaxCrossEntropy()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)
        return grad_output
    
    def train_step(self, x, y, learning_rate):
        # Forward pass
        output = self.forward(x)
        loss = self.loss_fn.forward(output, y)
        
        # Backward pass
        grad_output = self.loss_fn.backward()
        self.backward(grad_output, learning_rate)
        
        return loss
    
    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)
    
    def accuracy(self, x, y):
        predictions = self.predict(x)
        return np.mean(predictions == np.argmax(y, axis=1))

