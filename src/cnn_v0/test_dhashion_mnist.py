import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import to_categorical

from cnn import CNN

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train[:100]
y_train = y_train[:100]

x_train = x_train.reshape(-1, 1, 28, 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 1, 28, 28).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


cnn = CNN()
batch_size = 64
learning_rate = 0.001
epochs = 5
train_losses = []
val_accuracies = []


for epoch in range(epochs):
    epoch_loss = 0
    print(f"Epoch {epoch+1}/{epochs}")
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        loss = cnn.train_step(x_batch, y_batch, learning_rate)
        epoch_loss += loss
    
    epoch_loss /= (len(x_train) / batch_size)
    val_acc = cnn.accuracy(x_test[:100], y_test[:100]) 
    # Store metrics
    train_losses.append(epoch_loss)
    val_accuracies.append(val_acc)

    print(f"Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.show()
