import numpy as np
from keras.datasets import cifar10
from cnn import SimpleCNN
from utility_functions import softmax, one_hot, accuracy, cross_entropy_loss, cross_entropy_grad


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, y_train = x_train[:1000], y_train[:1000].flatten() 
x_train = x_train.astype(np.float32) / 255.0
x_train = np.transpose(x_train, (0, 3, 1, 2)) 
y_train_oh = one_hot(y_train, 10)


model = SimpleCNN()
lr = 0.01
epochs = 100
batch_size = 32

for epoch in range(epochs):
    idx = np.random.permutation(len(x_train))
    x_train, y_train_oh = x_train[idx], y_train_oh[idx]

    losses = []
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train_oh[i:i+batch_size]

        # forward
        logits = model.forward(x_batch)
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y_batch)
        losses.append(loss)

        # backward
        d_out = cross_entropy_grad(probs, y_batch)
        model.backward(d_out, lr)
    
    # Accuracy
    logits = model.forward(x_train)
    probs = softmax(logits)
    acc = accuracy(probs, y_train_oh)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.4f}, Accuracy: {acc*100:.4f}%")


