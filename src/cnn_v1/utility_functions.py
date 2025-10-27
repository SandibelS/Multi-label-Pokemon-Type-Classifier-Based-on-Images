import numpy as np

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ------

def one_hot(y, num_classes):
    one_hot_y = np.zeros((y.size, num_classes))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y

# -------------------------------
# Loss
# -------------------------------
def cross_entropy_loss(y_pred, y_true):
    eps = 1e-9
    loss = -np.sum(y_true * np.log(y_pred + eps)) / y_true.shape[0]
    return loss

def cross_entropy_grad(y_pred, y_true):
    return (y_pred - y_true) / y_true.shape[0]

# ------

def binary_cross_entropy(y_pred, y_true):
    eps = 1e-9
    loss = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
    return loss

def binary_cross_entropy_grad(y_pred, y_true):
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0]) 


# ------

def accuracy(y_pred, y_true):
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(pred_labels == true_labels)

def top_k_accuracy(y_pred, y_true, k=2):
    """
    Calcula la precisión Top-k para clasificación multi-label
    """
    correct = 0
    for i in range(len(y_true)):
        top_k_indices = np.argsort(y_pred[i])[::-1][:k]
        true_indices = np.where(y_true[i] == 1)[0]
        if any(label in top_k_indices for label in true_indices):
            correct += 1
            
    return correct / len(y_true)
