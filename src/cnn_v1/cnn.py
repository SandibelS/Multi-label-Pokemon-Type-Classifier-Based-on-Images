import numpy as np

# -------------------------------
# Layers
# -------------------------------
class Conv2D:

    def __init__(self, num_filters, filter_size, input_channels, stride=1, padding=0):
        """
        Capa convolucional 2D implementada manualmente.
        Aplica filtros sobre una entrada con múltiples canales (como RGB).

        Parametros:
            - num_filters (int): número de filtros (mapas de características) a aplicar.
            - filter_size (int): tamaño de cada filtro (ej. 3 para 3x3).
            - input_channels (int): número de canales de entrada (ej. 3 para RGB).
            - stride (int): paso de desplazamiento del filtro.
            - padding (int): cantidad de ceros agregados alrededor de la imagen.
        """
        
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.stride = stride
        self.padding = padding
        
        ## Inicialización Xavier para los filtros
        limit = np.sqrt(6 / (input_channels * filter_size * filter_size + num_filters))
        self.filters = np.random.uniform(-limit, limit, (num_filters, input_channels, filter_size, filter_size))
        self.bias = np.zeros((num_filters, 1))
    
    def forward(self, x):
        """
        Propagación hacia adelante: aplica convolución sobre la entrada.

        Parametros:
            - x (ndarray): tensor de entrada de forma (batch_size, channels, height, width)

        Retorna:
            - output (ndarray): tensor convolucionado
        """
        self.input = x
        batch_size, in_channels, in_h, in_w = x.shape
        f = self.filter_size
        out_h = int((in_h - f + 2 * self.padding) / self.stride) + 1
        out_w = int((in_w - f + 2 * self.padding) / self.stride) + 1
        
        self.output = np.zeros((batch_size, self.num_filters, out_h, out_w))
        
        # Aplicar padding si es necesario
        if self.padding > 0:
            x_padded = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x

        # Aplicar convolución
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + f
                w_start = j * self.stride
                w_end = w_start + f

                region = x_padded[:, :, h_start:h_end, w_start:w_end]
                self.output[:, :, i, j] = np.tensordot(region, self.filters, axes=([1,2,3],[1,2,3])) + self.bias.squeeze()
        return self.output

    def backward(self, d_out, lr):
        """
        Propagación hacia atrás.

        Parámetros:
        - d_out (ndarray): gradiente de salida
        - lr (float): tasa de aprendizaje

        Retorna:
        - d_input (ndarray): gradiente respecto a la entrada
        """

        batch_size, _, out_h, out_w = d_out.shape
        f = self.filter_size
        d_filters = np.zeros_like(self.filters)
        d_bias = np.sum(d_out, axis=(0,2,3)).reshape(self.bias.shape)
        d_input = np.zeros_like(self.input)
        
        if self.padding > 0:
            x_padded = np.pad(self.input, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
            d_x_padded = np.zeros_like(x_padded)
        else:
            x_padded = self.input
            d_x_padded = d_input

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + f
                w_start = j * self.stride
                w_end = w_start + f

                region = x_padded[:, :, h_start:h_end, w_start:w_end]

                for n in range(batch_size):
                    for k in range(self.num_filters):
                        d_filters[k] += region[n] * d_out[n, k, i, j]
                        d_x_padded[n, :, h_start:h_end, w_start:w_end] += self.filters[k] * d_out[n, k, i, j]
        
        if self.padding > 0:
            d_input = d_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = d_x_padded

        # Actualizar filtros y sesgos
        self.filters -= lr * d_filters / batch_size
        self.bias -= lr * d_bias / batch_size
        
        return d_input

class ReLU:
    """
    Capa de activación ReLU.

    Aplica la función f(x) = max(0, x).
    """

    def forward(self, x):
        """
        Parámetros:
            - x (ndarray): entrada

        Retorna:
            - salida activada
        """
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, d_out):
        """
        Propagación hacia atrás.

        Parámetros:
        - d_out (ndarray): gradiente de salida

        Retorna:
        - gradiente modificado por la derivada de ReLU
        """

        return d_out * (self.input > 0)

class MaxPool2D:

    def __init__(self, size=2, stride=2):
        """
        Capa de max pooling 2D.

        Reduce la resolución espacial tomando el valor máximo en regiones.

        Parámetros:
        - size (int): tamaño de la ventana de pooling.
        - stride (int): paso de desplazamiento de la ventana.
        """
    
        self.size = size
        self.stride = stride

    def forward(self, x):
        """
        Parámetros:
        - x (ndarray): entrada de forma (batch_size, channels, height, width)

        Retorna:
        - salida con resolución reducida
        """

        self.input = x
        batch_size, channels, h, w = x.shape
        out_h = (h - self.size)//self.stride + 1
        out_w = (w - self.size)//self.stride + 1
        output = np.zeros((batch_size, channels, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.size
                w_start = j * self.stride
                w_end = w_start + self.size
                region = x[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(region, axis=(2,3))
        self.output = output
        return output

    def backward(self, d_out):
        """
        Propagación hacia atrás.

        Parámetros:
        - d_out (ndarray): gradiente de salida

        Retorna:
        - gradiente respecto a la entrada
        """

        d_input = np.zeros_like(self.input)
        batch_size, channels, out_h, out_w = d_out.shape

        for i in range(out_h):
            for j in range(out_w):

                h_start = i * self.stride
                h_end = h_start + self.size
                w_start = j * self.stride
                w_end = w_start + self.size

                region = self.input[:, :, h_start:h_end, w_start:w_end]
                max_mask = (region == np.max(region, axis=(2,3), keepdims=True))
                d_input[:, :, h_start:h_end, w_start:w_end] += max_mask * (d_out[:, :, i, j])[:, :, None, None]

        return d_input

class Flatten:
    """
    Capa que aplana la entrada.

    Convierte un tensor 4D en un vector 2D para conectarlo a capas densas.
    """

    def forward(self, x):
        """
        Parámetros:
        - x (ndarray): entrada de forma (batch_size, channels, height, width)

        Retorna:
        - vector plano de forma (batch_size, features)
        """

        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, d_out):
        """
        Reconstruye la forma original del tensor.

        Parámetros:
        - d_out (ndarray): gradiente plano

        Retorna:
        - gradiente con forma original
        """

        return d_out.reshape(self.input_shape)

class Dense:

    def __init__(self, in_features, out_features):
        """
        Capa totalmente conectada (fully connected).

        Parámetros:
        - in_features (int): número de características de entrada.
        - out_features (int): número de neuronas de salida.
        """

        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros((1, out_features))

    def forward(self, x):
        """
        Parámetros:
        - x (ndarray): entrada de forma (batch_size, in_features)

        Retorna:
        - salida lineal de forma (batch_size, out_features)
        """

        self.input = x
        return x @ self.W + self.b


    def backward(self, d_out, lr):
        """
        Propagación hacia atrás.

        Parámetros:
        - d_out (ndarray): gradiente de salida
        - lr (float): tasa de aprendizaje

        Retorna:
        - gradiente respecto a la entrada
        """

        dW = self.input.T @ d_out
        db = np.sum(d_out, axis=0, keepdims=True)
        d_input = d_out @ self.W.T

        self.W -= lr * dW / d_out.shape[0]
        self.b -= lr * db / d_out.shape[0]
        return d_input


# -------------------------------
# Simple CNN Model
# -------------------------------
class SimpleCNN:
    """
    Red neuronal convolucional simple.

    """
    def __init__(self):
        """
        Inicializa las capas del modelo.
        """

        # 32 filters, 3x3, RGB input
        self.conv1 = Conv2D(32, 3, 3)  
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)


        self.flatten = Flatten()
        # CIFAR10: input 32x32 -> 15x15 after conv+pool
        self.fc1 = Dense(32*15*15, 10)  
        
    def forward(self, x):
        """
        Propagación hacia adelante del modelo completo.

        Parámetros:
        - x (ndarray): entrada de imágenes

        Retorna:
        - logits (ndarray): salida sin activación final
        """

        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        out = self.flatten.forward(out)
        out = self.fc1.forward(out)

        return out
    
    def backward(self, d_out, lr):
        """
        Propagación hacia atrás del modelo completo.

        Parámetros:
        - d_out (ndarray): gradiente de la función de pérdida
        - lr (float): tasa de aprendizaje
        """

        d = self.fc1.backward(d_out, lr)
        d = self.flatten.backward(d)
        d = self.pool1.backward(d)
        d = self.relu1.backward(d)
        d = self.conv1.backward(d, lr)


