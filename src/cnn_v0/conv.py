import numpy as np

class Conv2D:
    '''
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        '''
        Inicializa la capa convolucional 2D.

        Parámetros:
        - in_channels: Número de canales de entrada. Por ejemplo, 3 para una imagen en RGB y 1 para una 
                       imagen en escala de grises.
        - out_channels: Número de filtros (canales de salida).
        - kernel_size: Tamaño del filtro (entero o tupla).
        - stride: Paso de la convolución.
        - padding: Cantidad de relleno (padding) alrededor de la entrada.
        '''

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Inicialización He para los pesos
        scale = np.sqrt(2. / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weights = np.random.normal(0, scale, (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.biases = np.zeros(out_channels)
        
        # Parámetros del optimizador Adam
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)
        self.t = 0
        
    def forward(self, x):
        '''
        Realiza la operación de convolución sobre la entrada.

        Parámetros:
        - x: tensor de entrada de forma (batch_size, in_channels, altura, ancho)

        Retorna:
        - output: tensor de salida después de aplicar la convolución
        '''
         
        self.input = x
        batch_size, in_channels, in_h, in_w = x.shape
        
        # Cálculo de dimensiones de salida
        out_h = (in_h + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        # Aplicar padding si es necesario
        if self.padding > 0:
            padded_x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                             mode='constant')
        else:
            padded_x = x
            
        # Inicializar salida
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # Aplicar convolución
        for i in range(out_h):
            for j in range(out_w):

                # Variables para calcular la ventana
                h_start = i * self.stride
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride
                w_end = w_start + self.kernel_size[1]
                
                # Ventana
                x_slice = padded_x[:, :, h_start:h_end, w_start:w_end]

                # Aplicacion de los filtros en la ventana calculada para cada canal de salida
                for k in range(self.out_channels):
                    output[:, k, i, j] = np.sum(x_slice * self.weights[k, :, :, :], axis=(1, 2, 3)) + self.biases[k]
                    
        return output
    
    def backward(self, grad_output, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        '''
        Calcula el gradiente y actualiza los pesos usando el optimizador Adam.

        Parámetros:
        - grad_output: gradiente de la pérdida respecto a la salida de esta capa
        - learning_rate: tasa de aprendizaje
        - beta1, beta2, epsilon: hiperparámetros de Adam

        Retorna:
        - grad_input: gradiente respecto a la entrada
        '''

        batch_size, _, out_h, out_w = grad_output.shape
        grad_input = np.zeros_like(self.input)
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.zeros_like(self.biases)

        # Padding para entrada y gradiente si es necesario
        if self.padding > 0:
            padded_input = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                                mode='constant')
            grad_input_padded = np.zeros_like(padded_input)
        else:
            padded_input = self.input
            grad_input_padded = grad_input

        # Calcular gradientes
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride
                w_end = w_start + self.kernel_size[1]

                x_slice = padded_input[:, :, h_start:h_end, w_start:w_end]  # (B, C_in, kH, kW)

                for k in range(self.out_channels):
                    # Gradiente de los pesos
                    grad_weights[k] += np.sum(
                        x_slice * grad_output[:, k, i, j][:, None, None, None],
                        axis=0  # Sum over batch
                    )

                    # Gradiente con respecto al input
                    grad_input_padded[:, :, h_start:h_end, w_start:w_end] += (
                        self.weights[k][None, :, :, :] * grad_output[:, k, i, j][:, None, None, None]
                    )

                # Gradiente de los sesgos 
                grad_biases += np.sum(grad_output[:, :, i, j], axis=0)

        # Remover padding del gradiente de entrada
        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        # Normalizar gradientes
        grad_weights /= batch_size
        grad_biases /= batch_size

        # Actualización Adam para pesos
        self.t += 1
        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_weights
        self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_weights ** 2)
        m_w_hat = self.m_w / (1 - beta1 ** self.t)
        v_w_hat = self.v_w / (1 - beta2 ** self.t)
        self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)

        # Actualización Adam para sesgos
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_biases
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_biases ** 2)
        m_b_hat = self.m_b / (1 - beta1 ** self.t)
        v_b_hat = self.v_b / (1 - beta2 ** self.t)
        self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        return grad_input