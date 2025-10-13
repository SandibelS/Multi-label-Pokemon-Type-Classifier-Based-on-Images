import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        """
        Inicializa una capa densa (fully connected).

        Parámetros:
        - input_size: número de características de entrada.
        - output_size: número de neuronas de salida.

        Inicializa:
        - Pesos con la técnica de He (ideal para ReLU).
        - Sesgos en cero.
        - Parámetros del optimizador Adam (m, v, t).
        """

        # Inicialización He para mejorar la convergencia con activaciones ReLU        
        scale = np.sqrt(2. / input_size)
        self.weights = np.random.normal(0, scale, (input_size, output_size))
        self.biases = np.zeros(output_size)
        
        # Parámetros del optimizador Adam
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)
        self.t = 0
        
    def forward(self, x):
        """
        Realiza el paso hacia adelante de la capa densa.

        Parámetros:
        - x: tensor de entrada de forma (batch_size, input_size)

        Retorna:
        - salida: tensor de forma (batch_size, output_size)
        """

        # Guarda la entrada para usarla en la retropropagación
        self.input = x

        # Producto matricial + sesgo
        return np.dot(x, self.weights) + self.biases
    
    def backward(self, grad_output, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Realiza el paso hacia atrás (retropropagación) y actualiza los parámetros con Adam.

        Parámetros:
        - grad_output: gradiente de la pérdida respecto a la salida de esta capa
        - learning_rate: tasa de aprendizaje
        - beta1, beta2: coeficientes de decaimiento para el promedio móvil
        - epsilon: valor pequeño para evitar división por cero

        Retorna:
        - grad_input: gradiente respecto a la entrada de esta capa

        """

         # Gradiente respecto a la entrada (para capas anteriores)
        grad_input = np.dot(grad_output, self.weights.T)

        # Gradiente respecto a los pesos y sesgos
        grad_weights = np.dot(self.input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)
        
        # Actualización de pesos con Adam
        self.t += 1
        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_weights
        self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_weights ** 2)
        m_w_hat = self.m_w / (1 - beta1 ** self.t)
        v_w_hat = self.v_w / (1 - beta2 ** self.t)
        self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        
        # Actualización de sesgos con Adam
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_biases
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_biases ** 2)
        m_b_hat = self.m_b / (1 - beta1 ** self.t)
        v_b_hat = self.v_b / (1 - beta2 ** self.t)
        self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        
        return grad_input