import numpy as np

class ReLU:
    def forward(self, x):
        """
        Aplica la función de activación ReLU (Rectified Linear Unit) en el paso hacia adelante.

        Parámetros:
        - x: tensor de entrada (puede tener cualquier forma)

        Retorna:
        - Tensor de salida con la misma forma que x, donde cada elemento negativo se reemplaza por 0.
        """
        
        # Guarda la entrada para usarla en la retropropagación
        self.input = x 
        # ReLU: f(x) = max(0, x)
        return np.maximum(0, x)
    
    def backward(self, grad_output, learning_rate=None):
        """
        Calcula el gradiente de la función ReLU durante la retropropagación.

        Parámetros:
        - grad_output: gradiente de la pérdida respecto a la salida de esta capa
        - learning_rate: no se utiliza en ReLU, pero se incluye por compatibilidad

        Retorna:
        - grad_input: gradiente respecto a la entrada, donde solo se propaga en posiciones donde la entrada fue positiva
        """

        # Derivada de ReLU: 1 si input > 0, 0 en caso contrario
        return grad_output * (self.input > 0)