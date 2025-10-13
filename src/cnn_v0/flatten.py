import numpy as np

class Flatten:
    def forward(self, x):
        """
        Realiza el paso hacia adelante de la capa Flatten.

        Esta capa convierte un tensor multidimensional en un tensor 2D, 
        aplanando todas las dimensiones excepto la del batch.

        Parámetros:
        - x: tensor de entrada de forma (batch_size, ...), donde "..." representa cualquier número de dimensiones adicionales.

        Retorna:
        - Tensor de salida de forma (batch_size, características_planas), donde todas las dimensiones excepto la primera se combinan en una sola.

        """
        # Guarda la forma original para usarla en la retropropagación
        self.input_shape = x.shape

        # Aplana todas las dimensiones excepto la del batch
        return x.reshape(x.shape[0], -1)
    
    def backward(self, grad_output, learning_rate=None):
        """
        Realiza el paso hacia atrás (retropropagación) de la capa Flatten.

        Restaura la forma original del tensor antes de que fuera aplanado, 
        permitiendo que el gradiente fluya correctamente hacia capas anteriores.

        Parámetros:
        - grad_output: gradiente de la pérdida respecto a la salida de esta capa (forma aplanada)
        - learning_rate: no se utiliza en esta capa, pero se incluye por compatibilidad

        Retorna:
        - grad_input: gradiente con la misma forma que la entrada original
        """
        
        return grad_output.reshape(self.input_shape)
    
    