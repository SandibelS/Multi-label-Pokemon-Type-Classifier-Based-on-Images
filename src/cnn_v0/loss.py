import numpy as np

class SoftmaxCrossEntropy:
    def forward(self, x, y):
        """
        Realiza el paso hacia adelante de la combinación de Softmax y pérdida de entropía cruzada.

        Parámetros:
        - x: tensor de predicciones sin normalizar (logits), de forma (batch_size, num_clases)
        - y: tensor de etiquetas verdaderas codificadas en one-hot, de forma (batch_size, num_clases)

        Retorna:
        - loss: valor escalar de la pérdida promedio por entropía cruzada
        """

        # Guarda las etiquetas verdaderas para el paso hacia atrás
        self.y = y

        # Estabilización numérica: resta el máximo por fila para evitar overflow
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))

        self.softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Cálculo de la pérdida por entropía cruzada
        # Se añade un pequeño valor (1e-8) para evitar log(0)
        loss = -np.sum(y * np.log(self.softmax + 1e-8)) / x.shape[0]
        return loss
    
    def backward(self):
        """
        Realiza el paso hacia atrás (retropropagación) de la pérdida Softmax + Entropía Cruzada.

        Retorna:
        - grad_input: gradiente de la pérdida respecto a las entradas (logits), de forma (batch_size, num_clases)
        """
        
        # Derivada simplificada: softmax - etiquetas verdaderas
        return (self.softmax - self.y) / self.y.shape[0]