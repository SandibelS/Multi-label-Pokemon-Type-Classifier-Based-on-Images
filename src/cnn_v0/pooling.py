import numpy as np

class MaxPool2D:
    def __init__(self, kernel_size, stride=None, padding=0):
        """
        Inicializa la capa de MaxPooling 2D.

        Parámetros:
        - kernel_size: tamaño de la ventana de pooling (entero o tupla).
        - stride: paso entre ventanas. Si no se especifica, se usa el mismo valor que kernel_size.
        - padding: cantidad de relleno (padding) aplicado a los bordes de la entrada.

        """

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        # If stride is not specified, default to kernel size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        """
        Realiza la operación de max pooling sobre la entrada.

        Parámetros:
        - x: tensor de entrada con forma (batch_size, canales, altura, ancho)

        Retorna:
        - output: tensor de salida después de aplicar max pooling

        """

        self.input = x
        batch_size, channels, in_h, in_w = x.shape

        # Calculate output dimensions
        out_h = (in_h + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size[1]) // self.stride + 1

        # Add padding if specified
        if self.padding > 0:
            padded_x = np.pad(x, ((0, 0), (0, 0), 
                                  (self.padding, self.padding), 
                                  (self.padding, self.padding)), mode='constant')
        else:
            padded_x = x

        # Initialize output and max indices for backpropagation
        output = np.zeros((batch_size, channels, out_h, out_w))
        self.max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=int)

        # Perform max pooling
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride
                w_end = w_start + self.kernel_size[1]

                # Extract pooling window
                pool_region = padded_x[:, :, h_start:h_end, w_start:w_end]

                # Take maximum over spatial dimensions
                output[:, :, i, j] = np.max(pool_region, axis=(2, 3))

                # Store index of max value for each channel in each window 
                flat_indices = np.argmax(pool_region.reshape(batch_size, channels, -1), axis=2)
                self.max_indices[:, :, i, j, 0] = h_start + flat_indices // self.kernel_size[1]
                self.max_indices[:, :, i, j, 1] = w_start + flat_indices % self.kernel_size[1]

        return output

    def backward(self, grad_output, learning_rate=None):
        """
        
        Calcula el gradiente de la entrada durante la retropropagación.

        Parámetros:
        - grad_output: gradiente de la pérdida respecto a la salida de esta capa
        - learning_rate: no se utiliza en esta capa, pero se incluye por compatibilidad

        Retorna:
        - grad_input: gradiente respecto a la entrada

        """

        batch_size, channels, out_h, out_w = grad_output.shape

        # Initialize gradient input with zeros
        grad_input = np.zeros_like(self.input)

        # Backpropagate only to positions of maximum values
        for i in range(out_h):
            for j in range(out_w):
                for b in range(batch_size):
                    for c in range(channels):
                        h_idx, w_idx = self.max_indices[b, c, i, j]
                        grad_input[b, c, h_idx, w_idx] += grad_output[b, c, i, j]

        return grad_input