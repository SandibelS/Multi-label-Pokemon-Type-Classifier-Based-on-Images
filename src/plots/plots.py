import matplotlib.pyplot as plt

def plot_feature_maps(feature_maps, limit=10, path='feature_maps.png'):
    """
    Grafica y guarda los mapas de características generados por los filtros de una capa convolucional.

    Parámetros:
    - feature_maps (ndarray): salida de la capa convolucional, forma (n_filters, height, width)
    - limit (int): número máximo de mapas a graficar
    - path(str): ruta junto a nombre del archivo para guardar la imagen (opcional)

    Retorna:
    - None
    """
    n_maps = min(limit, feature_maps.shape[0])
    fig, axes = plt.subplots(1, n_maps, figsize=(3 * n_maps, 3))

    for i in range(n_maps):
        ax = axes[i] if n_maps > 1 else axes
        ax.imshow(feature_maps[i], cmap='viridis')
        ax.set_title(f'Filtro {i}')
        ax.set_xlabel('Ancho')
        ax.set_ylabel('Alto')
        ax.axis('on')

    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_maxpool_output(pool_output, limit=10, path='maxpool_output'):
    """
    Grafica y guarda los resultados de aplicar MaxPooling sobre los mapas de características.

    Parámetros:
    - pool_output (ndarray): salida de la capa MaxPool, forma (n_maps, pooled_height, pooled_width)
    - limit (int): número máximo de mapas a graficar
    - path (str): ruta junto a nombre del archivo para guardar la imagen (opcional)

    Retorna:
    - None
    """
    
    n_maps = min(limit, pool_output.shape[0])
    fig, axes = plt.subplots(1, n_maps, figsize=(3 * n_maps, 3))

    for i in range(n_maps):
        ax = axes[i] if n_maps > 1 else axes
        ax.imshow(pool_output[i], cmap='plasma')
        ax.set_title(f'MaxPool {i}')
        ax.set_xlabel('Ancho')
        ax.set_ylabel('Alto')
        ax.axis('on')

    plt.tight_layout()
    plt.savefig(path)

    plt.show()


def image_grid(imagenes, filas=4, columnas=4, tamaño=(15,15), filename='src/plots/image_grid.png'):
    """
    Muestra una cuadrícula de imágenes usando matplotlib.

    Parámetros:
    - imagenes (list o ndarray): lista o arreglo de imágenes (cada una de forma H x W x C)
    - filas (int): número de filas en la cuadrícula
    - columnas (int): número de columnas en la cuadrícula
    - tamaño (tuple): tamaño del gráfico (ancho, alto) 
    """
    plt.figure(figsize=tamaño)
    
    for i in range(filas * columnas):
        if i >= len(imagenes):
            break
        plt.subplot(filas, columnas, i + 1)
        plt.imshow(imagenes[i])
    
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)

