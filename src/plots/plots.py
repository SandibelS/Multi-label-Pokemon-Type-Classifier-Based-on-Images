import matplotlib.pyplot as plt
import numpy as np

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
    plt.close()



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
    plt.close()



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
    plt.close()


def plot_class_distribution(y, classes, path):
    """
    Genera y guarda un gráfico de barras con la cantidad de ocurrencias por clase en un arreglo multi-hot.

    Parámetros:
    - y (np.ndarray): arreglo de vectores multi-hot de forma (n_samples, n_classes)
    - classes (list of str): nombres de las clases correspondientes a cada columna de y
    - path (str): ruta completa (incluyendo nombre de archivo) donde se guardará el gráfico

    Retorna:
    - None
    """
    # Sumar ocurrencias por clase
    class_counts = np.sum(y, axis=0)

    # Crear gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(classes, class_counts, color='skyblue')
    plt.xlabel('Clases')
    plt.ylabel('Número de ocurrencias')
    plt.title('Distribución de clases en el conjunto multi-hot')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Guardar gráfico
    plt.savefig(path)
    plt.close()

def plot_loss_and_accuracy_keras(history,TopKCategoricalAccuracy = False, path_loss = 'src/plots/loss_keras.png', path_accuracy = 'src/plots/accuracy_keras.png'):

    # loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Función de pérdida')
    plt.legend()
    plt.show()
    plt.savefig(path_loss)
    plt.close()

    # Accuracy
    if ( TopKCategoricalAccuracy):
        plt.plot(history.history['top_k_categorical_accuracy'], label='Train Top-2 Accuracy')
        plt.plot(history.history['val_top_k_categorical_accuracy'], label='Val Top-2 Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Top-2 Accuracy')
        plt.title('Evolución de Top-2 precisión')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(path_accuracy)

    else:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Precisión')
        plt.legend()
        plt.show()
        plt.savefig(path_accuracy)

    plt.close()


def plot_cnn_from_scratch_metrics(model, path_loss='src/plots/loss_scratch.png', path_accuracy = 'src/plots/accuracy_scratch.png'):

    epochs = range(1, len(model.train_losses) + 1)

    # Loss

    plt.plot(epochs, model.train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, model.val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Evolución de la pérdida')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(path_loss)
    plt.close()


    # Precisión
    plt.plot(epochs, model.train_accuracies, label='Train Top-2 Accuracy', color='blue')
    plt.plot(epochs, model.val_accuracies, label='Val Top-2 Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Top-2 Accuracy')
    plt.title('Evolución Top-2 precisión')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(path_accuracy)
    plt.close()


