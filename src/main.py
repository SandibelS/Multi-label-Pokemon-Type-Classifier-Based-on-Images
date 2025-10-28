import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split


from preprocess import remove_parentheses_from_folders_and_files, preproccess_images
from plots.plots import plot_class_distribution, plot_loss_and_accuracy_keras, plot_cnn_from_scratch_metrics

from cnn_v1.cnn import CNN_from_Scratch
from cnn_keras.cnn_keras import CNN_keras

def get_classes(csv_path : str, index : int):
    df1 = pd.read_csv(csv_path)
    return np.array(df1.columns[index:])


def main():

    #------------------------------------ VARIABLES DE FLUJO -------------------------------------------#


    parser = argparse.ArgumentParser(description="Entrenamiento de clasificador de tipos de Pokémon")

    # Argumentos opcionales con valores por defecto
    parser.add_argument('--mode', type=int, default=1, help='Modo: From-Scratch (0) o Keras (1)')
    parser.add_argument('--input_resize', type=int, default=32, help='Tamaño de redimensionamiento de imágenes')
    parser.add_argument('--input_channels', type=int, default=3, help='Número de canales de entrada (RGB = 3)')
    parser.add_argument('--csv_path', type=str, default='data/dataset_of_32k_pokemon_Images_and_csv_json/pokemon_types_one_hot.csv', help='Ruta al CSV limpio')
    # parser.add_argument('--preserve_class_distribution', action='store_true', help='Preservar distribución de clases en el split')
    parser.add_argument('--preserve_class_distribution', type=bool, default=True, help='Preservar distribución de clases en el split')
    parser.add_argument('--test_size', type=float, default=0.3, help='Proporción del conjunto de validación')
    parser.add_argument('--model_id', type=int, default=0, help='ID del modelo a usar')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Tasa de aprendizaje')
    parser.add_argument('--epochs', type=int, default=20, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')

    args = parser.parse_args()

    # Variables de flujo

    # # Modo: From-Scratch - 0 , Keras - 1 (Por default en 1)
    mode = args.mode

    # Tamaño a redimensionar las imagenes, por defecto en 32
    input_resize = args.input_resize

    # Canales de la imagen, por defecto en 3 (RGB)
    input_channels = args.input_channels

    # Path al csv limpio (Solo nombres de pokemones y sus tipos en formato multi-hot)
    csv_path = args.csv_path

    # Variable para decidir el tipo de división para los conjuntos de entrenamiento y validación
    # para preservar la distribucion de clases
    preserve_class_distribution = args.preserve_class_distribution

    # Tamaño del conjunto de validación (El tam de set de entrenamiento se infiere)
    test_size = args.test_size

    # Id del modelo a usar 
    model_id = args.model_id

    # Taza de aprendizaje
    learning_rate = args.learning_rate

    # Cantidad de epocas
    epochs = args.epochs

    # Tamaño del batch
    batch_size = args.batch_size

    # Obtener clases
    classes = get_classes(csv_path, 1)

    print()
    # Mostrar configuración
    print("Configuración del entrenamiento:")
    print(f"Modo: {mode}")
    print(f"Redimensionamiento: {input_resize}px")
    print(f"Canales: {input_channels}")
    print(f"CSV: {csv_path}")
    print(f"Preservar distribución: {preserve_class_distribution}")
    print(f"Test size: {test_size}")
    print(f"Modelo ID: {model_id}")
    print(f"Learning rate: {learning_rate}")
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch_size}")
    print()


    # Clases (Los tipos de pokemon)
    classes = get_classes(csv_path, 1)


    #------------------------------------ PRE-PROCESAMIENTO --------------------------------------------#

    remove_parentheses_from_folders_and_files('data/dataset_of_32k_pokemon_Images_and_csv_json/Pokemon_Images_DB/Pokemon_Images_DB')

    # Una lista de algunos pokemones con casos especiales
    exceptions = [
                    'Castform Sunny Form','Castform Rainy Form', 'Castform Snowy Form', # Solo hay una carpeta de Castform
                    'Eternatus Eternamax', # solo hay una carpeta Eternatus
                    'Nidoran♀ (female) Nidoran♀', # solo hay una carpeta Nidoran♀
                    'Nidoran♂ (male) Nidoran♂',   # solo hay Nidoran♂
                    'Ursaluna Bloodmoon', # solo hay Ursaluna

                    "Farfetch'd",   # le cambiaron el nombre a Farfetch_d, hay que ajustarlo manualmente en el csv
                    "Galarian Farfetch'd",
                    "Oricorio Pa'u Style",
                    "Sirfetch'd",
                    
                    'Dudunsparce Three-Segment Form',   # No esta. Solo esta la carpeta Dudunsparce Two-Segment Form
                    'Maushold Family of Three',         # No esta. Solo esta la carptea Maushold Family of Four

                    'Gouging Fire',  # No esta, hay que revisar si en las otras carpetas estan
                    'Gourgeist Average Size',
                    'Gourgeist Small Size',
                    'Gourgeist Large Size',
                    'Gourgeist Super Size',
                    'Hydrapple',
                    'Iron Boulder',
                    'Ogerpon Wellspring Mask',
                    'Ogerpon Hearthflame Mask',
                    'Ogerpon Cornerstone Mask',
                    'Pumpkaboo Average Size',
                    'Pumpkaboo Small Size',
                    'Pumpkaboo Large Size',
                    'Pumpkaboo Super Size',
                    'Own Tempo Rockruff',
                    'Sinistcha',
                    'Squawkabilly Blue Plumage',
                    'Squawkabilly Yellow Plumage',
                    'Squawkabilly White Plumage',
                    'Terapagos Stellar Form',

                    'Mime Jr.', # la carpeta no tiene ese punto (es parte del nombre o solo es un error?)

                    'Partner Eevee',        # Esta es Eevee Partner Eevee, este es un problema general 
                    'Ash-Greninja',         # Greninja Ash-Greninja
                    'Hoopa Confined',       # Hoopa Hoopa Confined
                    "Hoopa Unbound",        # Hoopa Hoopa Unbound
                    'White Kyurem',         # Kyurem White Kyurem
                    'Black Kyurem',         # Kyurem Black Kyurem
                    'Dusk Mane Necrozma',   # Necrozma Dusk Mane Necrozma
                    'Dusk Wings Necrozma',  # Necrozma Dusk Wings Necrozma
                    'Dawn Wings Necrozma',  # Necrozma Dawn Wings Necrozma 
                    'Ultra Necrozma',       # Necrozma Ultra Necrozma
                    'Partner Pikachu',      # Pikachu Partner Pikachu
                    'Heat Rotom',           # Rotom Heat Rotom
                    'Wash Rotom',           # Rotom Wash Rotom
                    'Frost Rotom',          # Rotom Frost Rotom
                    'Fan Rotom',            # Rotom Fan Rotom
                    'Mow Rotom',            # Rotom Mow Rotom

                    'Type: Null', # Supongo que error de la limpieza del dataset
                ]

    # Obetenemos X (representacion en matrices de las imagenes) y su target y (Vectores multi-hot que representan sus tipos)
    X, y = preproccess_images(  'data/dataset_of_32k_pokemon_Images_and_csv_json/Pokemon_Images_DB/Pokemon_Images_DB', 
                                'data/dataset_of_32k_pokemon_Images_and_csv_json/pokemon_types_one_hot.csv',
                                input_resize, 
                                exceptions=exceptions,
                                only_multi_label=False
                            )

    print()
    print("X shape :", X.shape)
    print("y shape :", y.shape)

    if (preserve_class_distribution):
        X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size)
    
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size)
    

    print()
    print('Classes: ', classes)
    print()


    # Visualizar la distribucion de clases tanto en el set de entrenamiento como en el de validación
    plot_class_distribution(y_train, classes, 'src/plots/y_train_class_distribution.png')
    plot_class_distribution(y_test, classes,  'src/plots/y_test_class_distribution.png')

    #------------------------------------ PROCESAMIENTO -----------------------------------------------#

    if (mode == 1):

        model_keras = CNN_keras(input_resize, input_channels)

        # Entrenamiento
        history = model_keras.train(model_id, X_train, y_train, X_test, y_test, batch_size=batch_size, epochs=epochs)

        # Evaluacion
        loss, accuracy = model_keras.evaluate(model_id, X_test, y_test)

        print()
        print(f"Loss in test: {loss:.4f}")
        print(f"Accuracy in test: {accuracy:.4f}")
        print()

        # Graficar accuracy y loss del entrenamiento
        plot_loss_and_accuracy_keras(history, 
                                     path_loss=f'src/plots/loss_keras_model_{model_id}.png',
                                     path_accuracy=f'src/plots/accuracy_keras_model_{model_id}.png',
                                     TopKCategoricalAccuracy=True)
        

    elif (mode == 0):
        #
        model_from_scratch = CNN_from_Scratch()

        X_train = np.transpose(X_train, (0, 3, 1, 2)) 
        X_test = np.transpose(X_test, (0, 3, 1, 2)) 

        model_from_scratch.train_multi_label(model_id, X_train, y_train, X_test, y_test, learning_rate, epochs, batch_size)

        loss, accuracy = model_from_scratch.evaluate_multi_label(model_id, X_test, y_test, batch_size)

        print()
        print(f"Loss in test: {loss:.4f}")
        print(f"Accuracy in test: {accuracy:.4f}")
        print()

        plot_cnn_from_scratch_metrics(model_from_scratch, 
                                      f'src/plots/loss_scratch_model_{model_id}.png',
                                      f'src/plots/accuracy_scratch_model_{model_id}.png')

    
    else:
        print("Error indicando versión del modelo")
        print("Consejo: ")
        print("\t0 para CNN from Scratch (Lento)")
        print("\t1 para CNN Keras")

main()