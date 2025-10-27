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

    # Modo: From-Scratch - 0 , Keras - 1 (Por default en 1)
    mode = 0

    # Tamaño a redimensionar las imagenes, por defecto en 128
    input_resize = 32

    # Canales de la imagen, por defecto en 3 (RGB)
    input_channels = 3

    # Path al csv limpio (Solo nombres de pokemones y sus tipos en formato multi-hot)
    csv_path = 'data/dataset_of_32k_pokemon_Images_and_csv_json/pokemon_types_one_hot.csv'

    # Clases (Los tipos de pokemon)
    classes = get_classes(csv_path, 1)

    # Variable para decidir el tipo de división para los conjuntos de entrenamiento y validación
    # para preservar la distribucion de clases
    preserve_class_distribution = False

    # Tamaño del conjunto de validación (El tam de set de entrenamiento se infiere)
    test_size = 0.3

    # Id del modelo a usar 
    model_id = 0

    # Taza de aprendizaje
    learning_rate = 0.0001

    # Cantidad de epocas
    epochs = 20

    # Tamaño del batch
    batch_size = 32


    #------------------------------------ PRE-PROCESAMIENTO --------------------------------------------#

    # Esto tal vez deberia ir en otra parte
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