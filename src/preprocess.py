import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os


def remove_parentheses_from_folders_and_files(path : str):

    # Iterar sobre todos los elementos en la carpeta
    for name in os.listdir(path):

        full_path = os.path.join(path, name)

        # Verificar si es una carpeta
        if os.path.isdir(full_path):

            new_name = name.replace('(', '').replace(')', '')
            new_path = os.path.join(path, new_name)

            # Renombrar si el nombre cambió
            if new_name != name:

                os.rename(full_path, new_path)
                print(f'Renombrado: {name} -> {new_name}')
            
            # Recorrer todos los archivos dentro de la subcarpeta
            for filename in os.listdir(full_path):

                file_path = os.path.join(full_path, filename)

                # Renombrar si contiene paréntesis
                new_filename = filename.replace('(', '').replace(')', '')
                new_file_path = os.path.join(full_path, new_filename)

                if new_filename != filename:
                    os.rename(file_path, new_file_path)
                    print(f'Renombrado: {filename} -> {new_filename}')


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


def preproccess_images(images_path : str, csv_path : str, resize : int, channels = 3, exceptions = []):

    # Leemos el archivo de csv limpio para saber el nombre de los pokemones y asi su carpeta
    df1 = pd.read_csv(csv_path)

    images = []

    index_discard = []
    
    # Por cada fila vamos a buscar la imagen del dataset
    for i in tqdm(range(df1.shape[0])):

        if(df1['Pokemon'][i] in exceptions):
            index_discard.append(i)
            continue

        img = image.load_img(f"{images_path}/{df1['Pokemon'][i]}/{df1['Pokemon'][i]}.png",target_size=(resize, resize, channels))
        img = image.img_to_array(img)
        img = img/255
        images.append(img)

    X = np.array(images)

    print(X.shape)
    plt.imshow(X[2])
    plt.savefig('src/plots/image_example.png')

    image_grid(X[0:32])

    y = np.array( (df1.drop(['Pokemon'], axis=1)).drop(index_discard, axis=0) )

    return X, y


# Pre-procesamiento

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

X, y = preproccess_images( 'data/dataset_of_32k_pokemon_Images_and_csv_json/Pokemon_Images_DB/Pokemon_Images_DB', 
                                        'data/dataset_of_32k_pokemon_Images_and_csv_json/pokemon_types_one_hot.csv',
                                        400, exceptions=exceptions)


