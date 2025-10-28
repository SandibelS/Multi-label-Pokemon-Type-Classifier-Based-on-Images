# Multi-label-Pokemon-Type-Classifier-Based-on-Images

El objetivo del proyecto es implementar un clasificador multi-etiqueta de imágenes
con una red neuronal convolucional, con el objeto de predecir los tipos de un pokémon dada
su imagen.

# Requirements

- Python 3.10+
- NumPy, Matplotlib, TensorFlow 
- scikit-learn, scikit-multilearn 

## Setup and execution

1. Creación del entorno virtual

Para garantizar un entorno aislado para las dependencias, crea un entorno virtual con el siguiente comando:

```python -m venv nombre_del_entorno```

o

```python3 -m venv nombre_del_entorno```

2. Activación del entorno

Una vez creado el entorno, actívalo con:

```source nombre_del_entorno/bin/activate```

3. Instalación de dependencias

Instala las dependencias necesarias desde requirements.txt usando:

```pip install -r requirements.txt```

4. Ejecución del código

Para ejecutar el codigo:

Primero se realiza la limpieza ejecutando:

```python src/cleaning.py```

o

```python3 src/cleaning.py```

Luego

```python src/main.py [Argumentos]```

o

```python3 src/main.py [Argumentos]```

Donde los argumentos pueden ser:

- mode: define el modo de entrenamiento.  0 para entrenar desde cero y 1 para usar Keras. Por defecto es 1.

- input_resize: tamaño al que se redimensionan las imágenes antes de ser procesadas. Por defecto es 32.

- input_channels: número de canales de entrada. Para imágenes RGB, el valor es 3.

- csv_path: ruta al archivo CSV que contiene las etiquetas multi-hot de los Pokémon. Por defecto apunta al dataset limpio.

- preserve_class_distribution: indica si se debe preservar la distribución de clases al dividir el dataset en entrenamiento y validación. Por defecto es True.

- test_size: proporción del conjunto de validación. Por defecto es 0.3, es decir, 30% del total.

- model_id: ID del modelo a usar, de 0 a 2. Por defecto es 0.

- learning_rate: tasa de aprendizaje para el entrenamiento. Por defecto es 0.0001.

- epochs: número de épocas de entrenamiento. Por defecto es 20.

- batch_size: tamaño del batch para entrenamiento. Por defecto es 32.

## Dataset

Descargue el siguiente dataset, el archivo se debe descomprimir en la carpeta data/ y renombrarlo con el nombre 'dataset_of_32k_pokemon_Images_and_csv_json'

- [Dataset of 32000 Pokemon Images & CSV, JSON. Kaggle.](https://www.kaggle.com/datasets/divyanshusingh369/complete-pokemon-library-32k-images-and-csv/data)



## Structure

Este proyecto está organizado de la siguiente manera:

```
├── data/            # Datos originales (imágenes, CSV, JSON)
├── src/             # Código fuente principal
│ ├── cnn_keras/     # Modelos CNN implementados con Keras
│ │ ├── init.py
│ │ └── cnn_keras.py
│ ├── cnn_v0/       # Implementación de CNN desde cero v0 (NumPy)
│ │ ├── init.py
│ │ ├── cnn.py       # Arquitectura y entrenamiento manual
│ │ ├── pre/         # Scripts de prueba y resultados sobre CIFAR-10
│ │ │ ├── test_cifar-10.py
│ │ │ ├── test_cifar-10_out_0.txt
│ │ │ └── test_cifar-10_out_1.txt
│ │ └── utility_functions.py
│ ├── out/ # Resultados de los modelos
│ │ ├── old/
│ │ ├── keras/
│ │ ├── scratch/
│ │ └── plots/ # Gráficos generados durante el entrenamiento
│ └── main.py # Script principal de ejecución
│ └── preprocess.py  # Funciones de preprocesamiento de imágenes y etiquetas
│ └── cleaning.py    # Limpieza del csv

```

