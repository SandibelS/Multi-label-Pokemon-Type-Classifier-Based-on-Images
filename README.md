# Multi-label-Pokemon-Type-Classifier-Based-on-Images

El objetivo del proyecto es implementar un clasificador multi-etiqueta de imágenes
con una red neuronal convolucional, con el objeto de predecir los tipos de un pokémon dada
su imagen.

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

Para ejecutar el codigo solución de la parte 2, usa:

```python src/main.py```

o

```python3 src/main.py```

## Datasets

- [Dataset of 32000 Pokemon Images & CSV, JSON. Kaggle.](https://www.kaggle.com/datasets/divyanshusingh369/complete-pokemon-library-32k-images-and-csv/data)

- [Pokémon Pokédex: list of Pokémon with stats.](https://pokemondb.net/pokedex/all)


## Structure

Este proyecto está organizado de la siguiente manera:

```
├── data/                  # Carpeta para almacenar datasets
├── src/                   # Código fuente del proyecto
│   ├── plots/             # Scripts relacionados con visualización de datos    
│   ├── CNN.py             # Implementación de una red neuronal convolucional
│   └── main.py            # Script principal para ejecutar el proyecto
├── .gitignore             # Archivos y carpetas excluidos del control de versiones
├── Proyecto I.pdf         # Documento del proyecto (propuesta)
├── README.md              # Documentación del proyecto
└── requirements.txt       # Lista de dependencias necesarias
```

