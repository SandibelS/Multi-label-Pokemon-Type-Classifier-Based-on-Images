import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Cargo el dataset
df = pd.read_csv("../data/dataset_of_32k_pokemon_Images_and_csv_json/pokemonDB_dataset.csv")

pokemon_types = df["Type"].apply(lambda x: x.split(",")) # Convierto los tipos de cada pokemon en una lista

# Creo una nueva fila por cada tipo de un pokemon y quito los espacios en blanco en el nombre del tipo
pokemon_types = pokemon_types.explode().apply(lambda x: x.strip())
pokemon_types = list(pokemon_types.unique()) # Obtengo los tipos unicos de los pokemon

df2 = pd.DataFrame(columns=pokemon_types) # Creo un nuevo dataframe que tendrá las columnas de los tipos de pokemon

for index, row in df.iterrows(): # Itero sobre las filas del dataframe original
    for element in row.iloc[1].split(","): # Obtengo los tipos de un pokemon
       df2.at[index, element.strip()] = 1 # Asigno un 1 en la columna del tipo de mi nuevo dataframe para el pokemon correspondiente

df2 = df2.fillna(0).infer_objects(copy=False) # Relleno los valores nulos con 0

df_one_hot = pd.concat([df["Pokemon"], df2], axis=1) # Concateno los dos dataframes
df_one_hot.to_csv("../data/dataset_of_32k_pokemon_Images_and_csv_json/pokemon_types_one_hot.csv", index=False) # Guardo el nuevo dataframe en un csv

# Contadores de pokemons con uno o dos tipos
one_type = 0 
two_types = 0

for index, row in df_one_hot.iterrows(): # Itero sobre las filas del dataframe one hot 
    if row.iloc[1:].sum() == 1: # Si la suma de los tipos es 1, entonces tiene un solo tipo
        one_type += 1
    else:
        two_types += 1 # Si no, tiene dos tipos

print(f"Pokemons with one type: {one_type}")
print(f"Pokemons with two types: {two_types}")

# Gráfico de barras para la cantidad de pokemons con uno o dos tipos
plt.figure(figsize=(5,4))
plt.bar(['Un tipo', 'Dos tipos'], [one_type, two_types], color=['skyblue', 'orange'])
plt.ylabel('Cantidad de Pokémon')
plt.title('Pokémon por cantidad de tipos')
plt.savefig('./plots/pokemon_one_two_types_bar.png')  # Guarda la figura como un archivo PNG
plt.show()

for col in df_one_hot.columns[1:]: # Itero sobre las columnas del dataframe one hot (excepto la primera columna que es el nombre del pokemon)
    print(f"{col}: {df_one_hot[col].sum()}") # Imprimo la cantidad de pokemons por tipo

# Gráfico de barras para la cantidad de pokemons por tipo
labels = df_one_hot.columns[1:]
sizes = [df_one_hot[col].sum() for col in labels]

# Genera una lista de colores distintos usando un mapa de colores de matplotlib
colors = cm.get_cmap('tab20', len(labels))(np.arange(len(labels)))
plt.figure(figsize=(12,6))
plt.bar(labels, sizes, color=colors)
plt.ylabel('Cantidad de Pokémon')
plt.xlabel('Tipo')
plt.title('Cantidad de Pokémon por tipo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plots/pokemon_types_bar.png')  # Guarda la figura como un archivo PNG
plt.show()