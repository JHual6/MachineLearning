# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Nodes

def load_data(filepath: str) -> pd.DataFrame:
    """Carga datos desde un archivo CSV."""
    return pd.read_csv(filepath, on_bad_lines='skip')

def print_data_info(df: pd.DataFrame):
    """Imprime la información general del dataframe."""
    print(df.info())

def identify_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica filas con valores nulos y las imprime."""
    valores_nulos = df.isnull().any(axis=1)
    print("Filas con valores nulos:")
    print(df[valores_nulos])
    return df

def convert_rating_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte la columna 'rating' a numérico."""
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    return df

def calculate_average_rating(data: pd.DataFrame) -> pd.DataFrame:
    average_rating = data.groupby('match')['rating'].mean()
    return average_rating.to_frame(name='average_rating')

def fill_null_ratings(df: pd.DataFrame, promedio_rating_match: pd.Series) -> pd.DataFrame:
    """Rellena los valores nulos en 'rating' con la media por 'match'."""
    def fill_nulos_rating(row):
        if pd.isnull(row['rating']):
            if pd.isnull(row['match']):
                return row['rating']
            else:
                return promedio_rating_match.get(row['match'], row['rating'])
        else:
            return row['rating']
    
    df['rating'] = df.apply(fill_nulos_rating, axis=1)
    return df

def check_null_ratings(df: pd.DataFrame):
    """Verifica cuántos valores nulos quedan en 'rating'."""
    print(df['rating'].isnull().sum())

def drop_null_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las filas con valores nulos en la columna 'match'."""
    return df.dropna(subset=['match'])

def identify_null_rows_post_partition(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica filas con valores nulos después del particionamiento."""
    filas_con_nulos = df[df.isnull().any(axis=1)]
    print("Filas con datos nulos en 'aoe_2':")
    print(filas_con_nulos)
    return df

def drop_all_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina todas las filas con cualquier valor nulo."""
    return df.dropna()

def final_null_check(df: pd.DataFrame):
    """Verifica si quedan valores nulos en el dataframe."""
    print(df[df.isnull().any(axis=1)])

def explore_rating(df: pd.DataFrame):
    """Muestra un resumen estadístico de la columna 'rating'."""
    print(df['rating'].describe())

def plot_boxplot(df: pd.DataFrame):
    """Genera un boxplot de la columna 'rating'."""
    plt.figure(figsize=(8, 6))
    df.boxplot(column='rating')
    plt.title('Boxplot de Rating')
    plt.ylabel('Rating')
    plt.grid(False)
    plt.show()

def calculate_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula y filtra los valores atípicos de la columna 'rating' utilizando el IQR."""
    Q1_rating = df['rating'].quantile(0.25)
    Q3_rating = df['rating'].quantile(0.75)
    IQR_rating = Q3_rating - Q1_rating

    lower_bound_rating = Q1_rating - 1.5 * IQR_rating
    upper_bound_rating = Q3_rating + 1.5 * IQR_rating

    return df[(df['rating'] >= lower_bound_rating) & (df['rating'] <= upper_bound_rating)]

def plot_distribution_without_outliers(df: pd.DataFrame):
    """Genera un histograma de la distribución de 'rating' sin outliers."""
    plt.figure(figsize=(10, 6))
    df['rating'].hist(bins=50)
    plt.title('Distribución del Rating (Sin Outliers)')
    plt.xlabel('Rating')
    plt.ylabel('Frecuencia')
    plt.grid(False)
    plt.show()

def check_winner_values(df: pd.DataFrame):
    """Imprime los valores únicos de la columna 'winner'."""
    valores_unicos = df['winner'].unique()
    print("Valores únicos en la columna 'winner':")
    print(valores_unicos)

def transform_winner_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte la columna 'winner' a enteros."""
    df['winner'] = df['winner'].astype(int)
    print(df['winner'].unique())
    return df

def count_civilizations(df: pd.DataFrame) -> pd.Series:
    """Cuenta la frecuencia de cada civilización."""
    return df['civ'].value_counts()

def plot_top_civilizations(civ_counts: pd.Series):
    """Genera un gráfico de barras con las 10 civilizaciones más jugadas."""
    plt.figure(figsize=(12, 8))

    top_10_civilizaciones = civ_counts.head(10)
    colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'yellow', 'cyan']
    top_10_civilizaciones.plot(kind='bar', color=colores)

    plt.title('¿Qué civilización es la más jugada? - Frecuencia del uso de cada civilización (Top 10)')
    plt.xlabel('Civilizaciones')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.show()

def diferencia_entre_civilizaciones(df: pd.DataFrame) -> float:
    """Calcula la diferencia de victorias entre dos civilizaciones."""
    franks_britons = df[df['civ'].isin(['Franks', 'Britons'])]
    conteo_franks_britons = franks_britons['civ'].value_counts()
    diferencia_conteo = conteo_franks_britons['Franks'] - conteo_franks_britons['Britons']
    porcentaje_diferencia = (diferencia_conteo / conteo_franks_britons['Franks']) * 100
    print(f"Diferencia entre la primera civilización más jugada y la segunda: {porcentaje_diferencia:.2f}% ({diferencia_conteo} partidas)")
    return porcentaje_diferencia

def definir_colores_civilizaciones() -> dict:
    """Define un diccionario de colores para las civilizaciones."""
    colores_civilizaciones = {
        'Franks': 'red',
        'Britons': 'blue',
        'Mayans': 'green',
        'Mongols': 'orange',
        'Goths': 'purple',
        'Khmer': 'brown',
        'Persians': 'pink',
        'Aztecs': 'gray',
        'Huns': 'yellow',
        'Lithuanians': 'cyan',
        'Indians': 'pink',
        'Malians': 'black',
        'Slavs': 'slateblue',
        'Incas': 'coral',
        'Magyars': 'orchid',
        'Koreans': 'mistyrose',
        'Teutons': 'lime',
        'Portuguese': 'steelblue',
        'Berbers': 'darkseagreen',
    }
    return colores_civilizaciones

def conteo_matches_validos(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra los matches válidos (donde ambos jugadores están presentes)."""
    conteo_matches = df['match'].value_counts()
    matches_validos = conteo_matches[conteo_matches == 2].index
    df_h2 = df[df['match'].isin(matches_validos)]
    return df_h2

def ordenar_y_filtrar_partidas(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena y filtra las partidas para obtener las civilizaciones ganadoras y perdedoras."""
    df_sorted = df.sort_values(by=['match', 'winner'])
    matches_filtrados2 = df_sorted.groupby('match').filter(lambda x: len(x) == 2 and set(x['winner']) == {0, 1})
    civilizacion_win = matches_filtrados2.loc[matches_filtrados2['winner'] == 1, ['match', 'civ']]
    civilizacion_lose = matches_filtrados2.loc[matches_filtrados2['winner'] == 0, ['match', 'civ']]
    promedio_rating = matches_filtrados2.groupby('match')['rating'].mean().reset_index()
    particion_final = pd.merge(civilizacion_win, civilizacion_lose, on='match', suffixes=('_win', '_lose'))
    particion_final = pd.merge(particion_final, promedio_rating, on='match')
    particion_final.columns = ['match', 'civilizacion_win', 'civilizacion_lose', 'promedio_rating']
    return particion_final

def calcular_diferencias_victorias_derrotas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las diferencias entre victorias y derrotas por civilización."""
    resultados_finales = []
    civilizaciones = pd.concat([df['civilizacion_win'], df['civilizacion_lose']]).unique()

    for civ in civilizaciones:
        for oponente in civilizaciones:
            if civ != oponente:
                victorias = len(df[(df['civilizacion_win'] == civ) & (df['civilizacion_lose'] == oponente)])
                derrotas = len(df[(df['civilizacion_win'] == oponente) & (df['civilizacion_lose'] == civ)])
                diferencia = victorias - derrotas
                rating_promedio = df.loc[
                    ((df['civilizacion_win'] == civ) & (df['civilizacion_lose'] == oponente)) |
                    ((df['civilizacion_win'] == oponente) & (df['civilizacion_lose'] == civ)),
                    'promedio_rating'
                ].mean()
                resultados_finales.append({'civilizacion': civ, 'oponente': oponente, 'diferencia_victorias': diferencia, 'rating_promedio': rating_promedio})

    resultados_df = pd.DataFrame(resultados_finales)
    return resultados_df

def calcular_top_victorias(resultados_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula los top 10 en cuanto a diferencias de victorias."""
    return resultados_df.sort_values('diferencia_victorias', ascending=False).head(10)

def plot_top_victorias(top_10_victorias: pd.DataFrame, colores_civilizaciones: dict):
    """Genera un gráfico de barras de los top 10 en diferencias de victorias."""
    plt.figure(figsize=(12, 8))
    for i in range(len(top_10_victorias)):
        civilizacion = top_10_victorias.iloc[i]['civilizacion']
        color = colores_civilizaciones.get(civilizacion, 'gray')
        plt.bar(
            f"{top_10_victorias.iloc[i]['civilizacion']} vs {top_10_victorias.iloc[i]['oponente']}",
            top_10_victorias.iloc[i]['diferencia_victorias'],
            color=color
        )
    plt.title('¿Qué civilización tiene más victorias contra otras civilizaciones?')
    plt.xlabel('Civilización (vs. Oponente)')
    plt.ylabel('Diferencia de Victorias')
    plt.xticks(rotation=45)
    plt.show()

def calcular_ranking_victorias(resultados_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el ranking de victorias por civilización."""
    return resultados_df.groupby('civilizacion')['diferencia_victorias'].sum().sort_values(ascending=False)

def plot_ranking_victorias(ranking_victorias: pd.DataFrame, colores_civilizaciones: dict):
    """Genera un gráfico de barras del ranking de victorias por civilización."""
    plt.figure(figsize=(12, 8))
    for i in range(len(ranking_victorias)):
        civilizacion = ranking_victorias.index[i]
        color = colores_civilizaciones.get(civilizacion, 'gray')
        plt.bar(civilizacion, ranking_victorias.iloc[i], color=color)
    plt.title('Ranking de Civilizaciones con Más Victorias')
    plt.xlabel('Civilización')
    plt.ylabel('Diferencia de Victorias')
    plt.xticks(rotation=45)
    plt.show()

def calcular_ranking_derrotas(resultados_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el ranking de derrotas por civilización."""
    return resultados_df.groupby('civilizacion')['diferencia_victorias'].sum().sort_values()

def plot_ranking_derrotas(ranking_derrotas: pd.DataFrame, colores_civilizaciones: dict):
    """Genera un gráfico de barras del ranking de derrotas por civilización."""
    plt.figure(figsize=(12, 8))
    for i in range(len(ranking_derrotas)):
        civilizacion = ranking_derrotas.index[i]
        color = colores_civilizaciones.get(civilizacion, 'gray')
        plt.bar(civilizacion, ranking_derrotas.iloc[i], color=color)
    plt.title('Ranking de Civilizaciones con Más Derrotas')
    plt.xlabel('Civilización')
    plt.ylabel('Diferencia de Victorias')
    plt.xticks(rotation=45)
    plt.show()
