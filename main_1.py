import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Установка стиля для графиков
sns.set_theme(style="whitegrid")

# Путь к файлу данных
data_file = 'data/movies.csv'
output_dir = 'output_data'
os.makedirs(output_dir, exist_ok=True)

# 1. Размер файла на диске
file_size = os.path.getsize(data_file) / (1024 ** 2)  # Размер в МБ
print(f"Размер файла на диске: {file_size:.2f} МБ")

# 2. Определение количества строк в файле
with open(data_file, 'r', encoding='utf-8') as f:
    row_count = sum(1 for row in f) - 1  # Вычитаем 1, чтобы исключить заголовок
print(f"Количество строк в файле: {row_count}")

# 3. Анализ использования памяти по колонкам без загрузки всего файла

chunk_size = 100000

# Предварительно определим типы данных
initial_types = {}

# Используем первый чанк для определения типов данных
first_chunk = pd.read_csv(data_file, nrows=chunk_size, low_memory=False)

# Определяем типы данных для дальнейшего чтения
for col in first_chunk.columns:
    if col in ['votes', 'gross_income']:
        # Оставляем эти колонки как 'object' и будем обрабатывать их вручную
        initial_types[col] = 'object'
    elif first_chunk[col].dtype == 'object':
        initial_types[col] = 'object'
    else:
        initial_types[col] = first_chunk[col].dtype

# Анализ использования памяти по колонкам
memory_usage_per_column = {}
data_types = None

for chunk in pd.read_csv(data_file, chunksize=chunk_size, dtype=initial_types, low_memory=False):
    # Преобразуем 'votes' и 'gross_income' в числовые значения после удаления запятых
    chunk['votes'] = pd.to_numeric(chunk['votes'].str.replace(',', ''), errors='coerce')
    chunk['gross_income'] = pd.to_numeric(chunk['gross_income'].str.replace(',', ''), errors='coerce')

    # Заменяем нулевые значения на NaN без inplace
    chunk['gross_income'] = chunk['gross_income'].replace(0, np.nan)

    # Для первого чанка определяем типы данных
    if data_types is None:
        data_types = chunk.dtypes

    # Вычисляем использование памяти для этого чанка
    mem_usage = chunk.memory_usage(deep=True, index=False)
    # Суммируем использование памяти по колонкам
    for col in mem_usage.index:
        memory_usage_per_column[col] = memory_usage_per_column.get(col, 0) + mem_usage[col]

total_memory = sum(memory_usage_per_column.values())
column_memory_df = pd.DataFrame({
    'Column': list(memory_usage_per_column.keys()),
    'Memory Usage (bytes)': list(memory_usage_per_column.values()),
    'Memory Usage (%)': [mem / total_memory * 100 for mem in memory_usage_per_column.values()],
    'Data Type': [str(data_types[col]) for col in memory_usage_per_column.keys()]
})

# Вывод информации
print("\nСтатистика по колонкам до оптимизации:")
print(column_memory_df)

# Сохранение статистики
column_memory_df.to_json(f'{output_dir}/column_memory_stats.json', orient='records', force_ascii=False, indent=4)

# 4. Оптимизация типов данных

optimized_types = initial_types.copy()

for col in column_memory_df['Column']:
    if col in ['votes', 'gross_income']:
        continue  # Мы обработаем их отдельно после чтения (см. ниже)
    elif initial_types[col] == 'object':
        # Проверим, можно ли преобразовать колонку в категориальную
        num_unique_values = 0
        num_total_values = 0
        for chunk in pd.read_csv(data_file, usecols=[col], chunksize=chunk_size, low_memory=False):
            num_unique_values += chunk[col].nunique()
            num_total_values += len(chunk[col])
        if num_unique_values / num_total_values < 0.5:
            optimized_types[col] = 'category'
            print(f"Колонка '{col}' будет преобразована в 'category'. Уникальных значений: {num_unique_values}, Всего значений: {num_total_values}")
    elif 'int' in str(initial_types[col]) or 'float' in str(initial_types[col]):
        if 'int' in str(initial_types[col]):
            optimized_types[col] = 'Int32'
            print(f"Колонка '{col}' будет понижена до 'Int32'")
        else:
            optimized_types[col] = 'float32'
            print(f"Колонка '{col}' будет понижена до 'float32'")

# 5. Повторное чтение данных с оптимизированными типами и расчет использования памяти

new_memory_usage_per_column = {}
data_types_after = None

for chunk in pd.read_csv(data_file, dtype=optimized_types, chunksize=chunk_size, low_memory=False):
    # Преобразуем 'votes' и 'gross_income' в числовые значения после удаления запятых
    chunk['votes'] = pd.to_numeric(chunk['votes'].str.replace(',', ''), errors='coerce', downcast='integer')
    chunk['gross_income'] = pd.to_numeric(chunk['gross_income'].str.replace(',', ''), errors='coerce', downcast='float')

    # Заменяем нулевые значения на NaN без inplace
    chunk['gross_income'] = chunk['gross_income'].replace(0, np.nan)

    # После преобразования типов для 'votes' и 'gross_income', обновляем их типы в chunk
    chunk['votes'] = chunk['votes'].astype('Int32')
    chunk['gross_income'] = chunk['gross_income'].astype('float32')

    # Обновляем типы данных
    if data_types_after is None:
        data_types_after = chunk.dtypes

    mem_usage = chunk.memory_usage(deep=True, index=False)
    for col in mem_usage.index:
        new_memory_usage_per_column[col] = new_memory_usage_per_column.get(col, 0) + mem_usage[col]

new_total_memory = sum(new_memory_usage_per_column.values())
new_column_memory_df = pd.DataFrame({
    'Column': list(new_memory_usage_per_column.keys()),
    'Memory Usage (bytes)': list(new_memory_usage_per_column.values()),
    'Memory Usage (%)': [mem / new_total_memory * 100 for mem in new_memory_usage_per_column.values()],
    'Data Type': [str(data_types_after[col]) for col in new_memory_usage_per_column.keys()]
})

# Вывод информации
print("\nСтатистика по колонкам после оптимизации:")
print(new_column_memory_df)

# Сохранение статистики
new_column_memory_df.to_json(f'{output_dir}/column_memory_stats_optimized.json', orient='records', force_ascii=False, indent=4)

# 6. Работа с выбранными 11 колонками

selected_columns = [
    'id',
    'name',
    'year',
    'rating',
    'genre',
    'votes',
    'gross_income',
    'directors_name',
    'stars_name',
    'description',
    'duration'  
]

selected_types = {col: optimized_types.get(col, 'object') for col in selected_columns}
# Убедимся, что 'votes' и 'gross_income' читаются как 'object'
selected_types['votes'] = 'object'
selected_types['gross_income'] = 'object'
selected_types['duration'] = 'object'  # Читаем 'duration' как строку

chunks = []
for chunk in pd.read_csv(data_file, usecols=selected_columns, dtype=selected_types, chunksize=chunk_size, low_memory=False):
    # Преобразуем 'votes' и 'gross_income' в числовые значения после удаления запятых
    chunk['votes'] = pd.to_numeric(chunk['votes'].str.replace(',', ''), errors='coerce', downcast='integer')
    chunk['gross_income'] = pd.to_numeric(chunk['gross_income'].str.replace(',', ''), errors='coerce', downcast='float')

    # Заменяем нулевые значения на NaN без inplace
    chunk['gross_income'] = chunk['gross_income'].replace(0, np.nan)

    # После преобразования типов для 'votes' и 'gross_income', обновляем их типы в chunk
    chunk['votes'] = chunk['votes'].astype('Int32')
    chunk['gross_income'] = chunk['gross_income'].astype('float32')

    # Удаляем подстроку ' min' из строки и обрезаем возможные пробелы
    chunk['duration_clean'] = chunk['duration'].str.replace(' min', '').str.strip()
    # Преобразование очищенных строк в числовой формат
    chunk['duration_numeric'] = pd.to_numeric(chunk['duration_clean'], errors='coerce')

    chunks.append(chunk)

df_opt = pd.concat(chunks, ignore_index=True)
df_opt.to_csv(f'{output_dir}/movies_subset.csv', index=False)

# Преобразуем 'year' в числовой формат перед вычислениями
df_opt['year'] = pd.to_numeric(df_opt['year'], errors='coerce')

# 7. Построение графиков

# a) Распределение рейтингов фильмов
plt.figure(figsize=(10, 6))
sns.histplot(df_opt['rating'].dropna(), bins=20, kde=True)
plt.title('Распределение рейтингов фильмов')
plt.xlabel('Рейтинг')
plt.ylabel('Количество фильмов')
plt.savefig(f'{output_dir}/rating_distribution.png')
plt.show()

# b) Топ-10 жанров по количеству фильмов
plt.figure(figsize=(12, 8))
# Разбиваем жанры, если они перечислены через запятую
df_opt['genre_list'] = df_opt['genre'].str.split(',')
# Раскрываем списки жанров в отдельные строки
df_genres = df_opt.explode('genre_list')
df_genres['genre_list'] = df_genres['genre_list'].str.strip()
genre_counts = df_genres['genre_list'].value_counts().head(10)
sns.barplot(x=genre_counts.values, y=genre_counts.index)
plt.title('Топ-10 жанров по количеству фильмов')
plt.xlabel('Количество фильмов')
plt.ylabel('Жанр')
plt.savefig(f'{output_dir}/top_genres.png')
plt.show()

# c) Зависимость между рейтингом и количеством голосов
plt.figure(figsize=(10, 6))
sns.scatterplot(x='rating', y='votes', data=df_opt)
plt.title('Зависимость между рейтингом и количеством голосов')
plt.xlabel('Рейтинг')
plt.ylabel('Количество голосов')
plt.savefig(f'{output_dir}/rating_votes_correlation.png')
plt.show()

# d) Зависимость дохода от рейтинга
plt.figure(figsize=(10, 6))
# Отфильтруем данные, где 'gross_income' и 'rating' не являются NaN
df_income_rating = df_opt.dropna(subset=['gross_income', 'rating'])
sns.scatterplot(x='rating', y='gross_income', data=df_income_rating)
plt.title('Зависимость дохода от рейтинга фильмов')
plt.xlabel('Рейтинг')
plt.ylabel('Доход (Gross Income)')
plt.savefig(f'{output_dir}/gross_income_vs_rating.png')
plt.show()

# e) Круговая диаграмма распределения фильмов по продолжительности

# Проверяем наличие данных о продолжительности
df_opt['duration_clean'] = df_opt['duration'].str.replace(' min', '').str.strip()
df_opt['duration_numeric'] = pd.to_numeric(df_opt['duration_clean'], errors='coerce')

# Определим категории продолжительности
bins = [0, 90, 120, 150, np.inf]
labels = ['До 90 мин', '90-120 мин', '120-150 мин', 'Более 150 мин']
df_opt['duration_category'] = pd.cut(df_opt['duration_numeric'], bins=bins, labels=labels)

# Посчитаем количество фильмов в каждой категории
duration_counts = df_opt['duration_category'].value_counts().sort_index()

# Проверяем данные
print(duration_counts)

# Построим круговую диаграмму
plt.figure(figsize=(8, 8))
plt.pie(duration_counts.values, labels=duration_counts.index.astype(str), autopct='%1.1f%%')
plt.title('Распределение фильмов по продолжительности')
plt.savefig(f'{output_dir}/duration_distribution.png')
plt.show()

# f) Топ-10 режиссеров по количеству фильмов
plt.figure(figsize=(12, 8))
df_opt['directors_list'] = df_opt['directors_name'].str.split(',')
df_directors = df_opt.explode('directors_list')
df_directors['directors_list'] = df_directors['directors_list'].str.strip()
director_counts = df_directors['directors_list'].value_counts().head(10)
sns.barplot(x=director_counts.values, y=director_counts.index)
plt.title('Топ-10 режиссеров по количеству фильмов')
plt.xlabel('Количество фильмов')
plt.ylabel('Режиссер')
plt.savefig(f'{output_dir}/top_directors.png')
plt.show()

# g) Корреляционная матрица между числовыми переменными
plt.figure(figsize=(8, 6))
numeric_cols = ['rating', 'votes', 'gross_income', 'duration_numeric']
corr_matrix = df_opt[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.savefig(f'{output_dir}/correlation_matrix.png')
plt.show()
