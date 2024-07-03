import pandas as pd

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv('train.csv')

# Nombre total de textes
total_texts = len(df)

# Afficher le nombre total de textes
print("Nombre total de textes :", total_texts)

# Compter le nombre de textes pour chaque catégorie
counts = df['category'].value_counts()

# Afficher les résultats
print("\nNombre de textes par catégorie :")
print(counts)
