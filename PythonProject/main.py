import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. GERANDO UM CONJUNTO DE DADOS SIMULADO (Comportamento de Clientes)
np.random.seed(42)
# Criando 3 grupos distintos de clientes de e-commerce
grupo1 = np.random.normal(loc=[30, 20], scale=[5, 5], size=(30, 2))  # Renda Baixa, Gasto Baixo
grupo2 = np.random.normal(loc=[70, 80], scale=[7, 7], size=(40, 2))  # Renda Alta, Gasto Alto
grupo3 = np.random.normal(loc=[100, 30], scale=[8, 6], size=(30, 2)) # Renda Alta, Gasto Baixo

# NOVO GRUPO: Renda Baixa (loc=25), Gasto Alto (loc=75)
grupo4 = np.random.normal(loc=[25, 75], scale=[5, 5], size=(25, 2))

# Concatenando os dados em um único DataFrame
dados = np.vstack((grupo1, grupo2, grupo3, grupo4))
df = pd.DataFrame(dados, columns=['Renda_Anual_K', 'Score_Gastos'])

print("--- Primeiras linhas do conjunto de dados ---")
print(df.head())

# 2. CONFIGURANDO E TREINANDO O MODELO K-MEANS
# Sabemos visualmente que idealmente são 4 clusters (K=4)
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42, n_init=10)

# Treinamento do modelo
df['Cluster'] = kmeans.fit_predict(df)

# Obtendo os Centroides (o "centro" de cada grupo
centroides = kmeans.cluster_centers_

print("\n--- Posição dos Centroides finais ---")
print(f"Centroides:\n{centroides}")

# 3. PLOTANDO OS RESULTADOS
plt.figure(figsize=(8, 6))

# Cores para os clusters
cores = ['red', 'blue', 'green', 'yellow']

for i in range(num_clusters):
    # Plota os pontos de cada cluster
    plt.scatter(df[df['Cluster'] == i]['Renda_Anual_K'], 
                df[df['Cluster'] == i]['Score_Gastos'], 
                s=50, c=cores[i], label=f'Cluster {i+1}')

# Plota os centroides
plt.scatter(centroides[:, 0], centroides[:, 1], s=200, c='black', marker='X', label='Centroides')

plt.title('Segmentação de Clientes usando K-Means')
plt.xlabel('Renda Anual (em milhares de R$)')
plt.ylabel('Pontuação de Gastos (1-100)')
plt.legend()
plt.grid(True)

# Salva o gráfico para o repositório do GitHub
plt.savefig('resultado_clusters.png')
plt.show()