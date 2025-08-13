# Classificação de dados Iris com visualização da fronteira de decisão

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA  # Para redução de dimensionalidade

# 1) Carregar o conjunto de dados Iris
#    - 150 amostras de flores, 3 espécies
#    - Cada amostra possui 4 características: comprimento e largura da sépala e pétala
iris = load_iris()
X = iris.data      # Dados de entrada (features)
y = iris.target    # Rótulos/classes

# 2) Reduzir dimensionalidade para 2 componentes principais (PCA)
#    - Facilita visualização 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3) Dividir dados em treino (70%) e teste (30%)
#    - random_state garante resultados reproduzíveis
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42
)

# 4) Treinar o classificador Random Forest com 100 árvores
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 5) Fazer previsões no conjunto de teste
y_pred = modelo.predict(X_test)

# 6) Avaliar o desempenho do modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7) Preparar grade para visualização da fronteira de decisão
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.2),
    np.arange(y_min, y_max, 0.2)
)

# 8) Prever a classe para cada ponto da grade
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 9) Plotar a fronteira de decisão e os pontos de treino
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4)              # Áreas coloridas pela classe prevista
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20)  # Pontos de treino
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Fronteira de Decisão do Modelo Random Forest')
plt.show()

# 10) Testar previsão para nova amostra (dados originais com 4 características)
novo = [[5.1, 4.2, 3.9, 0.2]]

# Aplicar a transformação PCA para a nova amostra
novo_pca = pca.transform(novo)

# Fazer a predição usando o modelo treinado
predicao = modelo.predict(novo_pca)

# Mostrar resultado
print("Classe prevista (índice):", predicao)
print("Classe prevista (nome):", iris.target_names[predicao][0])
