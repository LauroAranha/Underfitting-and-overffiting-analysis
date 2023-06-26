import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""Lógica do altoritmo:"""

dados = pd.read_csv('dados.csv')

verticeX = np.array(dados['altura'])
verticeY = np.array(dados['peso'])
classes = np.array(dados['classe'])

# Separar dados para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(verticeX, verticeY, test_size=0.3, random_state=42)

# Regressão Linear no conjunto de treinamento
poliA, poliB = np.polyfit(X_train, y_train, 1)
coefRelacaoAnimal = np.corrcoef(X_train, y_train)[0, 1]
y_pred_train = poliA * X_train + poliB

# Regressão Linear no conjunto de teste
y_pred_test = poliA * X_test + poliB

# Curva Polinomial
coefs_polinomial = np.polyfit(X_train, y_train, 2)
altura_polinomial = np.linspace(min(verticeX), max(verticeX), 100)
peso_polinomial = np.polyval(coefs_polinomial, altura_polinomial)

# Ajuste do tamanho da figura
plt.figure(figsize=(12, 6))

# Plotando os resultados - Regressão Linear
plt.subplot(1, 2, 1)  # Cria o primeiro gráfico
plt.scatter(X_train, y_train, c='blue', label='Dados de Treinamento')
plt.scatter(X_test, y_test, c='red', label='Dados de Teste')
plt.plot(X_train, y_pred_train, color='red', label='Regressão Linear (Treinamento)')
plt.plot(X_test, y_pred_test, color='orange', label='Regressão Linear (Teste)')

plt.text(0.612, 1.1, f'Coeficiente de Correlação (Treinamento) = {coefRelacaoAnimal:.2f}',
         horizontalalignment='right', verticalalignment='bottom',
         transform=plt.gca().transAxes)
plt.text(0.700, 1.05, f'Reta de Regressão (Treinamento): y = {poliA:.2f}x + ({poliB:.2f})',
         horizontalalignment='right', verticalalignment='baseline',
         transform=plt.gca().transAxes)

plt.grid()
plt.legend()

# Plotando os resultados - Curva Polinomial
plt.subplot(1, 2, 2)  # Cria o segundo gráfico
plt.scatter(X_train, y_train, c='blue', label='Dados de Treinamento')
plt.scatter(X_test, y_test, c='red', label='Dados de Teste')
plt.plot(altura_polinomial, peso_polinomial, color='green', label='Curva Polinomial - grau 2')

plt.grid()
plt.legend()

plt.subplots_adjust(left=0, right=0.92, wspace=0.3)  # Ajuste das margens e espaçamento

plt.show()