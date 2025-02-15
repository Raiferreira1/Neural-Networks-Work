import numpy as np
import matplotlib.pyplot as plt

# Gerar dados sintéticos
np.random.seed(0)
x = np.random.uniform(-10, 10, 100)
epsilon = np.random.normal(0, 2, 100)
y = 3 * x + 5 + epsilon

# Dividir os dados em treino (80%) e teste (20%)
train_size = int(0.8 * len(x))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Visualizar os dados
plt.scatter(x, y, label='Dados Sintéticos')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()