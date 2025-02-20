import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

results_dir = os.path.join(os.path.dirname(__file__), "../resultados")
os.makedirs(results_dir, exist_ok=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def compute_gradient(X, y, weights):
    m = X.shape[0]
    y_pred = sigmoid(np.dot(X, weights))
    error = y_pred - y
    gradient = np.dot(X.T, error) / m
    return gradient

def gradient_descent(X, y, lr=0.1, epochs=1000):
    weights = np.zeros(X.shape[1])
    loss_history = []
    
    for _ in range(epochs):
        gradient = compute_gradient(X, y, weights)
        weights -= lr * gradient
        loss = cross_entropy_loss(y, sigmoid(np.dot(X, weights)))
        loss_history.append(loss)
    
    return weights, loss_history

# Gerar dados sintéticos
X, y = make_classification(n_samples=500, n_features=2, n_classes=2, n_redundant=0, random_state=42)
X = np.c_[np.ones(X.shape[0]), X]  # Adicionar bias (termo constante)

# Dividir em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar modelo
weights, loss_history = gradient_descent(X_train, y_train, lr=0.1, epochs=1000)

# Avaliação
y_pred_test = sigmoid(np.dot(X_test, weights)) >= 0.5
accuracy = np.mean(y_pred_test == y_test)
print(f'Acurácia no conjunto de teste: {accuracy:.2f}')

# Visualizar fronteira de decisão
xx, yy = np.meshgrid(np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100),
                     np.linspace(X[:,2].min()-1, X[:,2].max()+1, 100))
grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
preds = sigmoid(np.dot(grid, weights)).reshape(xx.shape)

plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.3)
plt.scatter(X_test[:, 1], X_test[:, 2], c=y_test, cmap='coolwarm', edgecolors='k')
plt.xlabel('Variável 1')
plt.ylabel('Variável 2')
plt.title('Fronteira de decisão da Regressão Logística')
plt.savefig(os.path.join(results_dir, "q2-fronteira-de-decisao.png"))
plt.show()
