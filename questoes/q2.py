import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

results_dir = os.path.join(os.path.dirname(__file__), "../resultados")
os.makedirs(results_dir, exist_ok=True)

# Gerando os dados sintéticos
np.random.seed(0)

# Gerando os dados
X, y = make_classification(n_samples=500, n_features=2, n_classes=2, n_redundant=0, random_state=42)

# Convertendo para tensores
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Tornar y_tensor 2D

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def train_logistic_regression(X_train, y_train, epochs=1000, lr=0.01):
    model = LogisticRegression()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        print("Epoch", epoch + 1, "/", epochs)
    
    return model

# Treina o modelo
model = train_logistic_regression(X_train, y_train, epochs=1000, lr=0.01)

# Avalia o modelo
model.eval()
with torch.no_grad():
    y_pred = model(X_test).round()
    accuracy = (y_pred == y_test).float().mean().item()
print("Acurácia:", accuracy)

# Gera a grade para a fronteira de decisão
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Converte para tensor e faz a previsão
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    Z = model(grid).reshape(xx.shape)

# Plota os dados e a fronteira de decisão
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="bwr", edgecolors="k")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Fronteira de Decisão da Regressão Logística")
plt.savefig(os.path.join(results_dir, "q2-fronteira-de-decisao.png"))  
plt.show()
