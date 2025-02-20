import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

results_dir = os.path.join(os.path.dirname(__file__), "../resultados")
os.makedirs(results_dir, exist_ok=True)

# Gerar dados
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
y = y.reshape(-1, 1)  # Ajustar formato da saída

# Divisão dos dados
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Converter para tensores
def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

X_train, y_train = to_tensor(X_train), to_tensor(y_train)
X_val, y_val = to_tensor(X_val), to_tensor(y_val)
X_test, y_test = to_tensor(X_test), to_tensor(y_test)

# Definição da rede MLP
class MLP(nn.Module):
    def __init__(self, n_neurons):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, n_neurons)
        self.output = nn.Linear(n_neurons, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Treinamento da rede
def train_model(n_neurons, epochs=1000, lr=0.01):
    model = MLP(n_neurons)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)
            val_losses.append(val_loss.item())
    
    return model, losses, val_losses

# Testando diferentes números de neurônios
neurons_list = [5, 10, 20, 50]
best_model = None
best_val_loss = float('inf')
best_neurons = 0

plt.figure(figsize=(12, 6))

for n in neurons_list:
    model, losses, val_losses = train_model(n)
    plt.plot(val_losses, label=f'{n} neurônios')
    
    if min(val_losses) < best_val_loss:
        best_val_loss = min(val_losses)
        best_model = model
        best_neurons = n

plt.xlabel('Épocas')
plt.ylabel('Loss de Validação')
plt.legend()
plt.title('Evolução da Loss na Validação')
plt.savefig(os.path.join(results_dir), "q3-evolucao-da-loss-na-validacao.png")
plt.show()

# Avaliação no conjunto de teste
best_model.eval()
with torch.no_grad():
    y_test_pred = best_model(X_test)
    y_test_pred = (y_test_pred >= 0.5).float()
    accuracy = (y_test_pred.eq(y_test).sum() / len(y_test)).item()
    
print(f'Melhor número de neurônios: {best_neurons}')
print(f'Acurácia no conjunto de teste: {accuracy:.4f}')

# Visualização da Fronteira de Decisão
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        preds = model(grid).numpy().reshape(xx.shape)
    
    plt.contourf(xx, yy, preds, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title('Fronteira de Decisão')
    plt.savefig(os.path.join(results_dir, "q3-fronteira-de-decisao.png"))
    plt.show()

plot_decision_boundary(best_model, X, y)
