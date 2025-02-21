import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

results_dir = os.path.join(os.path.dirname(__file__), "../resultados")
os.makedirs(results_dir, exist_ok=True)

# Configurar sementes para reprodutibilidade
np.random.seed(0)
torch.manual_seed(0)

# Função para gerar dados sintéticos
def generate_synthetic_data(N, a=3, b=5, sigma=2):
    x = np.random.uniform(-10, 10, N)
    epsilon = np.random.normal(0, sigma, N)
    y = a * x + b + epsilon
    return x, y

# Função para normalizar os dados
def normalize_data(x_train, x_test):
    mean = x_train.mean()
    std = x_train.std()
    x_train_normalized = (x_train - mean) / std
    x_test_normalized = (x_test - mean) / std
    return x_train_normalized, x_test_normalized

# Função para adicionar termo de bias
def add_bias_term_torch(x):
    ones = torch.ones((x.shape[0], 1))  # Criar uma coluna de 1s
    return torch.cat((ones, x), dim=1)  # Concatenar no eixo das colunas (dim=1)

# Função para calcular a pseudo-inversa
def calculate_pseudo_inverse_torch(X_train, y_train):
    return torch.linalg.pinv(X_train) @ y_train  #

# Função para visualizar as perdas durante o treinamento
def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Perda durante o treinamento')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "q1-visualizacao-de-perdas.png"))  
    plt.show()

# Função para visualizar os resultados
def plot_results(x, y, y_pred_pinv, y_pred_nn, train_size):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  

    # Gráfico 1: Pseudo-Inversa
    axs[0].scatter(x, y, label="Dados reais", alpha=0.6, color='blue')  
    axs[0].plot(x[train_size:], y_pred_pinv, label="Pseudo-Inversa", color="red", linewidth=2) 
    axs[0].set_title("Pseudo-Inversa")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].grid(True)
    axs[0].legend()

    # Gráfico 2: Rede Neural
    axs[1].scatter(x, y, label="Dados reais", alpha=0.6, color='blue')  
    axs[1].plot(x[train_size:], y_pred_nn, label="Rede Neural", color="yellow", linewidth=2)  
    axs[1].set_title("Rede Neural")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].grid(True)
    axs[1].legend()

    plt.suptitle("Regressão Linear: Pseudo-Inversa vs Rede Neural", fontsize=16)
    plt.tight_layout()  
    plt.subplots_adjust(top=0.85) 
    plt.savefig(os.path.join(results_dir, "q1-regressao-linear-comparacao.png"))  
    plt.show()

# Implementação com Rede Neural
class LinearRegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  

    def forward(self, x):
        return self.linear(x)

# Função para treinar a rede neural
def train_neural_network(X_train_torch, y_train_torch, epochs=1000, lr=0.01):
    model = LinearRegressionNN()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train_torch)
        loss = criterion(y_pred, y_train_torch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 100 == 0:
         print(f'Época {epoch}, Erro: {loss.item():.4f}')

    return model, losses

N = 100
x, y = generate_synthetic_data(N)

x, y = shuffle(x, y, random_state=0)

# Dividir os dados em treino e teste
train_size = int(0.8 * len(x))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Dividir os dados em treino (80%) e teste (20%)
train_size = int(0.8 * N)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

x_train, x_test = normalize_data(x_train, x_test)

X_train = add_bias_term_torch(torch.tensor(x_train, dtype=torch.float32).view(-1, 1))
X_test = add_bias_term_torch(torch.tensor(x_test, dtype=torch.float32).view(-1, 1))

# Regressão Linear via Pseudo-Inversa
theta = calculate_pseudo_inverse_torch(X_train, torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
print("Coeficientes da regressão (Mínimos Quadrados):", theta.numpy().flatten())

# Previsões no conjunto de teste
y_pred_pinv = X_test @ theta

X_train_torch = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_torch = torch.tensor(x_test, dtype=torch.float32).view(-1, 1)

# Treinamento da Rede Neural
model, losses = train_neural_network(X_train_torch, y_train_torch)

plot_losses(losses)

w_nn, b_nn = model.linear.weight.item(), model.linear.bias.item()
print("Coeficientes da Rede Neural: w =", w_nn, ", b =", b_nn)

y_pred_nn = model(X_test_torch).detach().numpy()

mse_pinv = mean_squared_error(y_test, y_pred_pinv)
mse_nn = mean_squared_error(y_test, y_pred_nn)

print(f"MSE (Pseudo-Inversa): {mse_pinv:.4f}")
print(f"MSE (Rede Neural): {mse_nn:.4f}")

plot_results(x, y, y_pred_pinv, y_pred_nn, train_size)
