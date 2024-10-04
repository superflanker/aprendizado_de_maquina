import numpy as np

# Função sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Função de previsão
def f(w, b, X):
    return sigmoid(np.dot(X, w) + b)

# Função para calcular o gradiente
def compute_gradient(X, y, w, b):
    m = X.shape[0]  # Número de amostras

    # Previsões para todas as amostras
    y_hat = f(w, b, X)

    # Cálculo dos erros (diferente entre a previsão e o valor real)
    erro = y_hat - y

    # Gradiente para os pesos w (vetorizado)
    dw = (1/m) * np.dot(X.T, erro)

    # Gradiente para o bias b (vetorizado)
    db = (1/m) * np.sum(erro)

    return db, dw

# Exemplo de uso

# Dados de entrada (6 amostras, 2 variáveis)
X_trein = np.array([[0., 1], [1., 2], [2., 3], [3., 4], [4., 5], [5., 6]])  # 6 amostras, 2 variáveis
y_trein = np.array([0, 0, 0, 1, 1, 1])  # Rótulos correspondentes
m = len(X_trein)
indices = np.random.permutation(m)
X_shuffled = X_trein[indices]
y_shuffled = y_trein[indices]

w = np.array([1, 1])

b = 0

for j in range(m):
    # Seleciona uma única amostra (vetor de características e valor alvo)
    x_i = X_shuffled[j]   # Vetor (n,)
    y_i = y_shuffled[j]   # Escalar
    print(np.array()[x_i]), np.array([y_i]))
    # Calcula o gradiente para essa amostra
    db, dw = compute_gradient(np.array([x_i]), np.array([y_i]), w, b)

    print(f"Gradiente dos pesos (dw): {dw}")
    print(f"Gradiente do bias (db): {db}")

