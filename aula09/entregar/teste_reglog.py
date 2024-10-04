import numpy as np

# Dados de entrada
x_trein = np.array([0., 1, 2, 3, 4, 5])
y_trein = np.array([0,  0, 0, 1, 1, 1])

# Inicialização de w, b e taxa de aprendizado
w = 0.0
b = 0.0
alpha = 0.5  # Taxa de aprendizado

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Previsão da probabilidade para os valores de x_trein
z = w * x_trein + b
y_pred = sigmoid(z)

def compute_cost(y, y_pred):
    n = len(y)
    cost = -(1/n) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost

# Função de atualização dos parâmetros
def update_parameters(w, b, x, y, y_pred, alpha):
    n = len(y)
    # Calculando os gradientes
    dw = (1/n) * np.dot(x, (y_pred - y))
    db = (1/n) * np.sum(y_pred - y)
    
    # Atualizando w e b
    w = w - alpha * dw
    b = b - alpha * db
    
    return w, b

# Número de iterações
num_iterations = 5000

for i in range(num_iterations):
    # Cálculo do valor de z e da previsão
    z = w * x_trein + b
    y_pred = sigmoid(z)
    
    # Calculando e imprimindo o custo a cada 100 iterações
    if i % 100 == 0:
        cost = compute_cost(y_trein, y_pred)
        print(f"Iteração {i}: Custo = {cost}")
    
    # Atualizando os parâmetros
    w, b = update_parameters(w, b, x_trein, y_trein, y_pred, alpha)

# Valores finais de w e b
print(f"Peso final w: {w}, Bias final b: {b}")

