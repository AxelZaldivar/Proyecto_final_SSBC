import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = 0.01
        self.iterations = 10000

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X):
        # Capa oculta
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input,)

        # Capa de salida
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        A = self.sigmoid(output_input)

        return A, hidden_output

    def backward_propagation(self, X, A, hidden_output, y):
        m = len(y)

        # Gradiente en la capa de salida
        dz_output = A - y
        dw_output = np.dot(hidden_output.T, dz_output) / m
        db_output = np.sum(dz_output, axis=0, keepdims=True) / m

        # Propagar el error a la capa oculta
        dz_hidden = np.dot(dz_output, self.weights_hidden_output.T) * hidden_output * (1 - hidden_output)
        dw_hidden = np.dot(X.T, dz_hidden) / m
        db_hidden = np.sum(dz_hidden, axis=0, keepdims=True) / m

        return dw_hidden, db_hidden, dw_output, db_output

    # Entrenamiento con descenso del gradiente
    def train(self, X, y):
        m = len(y)
        cost_history = np.zeros(self.iterations)

        for i in range(self.iterations):
            A, hidden_output = self.forward_propagation(X)
            dw_hidden, db_hidden, dw_output, db_output = self.backward_propagation(X, A, hidden_output, y)

            # Actualizar pesos y sesgos
            self.weights_input_hidden -= self.learning_rate * dw_hidden
            self.bias_hidden -= self.learning_rate * db_hidden
            self.weights_hidden_output -= self.learning_rate * dw_output
            self.bias_output -= self.learning_rate * db_output

            epsilon = 1e-10
            cost = -1 / m * np.nansum(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))
            cost_history[i] = cost

        return cost_history

def neural_network(file_path, opc):
    # Cargar los datos desde el archivo CSV
    df = pd.read_csv(file_path)

    # One-hot encoding para variables categóricas
    if opc == 0:
        categorical_columns = ['Education', 'City', 'Gender']
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
        df_encoded['EverBenched'] = df_encoded['EverBenched'].map({'No': 0, 'Yes': 1})
    elif opc == 1:
        categorical_columns = []
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
    elif opc == 2:
        categorical_columns = ['animal_name']
        df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # Columna objetivo
    target_column = ['LeaveOrNot', 'output', 'class_type'][opc]

    # Ajustar los parámetros de entrenamiento y prueba, y definir las matrices
    if opc == 2:  # Validación cruzada solo para "zoo.csv"
        train_data, test_data = train_test_split(df_encoded, test_size=0.2, random_state=42)
    else:
        train_idx = np.random.rand(len(df)) < 0.8
        train_data = df_encoded[train_idx]
        test_data = df_encoded[~train_idx]

    X_train = train_data.drop(target_column, axis=1).values.astype(float)
    y_train = train_data[target_column].values.reshape(-1, 1).astype(float)
    X_test = test_data.drop(target_column, axis=1).values.astype(float)
    y_test = test_data[target_column].values.reshape(-1, 1).astype(float)

    # Normalizar las características
    if opc < 2:
        mean_X = np.mean(X_train, axis=0)
        std_X = np.std(X_train, axis=0)
        X_train = (X_train - mean_X) / std_X
        X_test = (X_test - mean_X) / std_X

    # Inicialización de la red neuronal con 2 neuronas en la capa oculta
    input_size = X_train.shape[1]
    hidden_size = 4
    output_size = 1
    neural_net = NeuralNetwork(input_size, hidden_size, output_size)

    # Entrenar la red neuronal
    cost_history = neural_net.train(X_train, y_train)

    # Evaluar la red neuronal con los datos de prueba
    y_pred_prob, _ = neural_net.forward_propagation(X_test)
    y_pred = (y_pred_prob >= 0.4).astype(int)

    # Calcular métricas de evaluación (accuracy, precision, sensitivity, F1 score)
    correct_predictions = np.sum(y_pred == y_test)
    accuracy = correct_predictions / len(y_test)

    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    precision = true_positives / (true_positives + false_positives)
    sensitivity = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    # Visualizar la función de costo a lo largo de las iteraciones
    plt.plot(range(1, neural_net.iterations + 1), cost_history, color='red', label='Función de costo')
    plt.scatter(range(1, neural_net.iterations + 1), cost_history, color='blue', s=40, label='Puntos de datos')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    if opc == 2:
        plt.title('Red Neuronal (zoo.csv) con Cross-validation')
    else:
        plt.title('Red Neuronal (' + file_path + ')')

    # Visualizar las métricas
    plt.text(0.5, 0.9, f'Accuracy: {accuracy:.2f}', transform=plt.gca().transAxes, ha='center', va='center')
    plt.text(0.5, 0.8, f'Precision: {precision:.2f}', transform=plt.gca().transAxes, ha='center', va='center')
    plt.text(0.5, 0.7, f'Sensitivity: {sensitivity:.2f}', transform=plt.gca().transAxes, ha='center', va='center')
    plt.text(0.5, 0.6, f'F1 Score: {f1_score:.2f}', transform=plt.gca().transAxes, ha='center', va='center')

    # Mostrar el gráfico
    plt.show()

# Ejecutar los algoritmos
neural_network("Employee.csv", 0)
neural_network("heart.csv", 1)
neural_network("zoo.csv", 2)