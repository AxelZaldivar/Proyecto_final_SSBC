import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función Descenso del gradiente con regularización L2
def des_grad(X_train, y_train, iterations, X_test):

    # Inicialización de los parámetros del modelo
    theta = np.random.randn(X_train.shape[1]) * 0.01
    learning_rate = 0.01

    # Descenso del gradiente sin regularización
    m = len(y_train)
    cost_history = np.zeros(iterations)
    lambda_reg = 0.01

    for i in range(iterations):
        h = X_train @ theta
        gradient = (X_train.T @ (h - y_train) + lambda_reg * theta) / m 
        theta -= learning_rate * gradient

        # Función de costo (error cuadrático medio)
        cost = np.sum((h - y_train) ** 2) / (2 * m)
        cost_history[i] = cost

    # Evaluar el modelo con los datos de prueba
    y_pred = (X_test @ theta) >= 0

    return cost_history, y_pred

# Función para calcular métricas
def metricas(y_pred, y_train):

    # Calcular métricas de evaluación (accuracy)
    correct_predictions = np.sum(np.round(y_pred) == y_train)
    accuracy = correct_predictions / len(y_train)

    # Calcular precision, sensitivity y F1 score
    true_positives = np.sum((np.round(y_pred) == 1) & (y_train == 1))
    false_positives = np.sum((np.round(y_pred) == 1) & (y_train == 0))
    false_negatives = np.sum((np.round(y_pred) == 0) & (y_train == 1))
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return accuracy, precision, recall, f1

# Función para regresión lineal con cross-validation
def linear_regression_cv(file_path, opc):

    # Cargar los datos desde el archivo CSV
    df = pd.read_csv(file_path)

    # One-hot encoding para variables categóricas
    if opc == 0:
        categorical_columns = ['Education', 'City', 'Gender']
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
        df_encoded['EverBenched'] = df_encoded['EverBenched'].map({'No': 0, 'Yes': 1})
        target_column = 'LeaveOrNot'
    elif opc == 1:
        categorical_columns = []
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
        target_column = 'output'
    elif opc == 2:
        categorical_columns = ['animal_name']
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
        target_column = 'class_type'

    # No aplicar cross-validation para los dos dataset diferentes de "zoo.csv"
    if "zoo.csv" != file_path:
        # Ajustar los parámetros de entrenamiento y prueba, y definir las matrices
        X_train = df_encoded.drop(target_column, axis=1).values.astype(float)
        y_train = df_encoded[target_column].values.astype(float)

        # Normalizar las características
        mean_X = np.mean(X_train, axis=0)
        std_X = np.std(X_train, axis=0)
        X_train = (X_train - mean_X) / std_X

        # Descenso del graiente        
        iterations = 400
        cost_history,y_pred = des_grad(X_train,y_train,iterations,X_train)
        accuracy,precision,recall,f1 = metricas(y_pred,y_train)

        # Visualizar la función de costo a lo largo de las iteraciones
        plt.plot(range(1, iterations + 1), cost_history, color='red', label='Función de costo')
        plt.scatter(range(1, iterations + 1), cost_history, color='blue', s=40, label='Puntos de datos')
        plt.xlabel('Iteraciones')
        plt.ylabel('Costo')
        plt.title('Regresión Lineal (' + file_path + ')')

        # Visualizar las métricas
        plt.text(0.5, 0.9, f'Accuracy: {accuracy:.2f}', transform=plt.gca().transAxes, ha='center', va='center')
        plt.text(0.5, 0.85, f'Precision: {precision:.2f}', transform=plt.gca().transAxes, ha='center', va='center')
        plt.text(0.5, 0.8, f'Sensitivity: {recall:.2f}', transform=plt.gca().transAxes, ha='center', va='center')
        plt.text(0.5, 0.75, f'F1 Score: {f1:.2f}', transform=plt.gca().transAxes, ha='center', va='center')

    # Aplicar cross-validation solo para "zoo.csv"
    else:
        # Dividir los datos en los pliegues para cross-validation
        num_folds = 5
        fold_size = len(df) // num_folds
        folds = [df_encoded[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]

        # Inicializar listas para almacenar métricas de rendimiento de cada fold
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        for i in range(num_folds):
            # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
            test_data = folds[i]
            train_data = pd.concat([folds[j] for j in range(num_folds) if j != i])

            # Ajustar los parámetros de entrenamiento y prueba, y definir las matrices
            X_train = train_data.drop(target_column, axis=1).values.astype(float)
            y_train = train_data[target_column].values.astype(float)
            X_test = test_data.drop(target_column, axis=1).values.astype(float)
            y_test = test_data[target_column].values.astype(float)

            # Descenso del graiente        
            iterations = 400
            cost_history,y_pred = des_grad(X_train,y_train,iterations,X_test)
            accuracy,precision,recall,f1 = metricas(y_pred,y_test)

            # Almacenar las métricas del rendimiento en las listas
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        # Calcular promedio de métricas de rendimiento de todos los pliegues
        avg_accuracy = np.mean(accuracy_list)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)

        # Mostrar el gráfico promedio
        plt.plot(range(1, iterations + 1), cost_history, color='red', label='Función de costo')
        plt.scatter(range(1, iterations + 1), cost_history, color='blue', s=40, label='Puntos de datos')
        plt.xlabel('Iteraciones')
        plt.ylabel('Costo')
        plt.title(f'Regresión Lineal con Cross-Validation (Zoo.csv)')

        # Visualizar las métricas promedio
        plt.text(0.5, 0.9, f'Average Accuracy: {avg_accuracy:.2f}', transform=plt.gca().transAxes, ha='center', va='center')
        plt.text(0.5, 0.85, f'Average Precision: {avg_precision:.2f}', transform=plt.gca().transAxes, ha='center', va='center')
        plt.text(0.5, 0.8, f'Average Sensitivity: {avg_recall:.2f}', transform=plt.gca().transAxes, ha='center', va='center')
        plt.text(0.5, 0.75, f'Average F1 Score: {avg_f1:.2f}', transform=plt.gca().transAxes, ha='center', va='center')

    # Mostrar el gráfico
    plt.show()

# Ejecutar los algoritmos
linear_regression_cv("Employee.csv", 0)
linear_regression_cv("heart.csv", 1)
linear_regression_cv("zoo.csv", 2)