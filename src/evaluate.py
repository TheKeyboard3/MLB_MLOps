import pandas as pd
import yaml
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Налаштування matplotlib для сервера (без GUI)
plt.switch_backend('Agg')

# 1. Завантаження конфігурації
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Налаштування MLFlow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['project']['experiment_name'])

def evaluate():
    print("Starting evaluation...")

    # 2. Завантаження моделі та даних
    with open(config['paths']['model_output'], 'rb') as f:
        model = pickle.load(f)

    test_df = pd.read_csv(config['paths']['test_data_output'])
    target_col = config['data']['target_col']

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # 3. Передбачення
    y_pred = model.predict(X_test)

    # 4. Обчислення метрик
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1_macro": f1_score(y_test, y_pred, average='macro')
    }

    print(f"Metrics: {metrics}")

    # 5. Логування в MLFlow
    with mlflow.start_run(run_name="Baseline_LogReg"):
        # Логуємо параметри (з конфігу)
        mlflow.log_params(config['logistic_regression'])
        mlflow.log_param("test_size", config['split']['test_size'])

        # Логуємо метрики
        mlflow.log_metrics(metrics)

        # 6. Матриця змішування (Confusion Matrix)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Збереження картинки
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()

        # Логування картинки як артефакту
        mlflow.log_artifact(plot_path)

        # Логування самої моделі (опціонально, sklearn формат)
        mlflow.sklearn.log_model(model, "model")

        print("Evaluation logged to MLFlow successfully.")

if __name__ == "__main__":
    evaluate()
