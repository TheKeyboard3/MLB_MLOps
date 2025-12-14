import h2o
from h2o.automl import H2OAutoML
import mlflow
import mlflow.h2o
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Налаштування для графіків без GUI (Linux server/terminal)
plt.switch_backend('Agg')

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_automl():
    # 1. Завантаження конфігу та ініціалізація
    cfg = load_config()

    # Ініціалізація H2O (локальний кластер)
    # nthreads=-1 використовує всі ядра CPU
    print("Initializing H2O...")
    h2o.init(nthreads=-1, max_mem_size='2G') # Можна збільшити RAM, якщо є можливість

    # Налаштування MLflow
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['project']['experiment_name'])

    with mlflow.start_run(run_name="H2O_AutoML_Run"):
        print("Loading data...")

        # 2. Завантаження даних в H2OFrame
        # Варіант А: Читаємо raw (якщо processed ще немає)
        data_path = cfg['data']['raw_path']
        df_h2o = h2o.import_file(data_path)

        target = cfg['data']['target_col']

        # Видалення ID, якщо вказано
        if cfg['data']['id_col'] and cfg['data']['id_col'] in df_h2o.columns:
            df_h2o = df_h2o.drop(cfg['data']['id_col'])

        # ВАЖЛИВО: Для класифікації цільова змінна МАЄ бути фактором (enum)
        df_h2o[target] = df_h2o[target].asfactor()

        # 3. Розділення даних (Train / Test)
        # Використовуємо split_frame для внутрішнього формату H2O
        splits = df_h2o.split_frame(ratios=[1 - cfg['split']['test_size']],
                                    seed=cfg['split']['random_state'])
        train = splits[0]
        test = splits[1]

        print(f"Train size: {train.nrow}, Test size: {test.nrow}")

        # Визначення ознак (x) та цілі (y)
        y = target
        x = df_h2o.columns
        x.remove(y)

        # 4. Налаштування та запуск AutoML
        print(f"Starting AutoML for {cfg['h2o_automl']['max_runtime_secs']} seconds...")
        print("Note: max_models set to Unlimited to utilize full runtime.")

        aml = H2OAutoML(
            max_runtime_secs=cfg['h2o_automl']['max_runtime_secs'], # 900 сек (15 хв)

            # ЗМІНА 1: Знімаємо обмеження на кількість моделей.
            # Тепер H2O буде створювати моделі, поки не закінчиться час (15 хв).
            max_models=None,

            seed=cfg['project']['random_state'],
            sort_metric=cfg['h2o_automl']['sort_metric'],
            exclude_algos=cfg['h2o_automl'].get('exclude_algos', []),
            balance_classes=cfg['model']['balance_classes'],
            project_name=cfg['project']['name'],

            # ЗМІНА 2: Вимикаємо ранню зупинку (Early Stopping),
            # щоб H2O не зупинився, якщо метрика перестане покращуватися протягом 3-х моделей.
            # Ми хочемо, щоб він шукав складні комбінації до останньої секунди.
            stopping_rounds=0
        )

        aml.train(x=x, y=y, training_frame=train)

        # 5. Отримання лідерборду
        lb = aml.leaderboard
        print("Leaderboard (Top 5):")
        print(lb.head(rows=5))

        # Збереження лідерборду як артефакт (HTML та CSV)
        lb_df = lb.as_data_frame()
        lb_df.head(10).to_csv("leaderboard.csv", index=False)
        mlflow.log_artifact("leaderboard.csv")

        # 6. Оцінка лідера (Best Model)
        leader = aml.leader
        print(f"Leader Model: {leader.model_id}")

        # Отримуємо метрики на TEST dataset
        perf = leader.model_performance(test)

        # Витягуємо метрики (для бінарної класифікації)
        # H2O повертає таблиці порогів, беремо значення для оптимального F1
        metrics = {
            "auc": perf.auc(),
            "logloss": perf.logloss(),
            "accuracy": perf.accuracy()[0][1], # [0] - max threshold metric
            "precision": perf.precision()[0][1],
            "recall": perf.recall()[0][1],
            "f1": perf.F1()[0][1]
        }

        print("Test Metrics:", metrics)
        mlflow.log_metrics(metrics)

        # Логування параметрів AutoML
        mlflow.log_params({
            "max_runtime": cfg['h2o_automl']['max_runtime_secs'],
            "leader_algo": leader.algo,
            "balance_classes": cfg['model']['balance_classes']
        })

        # 7. Візуалізація: Confusion Matrix (Heatmap)
        # Отримуємо матрицю як Pandas DataFrame
        cm_table = perf.confusion_matrix().table.as_data_frame()
        # Зазвичай H2O повертає структуру, де останні колонки - це totals/rates
        # Нам потрібна лише матриця N x N (перші 2x2 для бінарної)
        cm_matrix = cm_table.iloc[0:2, 1:3].astype(int) # коригування індексів залежить від версії, зазвичай 1:3 це predicted класи

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Greens')
        plt.title(f'Confusion Matrix ({leader.model_id})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("h2o_confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("h2o_confusion_matrix.png")

        # 8. Візуалізація: ROC Curve
        fpr = perf.fprs
        tpr = perf.tprs
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {perf.auc():.3f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("h2o_roc_curve.png")
        plt.close()
        mlflow.log_artifact("h2o_roc_curve.png")

        # 9. Збереження моделі
        # MLflow має нативну підтримку H2O, це збереже модель у форматі, придатному для serving
        mlflow.h2o.log_model(leader, "h2o_best_model")

        print("AutoML run completed and logged to MLflow.")

if __name__ == "__main__":
    try:
        run_automl()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        h2o.cluster().shutdown(prompt=False) # Розкоментуйте, якщо хочете авто-вимкнення кластера
