Запит:
Контекст: я студент, виконую практичну роботу з MLOps у Jupyter на Python 3.12.12, ОС Linux, інструмент AutoML - H2O AutoML
Дані: тип задачі: бінарна класифікація, CSV ~600 рядків, 30 ознак, ціль 'diagnosis', дизбаланс ~3:2.
Мета: отримати план кроків для baseline + AutoML з трекінгом у  MLFlow.
Обмеження: без Docker, локальний ноутбук, без GPU, час на AutoML 15 хв.
Формат відповіді:
  1) Дерево.
  2) config.yaml
  3) .gitignore
  4) requirements.txt.
Задання: Згенеруй дерево папок під ML проект і мінімальний configs/config.yaml
Відповідь:
Це чудова структура для старту. Оскільки ви працюєте на Linux без Docker, ми побудуємо структуру так, щоб вона була максимально чистою та відтворюваною через віртуальне оточення (`venv`).

Ось готове рішення згідно з вашими вимогами:

### 1\. Структура проекту (Дерево)

Ми використовуємо спрощену версію стандарту *Cookiecutter Data Science*, адаптовану під навчальний процес та локальний MLFlow.

```text
mlops_h2o_project/
├── configs/
│   └── config.yaml          # Всі параметри тут (шляхи, гіперпараметри)
├── data/
│   ├── raw/                 # Сюди покласти ваш вихідний CSV
│   └── processed/           # Для збереження очищених даних (якщо треба)
├── mlruns/                  # Тут MLFlow буде зберігати локальні метрики та артифакти
├── notebooks/
│   └── 01_h2o_automl.ipynb  # Основний робочий зошит
├── src/                     # Допоміжний код (щоб не засмічувати ноутбук)
│   ├── __init__.py
│   └── utils.py             # Наприклад, функція для завантаження YAML
├── .gitignore               # Важливо для git
├── README.md                # Опис проекту
└── requirements.txt         # Залежності
```

-----

### 2\. configs/config.yaml

Цей файл дозволяє керувати експериментом без зміни коду. Я врахував обмеження у 15 хвилин та специфіку даних.

```yaml
project:
  name: "h2o_automl_diagnosis"
  random_state: 42
  experiment_name: "H2O_AutoML_Baseline"

data:
  raw_path: "../data/raw/data.csv"
  processed_path: "../data/processed/"
  target_col: "diagnosis"
  id_col: null            # Вкажіть, якщо є ID колонки, яку треба видалити

model:
  type: "binary_classification"
  balance_classes: true   # Корисно, бо у вас дизбаланс 3:2
  folds: 5                # Крос-валідація

h2o_automl:
  max_runtime_secs: 900   # 15 хвилин (жорстке обмеження)
  max_models: 20          # Додаткове обмеження, щоб не перевантажити RAM
  exclude_algos:          # Опціонально: можна виключити "DeepLearning" без GPU
    - "DeepLearning"      # DL на CPU може бути повільним, краще GBM/XGBoost
  sort_metric: "AUC"      # Стандарт для бінарної класифікації

mlflow:
  tracking_uri: "../mlruns"
```

-----

### 3\. .gitignore

Критично важливо не завантажувати дані та локальні артефакти моделей у репозиторій.

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Virtual Environment
venv/
env/
ENV/

# Data (Never commit data!)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# MLFlow & H2O artifacts
mlruns/
h2o_models/
*.zip
```

-----

### 4\. requirements.txt

Мінімальний набір бібліотек для вашого Python 3.12. H2O потребує Java (JDK), переконайтеся, що вона встановлена в Linux (`sudo apt install default-jdk`).

```text
# Data Manipulation
pandas>=2.1.0
numpy>=1.26.0

# AutoML
h2o>=3.44.0.3

# Experiment Tracking
mlflow>=2.9.0

# Configuration
pyyaml>=6.0.1

# Visualization & Utils
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.3.0  # Для метрик порівняння або спліту даних

# Jupyter
jupyterlab>=4.0.0
ipykernel>=6.29.0
```

-----

### Що далі?

Щоб почати роботу, виконайте в терміналі:

1.  `python3.12 -m venv venv`
2.  `source venv/bin/activate`
3.  `pip install -r requirements.txt`

Чи бажаєте ви, щоб я написав **код для `src/utils.py`** (завантаження конфігу) та **шаблон Notebook-а**, який ініціалізує H2O та логує результати в MLFlow?



Запит:
Дай код для baseline (LogisticRegression) з читанням параметрів із configs/config.yml розділенням train/test, обчисленням accuracy, precision, recall, та f1 (macro), і логуванням у MLflow (параметри, метрики, матриця змішування як артефакт).
Формат:
  1) train.py (код-блок);
  2) evaluate.py (код-блок);
  3) команди запуску.
Відповідь:
Ось реалізація **Baseline** рішення. Ми розділимо процес на тренування (`train.py`) та оцінку (`evaluate.py`), що є гарною практикою в MLOps для відтворюваності.

Спочатку додайте ці налаштування у ваш `configs/config.yaml` (доповнення до попереднього):

```yaml
# configs/config.yaml (додаткова секція)
split:
  test_size: 0.2
  random_state: 42

logistic_regression:
  C: 1.0
  solver: "lbfgs"
  max_iter: 1000

paths:
  model_output: "../models/baseline_model.pkl"
  test_data_output: "../data/processed/test_data.csv"
```

-----

### 1\. `src/train.py`

Цей скрипт завантажує дані, розділяє їх, тренує модель та зберігає її (pickle) разом із тестовим набором даних для етапу валідації.

```python
import pandas as pd
import yaml
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

# 1. Завантаження конфігурації
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Підготовка даних
print("Loading data...")
df = pd.read_csv(config['data']['raw_path'])

# Проста обробка: видалення ID (якщо є) та кодування цільової змінної
if config['data']['id_col'] and config['data']['id_col'] in df.columns:
    df = df.drop(columns=[config['data']['id_col']])

target = config['data']['target_col']
X = df.drop(columns=[target])
y = df[target]

# Якщо цільова змінна текстова (B/M), кодуємо в 0/1
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# 3. Розділення (Train/Test)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config['split']['test_size'],
    random_state=config['split']['random_state'],
    stratify=y # Важливо при дисбалансі
)

# 4. Створення пайплайну (Scaling + LogReg)
# LogisticRegression чутливий до масштабу, тому додаємо StandardScaler
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=config['logistic_regression']['C'],
        solver=config['logistic_regression']['solver'],
        max_iter=config['logistic_regression']['max_iter'],
        random_state=config['project']['random_state']
    )
)

# 5. Тренування
print("Training Baseline model...")
model.fit(X_train, y_train)

# 6. Збереження артефактів для етапу evaluate
os.makedirs(os.path.dirname(config['paths']['model_output']), exist_ok=True)
os.makedirs(os.path.dirname(config['paths']['test_data_output']), exist_ok=True)

# Зберігаємо модель
with open(config['paths']['model_output'], 'wb') as f:
    pickle.dump(model, f)

# Зберігаємо тестовий набір (X + y) для валідації
test_df = X_test.copy()
test_df[target] = y_test
test_df.to_csv(config['paths']['test_data_output'], index=False)

print(f"Training complete. Model saved to {config['paths']['model_output']}")
```

-----

### 2\. `src/evaluate.py`

Цей скрипт виконує логування в MLflow. Він завантажує збережену модель і тестові дані, рахує метрики та будує графіки.

```python
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
```

-----

### 3\. Команди запуску

Виконуйте команди з кореневої папки проекту (`mlops_h2o_project/`):

1.  **Активуйте середовище (якщо ще не активовано):**

    ```bash
    source venv/bin/activate
    ```

2.  **Запустіть тренування:**

    ```bash
    # Переконайтеся, що папка models існує, або скрипт її створить
    python src/train.py
    ```

3.  **Запустіть оцінку (логування в MLFlow):**

    ```bash
    python src/evaluate.py
    ```

4.  **Перевірте результати:**
    Запустіть інтерфейс MLFlow і перейдіть за адресою [http://127.0.0.1:5000](https://www.google.com/search?q=http://127.0.0.1:5000)

    ```bash
    mlflow ui --backend-store-uri mlruns
    ```

### Чи бажаєте ви, щоб я показав, як тепер інтегрувати H2O AutoML у цей самий пайплайн (у файл `01_h2o_automl.ipynb`), щоб порівняти його з цим Baseline?


Запит:
Запропонуй метрики для оцінювання моделі-лідера, що буде пізніше отримано з H2O AutoML

Відповідь:
Для задачі **медичної діагностики** з помірним дисбалансом (3:2), вибір метрик критично впливає на корисність моделі. У медицині ціна помилки (пропустити хворобу) зазвичай вища, ніж хибна тривога.

Ось рекомендований набір метрик для оцінки лідера H2O AutoML з поясненням, чому вони важливі саме тут:

### 1\. Основні метрики (Must-have)

Ці метрики треба обов'язково логувати в MLflow для порівняння з Baseline.

  * **AUC-ROC (Area Under the Curve):**
      * *Чому:* Це стандартна метрика для AutoML. Вона показує, наскільки добре модель вміє розділяти класи (хворий/здоровий) незалежно від обраного порогу (threshold).
      * *Для дисбалансу:* Стійка до помірного дисбалансу (3:2).
  * **Recall (Повнота / Sensitivity):**
      * *Чому:* **Найважливіша метрика для медицини.** Вона відповідає на питання: "Скількох хворих ми реально знайшли?". Низький Recall означає, що ми відпускаємо додому хворих людей.
  * **Precision (Точність):**
      * *Чому:* Показує, скільки з тих, кого ми назвали "хворими", справді хворі. Низька точність означає багато хибних тривог (стрес для пацієнтів, зайві аналізи).
  * **F1-Score (Macro):**
      * *Чому:* Гармонічне середнє між Precision та Recall. Дає одну цифру для загальної оцінки балансу.

### 2\. Просунуті метрики (Nice-to-have)

Для глибшого аналізу в студентській роботі:

  * **F2-Score:**
      * *Чому:* Це модифікація F1, яка **надає пріоритет Recall** (вдвічі важливіше за Precision). Це ідеальна метрика для вибору "фінальної" моделі в медицині.
      * *Формула:* $F2 = (1 + 2^2) \cdot \frac{Precision \cdot Recall}{(2^2 \cdot Precision) + Recall}$
  * **LogLoss (Logarithmic Loss):**
      * *Чому:* H2O часто використовує її для оптимізації. Вона карає модель за "впевнені" неправильні відповіді (наприклад, модель на 99% впевнена, що пацієнт здоровий, а він хворий).
  * **Specificity (Специфічність):**
      * *Чому:* Здатність моделі правильно визначати *здорових* людей.

### 3\. Як дістати ці метрики з H2O (Code snippet)

H2O об'єкти відрізняються від Scikit-Learn. Вони зберігають метрики всередині об'єкта моделі.

Ось функція, яку можна додати в `src/utils.py` або використати в ноутбуці для отримання словника метрик для MLFlow:

```python
from h2o.model.metrics_base import H2OMetrics
import h2o

def get_h2o_metrics(model, test_frame):
    """
    Витягує метрики з моделі H2O на тестовому наборі даних.
    Повертає словник для MLflow.
    """
    # Генеруємо метрики на тестовому сеті
    perf = model.model_performance(test_frame)
    
    # Отримуємо confusion matrix як таблицю H2O (для розрахунку F2)
    # H2O за замовчуванням вибирає поріг за F1-score, але можна брати 'max_recall'
    cm = perf.confusion_matrix() 
    
    # Витягуємо значення (для бінарної класифікації)
    # Структура CM в H2O: [TN, FP], [FN, TP] (залежить від версії, краще брати через методи)
    
    # Стандартні методи H2O
    metrics = {
        "h2o_auc": perf.auc(),
        "h2o_logloss": perf.logloss(),
        "h2o_accuracy": perf.accuracy()[0][1], # [0][1] бере значення для оптимального порогу
        "h2o_precision": perf.precision()[0][1],
        "h2o_recall": perf.recall()[0][1],     # Це і є Sensitivity
        "h2o_f1": perf.F1()[0][1]
    }
    
    # Ручний розрахунок F2-score (оскільки H2O його прямо не віддає в perf)
    p = metrics["h2o_precision"]
    r = metrics["h2o_recall"]
    if p + r > 0:
        metrics["h2o_f2"] = (5 * p * r) / (4 * p + r)
    else:
        metrics["h2o_f2"] = 0.0
        
    return metrics
```

### 4\. Порада для візуалізації в MLFlow

H2O має чудові вбудовані графіки. У ноутбуці або скрипті обов'язково збережіть **ROC Curve** як картинку і залогіть її як артефакт:

```python
import matplotlib.pyplot as plt

# Отримання даних для графіку
perf = leader_model.model_performance(test_h2o)
fpr = perf.fprs
tpr = perf.tprs

# Побудова
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {perf.auc():.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("H2O Leader ROC Curve")
plt.legend()
plt.savefig("h2o_roc_curve.png")

# Логування
mlflow.log_artifact("h2o_roc_curve.png")
```

Запит:
Напиши код для запуску H2O AutoML за парамтерами, вказаними у config.yaml із логуванням у MLflow.
Включи також ініціалізацію H2O, отримання лідерборду (найкращі 5), оцінювання лідера на test (accuracy, precision, recall, матриця змішування, AUC-ROC).
Формат:
  1) automl_run.py (код-блок);
  2) команди CLI;

Відповідь:
Ось реалізація скрипту `src/automl_run.py`, який повністю інтегрує H2O AutoML з MLflow, використовуючи ваш `config.yaml`.

Цей скрипт виконує повний цикл: ініціалізація -\> підготовка даних -\> навчання (AutoML) -\> вибір лідера -\> оцінка -\> логування.

### 1\. `src/automl_run.py`

```python
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
        
        aml = H2OAutoML(
            max_runtime_secs=cfg['h2o_automl']['max_runtime_secs'],
            max_models=cfg['h2o_automl']['max_models'],
            seed=cfg['project']['random_state'],
            sort_metric=cfg['h2o_automl']['sort_metric'],
            exclude_algos=cfg['h2o_automl'].get('exclude_algos', []),
            balance_classes=cfg['model']['balance_classes'],
            project_name=cfg['project']['name']
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
    # finally:
        # h2o.shutdown(prompt=False) # Розкоментуйте, якщо хочете авто-вимкнення кластера
```

-----

### 2\. Команди CLI

Виконуйте з кореневої папки проекту (`mlops_h2o_project/`).

**Крок 1: Активація віртуального середовища**

```bash
source venv/bin/activate
```

**Крок 2: Запуск AutoML**
Цей процес займе приблизно той час, який ви вказали у `max_runtime_secs` + час на ініціалізацію та збереження (близько 16-17 хвилин загалом).

```bash
python src/automl_run.py
```

**Крок 3: Перегляд результатів (Dashboard)**
Якщо сервер MLflow ще не запущений:

```bash
mlflow ui --backend-store-uri mlruns
```

Перейдіть на `http://127.0.0.1:5000` у браузері.

**Крок 4: (Опціонально) Перегляд інтерфейсу H2O Flow**
Поки скрипт працює (або якщо ви закоментували `h2o.shutdown`), ви можете бачити хід тренування в реальному часі у "рідному" інтерфейсі H2O.
Відкрийте у браузері: `http://localhost:54321`

### Особливості реалізації:

1.  **H2O Init:** Використовує параметр `max_mem_size='2G'`, щоб не "з'їсти" всю пам'ять ноутбука, але мати достатньо для 600 рядків.
2.  **Conversion:** Критично важливий рядок `df_h2o[target].asfactor()` — без нього H2O може подумати, що це задача регресії (якщо діагноз 0/1), і метрики типу AUC/Confusion Matrix будуть недоступні.
3.  **Artifacts:** Ми зберігаємо не тільки цифри, а й картинки (матрицю, ROC криву) та CSV з лідербордом, щоб викладачу було зручно перевіряти роботу через MLflow UI.
