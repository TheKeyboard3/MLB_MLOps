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
