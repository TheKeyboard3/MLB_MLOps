import pandas as pd
import yaml
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

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
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        solver='liblinear', # liblinear добре працює на малих датасетах
        random_state=config['project']['random_state'],
        max_iter=2000
    )
)

# Сітка параметрів для перебору
# logisticregression__C: керує силою регуляризації (менше значення = сильніша регуляризація)
# class_weight: 'balanced' автоматично підніме вагу меншого класу, що критично для F1
param_grid = {
    'logisticregression__C': [0.1, 1.0, 5.0, 10.0, 50.0],
    'logisticregression__penalty': ['l1', 'l2'], # Спробуємо L1 (Lasso) та L2 (Ridge)
    'logisticregression__class_weight': [None, 'balanced']
}

# 5. Тренування з крос-валідацією
print("Tuning hyperparameters with GridSearchCV...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro', # Явно оптимізуємо ту метрику, яка вам потрібна
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

# 6. Збереження артефактів для етапу evaluate
os.makedirs(os.path.dirname(config['paths']['model_output']), exist_ok=True)
os.makedirs(os.path.dirname(config['paths']['test_data_output']), exist_ok=True)

# 7. Збереження найкращої моделі
# Ми зберігаємо best_estimator_, щоб evaluate.py працював без змін
best_model = grid_search.best_estimator_

os.makedirs(os.path.dirname(config['paths']['model_output']), exist_ok=True)
with open(config['paths']['model_output'], 'wb') as f:
    pickle.dump(best_model, f)

# Зберігаємо тестовий набір (X + y) для валідації
test_df = X_test.copy()
test_df[target] = y_test
test_df.to_csv(config['paths']['test_data_output'], index=False)

print(f"Training complete. Model saved to {config['paths']['model_output']}")
