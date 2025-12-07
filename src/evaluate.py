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
