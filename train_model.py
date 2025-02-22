import os
import pandas as pd
import joblib
import numpy as np
import scipy.sparse as sp
from docx import Document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === 1. Извлечение признаков ===
def extract_docx_features(filepath, label):
    """Извлекает текст, шрифт и стиль из .docx."""
    doc = Document(filepath)
    features = []

    for para in doc.paragraphs:
        if para.text.strip():  # Проверяем, что текст не пустой
            for run in para.runs:
                features.append({
                    "text": para.text.strip(),
                    "font_name": run.font.name or "Unknown",
                    "font_size": run.font.size.pt if run.font.size else 0,
                    "bold": int(run.bold or 0),
                    "italic": int(run.italic or 0),
                    "label": label
                })

    return features

# === 2. Загрузка датасета ===
def load_dataset():
    """Загружает документы из datasets/ и создает DataFrame."""
    dataset = []

    for label, folder in [(1, "datasets/correct"), (0, "datasets/incorrect")]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.endswith(".docx"):
                    filepath = os.path.join(folder, filename)
                    dataset.extend(extract_docx_features(filepath, label))

    return pd.DataFrame(dataset)

# === 3. Основной процесс обучения ===
def train_model():
    """Обучение модели и сохранение ее на диск."""
    print("📥 Загрузка данных...")
    data = load_dataset()

    if data.empty:
        print("❌ Данные не загружены! Проверьте папки datasets/correct и datasets/incorrect.")
        return

    print(f"🔍 Загружено {len(data)} строк.")
    print(data.head())  # Проверка структуры данных

    # Подготовка признаков
    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(data["text"])

    # Объединение признаков (текст + стиль)
    X_combined = sp.hstack([
        X_text,
        np.array(data["font_size"]).reshape(-1, 1),
        np.array(data["bold"]).reshape(-1, 1),
        np.array(data["italic"]).reshape(-1, 1)
    ])

    y = data["label"]

    # Разделение на обучающую и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

    # Обучение модели
    print("🚀 Обучение модели...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Оценка точности
    y_pred = model.predict(X_test)
    print("\n📊 Отчет по классификации:")
    print(classification_report(y_test, y_pred))

    # Сохранение модели и векторизатора
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/gost_checker_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")

    print("✅ Обучение завершено. Модель сохранена в папке 'model'.")

# === 4. Запуск ===
if __name__ == "__main__":
    train_model()
