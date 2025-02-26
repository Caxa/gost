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

def load_dataset(correct_folder, incorrect_folder):
    """Загружает документы из указанных папок и создает DataFrame."""
    dataset = []

    for label, folder in [(1, correct_folder), (0, incorrect_folder)]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.endswith(".docx"):
                    filepath = os.path.join(folder, filename)
                    dataset.extend(extract_docx_features(filepath, label))

    return pd.DataFrame(dataset)

def train_model(correct_folder, incorrect_folder):
    """Обучение модели и сохранение её на диск."""
    print("Загрузка данных...")
    data = load_dataset(correct_folder, incorrect_folder)

    if data.empty:
        print("Данные не загружены! Проверьте указанные папки.")
        return

    print(f"Загружено {len(data)} строк.")
    print(data.head())  # Проверка структуры данных

    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(data["text"])

    X_combined = sp.hstack([
        X_text,
        np.array(data["font_size"]).reshape(-1, 1),
        np.array(data["bold"]).reshape(-1, 1),
        np.array(data["italic"]).reshape(-1, 1)
    ])

    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

    print("Обучение модели...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/gost_checker_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")

    print("Обучение завершено. Модель сохранена в папке 'model'.")

if __name__ == "__main__":
    correct_path = input("Введите путь к папке с корректными документами: ")
    incorrect_path = input("Введите путь к папке с некорректными документами: ")
    train_model(correct_path, incorrect_path)
