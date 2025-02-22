import os
import joblib
import numpy as np
import scipy.sparse as sp
from docx import Document
from sklearn.feature_extraction.text import CountVectorizer

# Загрузка модели и векторизатора
model = joblib.load("model/gost_checker_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def extract_docx_features(filepath):
    """Извлекает текст и форматирование из docx."""
    doc = Document(filepath)
    features = []

    for para in doc.paragraphs:
        for run in para.runs:
            features.append({
                "text": para.text.strip(),
                "font_name": run.font.name,
                "font_size": run.font.size.pt if run.font.size else None,
                "bold": run.bold,
                "italic": run.italic
            })

    return features

def check_document(filepath):
    """Проверяет документ на соответствие ГОСТ."""
    features = extract_docx_features(filepath)
    texts = [f["text"] for f in features]
    font_sizes = [f["font_size"] or 0 for f in features]
    bolds = [int(f["bold"] or 0) for f in features]
    italics = [int(f["italic"] or 0) for f in features]

    # Преобразование текста
    X_text = vectorizer.transform(texts)

    # Объединение признаков
    X_combined = sp.hstack([
        X_text,
        np.array(font_sizes).reshape(-1, 1),
        np.array(bolds).reshape(-1, 1),
        np.array(italics).reshape(-1, 1)
    ])

    # Предсказание
    predictions = model.predict(X_combined)

    # Анализ ошибок
    errors = [features[i] for i in range(len(predictions)) if predictions[i] == 0]
    if errors:
        print("❌ Найдены ошибки:")
        for error in errors:
            print(f"- Текст: '{error['text']}', Шрифт: {error['font_name']}, Размер: {error['font_size']}")
    else:
        print("✅ Документ соответствует ГОСТ!")

# Запуск проверки
if __name__ == "__main__":
    filepath = input("Введите путь к .docx файлу: ")
    if os.path.exists(filepath):
        check_document(filepath)
    else:
        print("❌ Файл не найден.")
