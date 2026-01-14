import joblib
from preprocess import clean_text

model = joblib.load("model/resume_classifier.pkl")

while True:
    text = input("Paste resume text (or 'exit'): ")
    if text.lower() == "exit":
        break

    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]

    print(f"Predicted category: {prediction}")
