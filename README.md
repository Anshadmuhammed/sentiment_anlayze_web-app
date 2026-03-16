# 💬 Sentiment Analysis Web App

A Machine Learning web application that classifies text as **Positive**, **Negative**, or **Neutral** using NLP and Logistic Regression, deployed with Streamlit.

---

## 📁 Project Structure

```
sentiment_analysis/
├── app.py           # Streamlit web app
├── train.py         # Model training script
├── predict.py       # Prediction helper
├── requirements.txt # Dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This generates `model.pkl`.

### 3. Launch the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | TF-IDF + Logistic Regression |
| Streamlit | Web interface |
| NLTK / re | Text preprocessing |

---

## 💡 How It Works

1. User enters text in the web app
2. Text is preprocessed (lowercased, cleaned, tokenized)
3. TF-IDF vectorizer converts text to numerical features
4. Logistic Regression model predicts sentiment + confidence
5. Result is displayed in real time

---

## 📌 Notes

- The sample dataset in `train.py` is minimal. Replace it with a larger dataset (e.g. [IMDB](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) or [Twitter Sentiment](https://www.kaggle.com/datasets/kazanova/sentiment140)) for better accuracy.

---

*Built by Muhammed Anshad M*
