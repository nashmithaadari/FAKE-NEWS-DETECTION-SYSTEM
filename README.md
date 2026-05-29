FAKE-NEWS-DETECTION-SYSTEM
In today's digital age, misinformation spreads rapidly across social media platforms, making it difficult for users to distinguish between real and fake news. The Fake News Detection System addresses this problem by using Machine Learning algorithms to automatically analyze news article text. The system preprocesses the text, converts it into numerical features using TF-IDF vectorization, and then applies a Random Forest classifier to predict whether the news is REAL or FAKE. The results are displayed through an intuitive web interface with confidence scores, helping users make informed decisions about the content they consume.
Machine Learning based Fake News Detection System with 99.7% accuracy.
 Fake News Detection System

 🎯 Objective
To develop a machine learning system that classifies news articles as REAL or FAKE using Natural Language Processing (NLP) and classification algorithms.

 
🛠️ Tech Stack
| Category | Technologies |
|----------|--------------|
| Backend | Python, Flask |
| ML Libraries | scikit-learn, pandas, numpy, joblib |
| Frontend | HTML5, CSS3, JavaScript |
| Algorithms | Random Forest, Logistic Regression, TF-IDF |


 ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Text Preprocessing** | Cleans text (lowercase, remove punctuation/numbers) |
| 📊 **TF-IDF Vectorization** | Converts text to numerical features with n-gram support |
| 🧠 **Multiple Models** | Random Forest (99.72% accuracy) + Logistic Regression |
| 🌐 **Web Interface** | Flask-based responsive UI with confidence scores |
| 📱 **Mobile Friendly** | Responsive design works on all devices |
| 🌙 **Dark Mode** | Auto-detects system theme preference |
| ⚡ **Real-time Predictions** | Instant classification with confidence percentages |

 📊 Dataset

| Parameter | Details |
|-----------|---------|
| **Source** | ISOT Fake News Dataset (Kaggle) |
| **Total Articles** | 44,908 |
| **Fake News** | 23,481 articles |
| **Real News** | 21,427 articles |
| **Features** | title, text, subject, date |

---

 📁 Project Structure
FAKE-NEWS-DETECTION-SYSTEM/
│
├── 📄 app.py # Flask web application (main entry point)
├── 📄 model_training.py # ML model training script
├── 📄 train_model.py # Data preprocessing script
├── 📄 requirements.txt # Python dependencies list
├── 📄 README.md # Project documentation
├── 📄 .gitignore # Git ignore file
│
├── 📁 data/ # Dataset directory (not on GitHub due to size)
│ ├── 📄 Fake.csv # 23,481 fake news articles
│ ├── 📄 True.csv # 21,427 real news articles
│ └── 📄 processed_news.csv # Preprocessed dataset
│
├── 📁 models/ # Trained models (not on GitHub due to size)
│ ├── 📄 fake_news_model.pkl # Random Forest model (99.72% accuracy)
│ └── 📄 tfidf_vectorizer.pkl # TF-IDF vectorizer
│
└── 📁 templates/ # HTML templates
└── 📄 index.html # Main web interface

 📊 Code Statistics

| Component | Lines of Code | Files |
|-----------|--------------|-------|
| Python Backend | ~350 lines | 3 files |
| HTML/CSS/JS Frontend | ~400 lines | 1 file |
| Total | ~750 lines | 4 core files |

 📊 Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Random Forest** | **99.72%** | **1.00** | **1.00** | **1.00** | 45 seconds |
| Logistic Regression | 98.92% | 0.99 | 0.99 | 0.99 | 10 seconds |

 
