import pandas as pd
import numpy as np
import joblib
import nltk
import re
import string

from sklearn.metrics import f1_score, precision_score, confusion_matrix, accuracy_score, classification_report, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

# load the dataset
df = pd.read_csv("FA-KES-Dataset.csv", encoding="latin-1")

# data exploration
print(df.head())   

# data preprocessing


# text cleaning
stops_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stops_words])
    return text 
df["text"] = df["article_title"].fillna("") + " " + df["article_content"].fillna("")
df["cleaned_text"] = df["text"].apply(clean_text)

#split into test and training 
X = df["cleaned_text"]
y = df["labels"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# model loading
models = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "naive_bayes": MultinomialNB(),
    "svm": LinearSVC()
}
# model evaluation and training
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    print(f"Model: {name}")
    print(f"Accuracy: {results[name]['accuracy']}")
    print(f"Precision: {results[name]['precision']}")
    print(f"Recall: {results[name]['recall']}")
    print(f"F1 Score: {results[name]['f1_score']}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
results_df = pd.DataFrame(results).T.sort_values(by="f1_score", ascending=False)

best_model_name = max(results_df.index, key=lambda x: results_df.loc[x]["f1_score"])
best_model = models[best_model_name]

if best_model_name in ["logistic_regression", "random_forest", "decision_tree", "naive_bayes", "svm"]:
    best_model = models[best_model_name]
print(f"Best model: {best_model_name} with F1 Score: {results_df.loc[best_model_name]['f1_score']}")
# save the model and vectorizer
joblib.dump(best_model, "best_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
