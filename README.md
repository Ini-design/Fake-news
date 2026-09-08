# Fake News Detection Project

## Overview
This project builds a machine learning system that classifies news articles as likely real or likely fake. The goal is to help users quickly assess whether a piece of text is trustworthy based on language patterns, tone, and content quality.

The system uses text preprocessing, TF-IDF feature extraction, and several supervised learning models. It compares model performance using standard classification metrics, f1_score and saves the best-performing model for use in a Streamlit web app.

---

## The Problem
With the rapid growth of online media, misinformation spreads much faster than reliable reporting. Fake news often looks convincing because it uses emotional, sensational, or urgent language.

People may not have time to verify every article manually, and traditional fact-checking is slow and not scalable. There is a clear need for a fast, automated way to flag suspicious content before it spreads widely.

This project addresses that problem by creating a text-based classifier that can analyze article content and estimate whether it is likely real or fake.

---

## How I Solved It
I built a realistic machine learning pipeline that follows the standard workflow for text classification:

1. Loaded the dataset from FA-KES-Dataset.csv
2. Combined article title and article content into a single text field
3. Cleaned the text by:
   - converting to lowercase
   - removing URLs and usernames
   - removing punctuation and non-letter characters
   - removing stop words
   - lemmatizing words to their root form
4. Converted text into numerical features using TF-IDF vectorization
5. Trained multiple classifiers:
   - Logistic Regression
   - Random Forest
   - Decision Tree
   - Naive Bayes
   - Support Vector Machine (LinearSVC)
6. Evaluated each model with accuracy, precision, recall, and F1-score
7. Selected the model with the highest F1-score
8. Saved the best model and vectorizer for deployment
9. Built a user-facing web app with Streamlit for real-time predictions

This approach works well because fake news often has repeated patterns such as exaggerated language, unusual phrasing, and strong emotional cues, which machine learning models can learn from text features.

---

## Technologies Used
- Python
- Pandas for dataset handling
- NumPy for numerical operations
- NLTK for text cleaning and lemmatization
- scikit-learn for:
  - TF-IDF vectorization
  - train/test splitting
  - model training
  - evaluation metrics
- Joblib for saving model artifacts
- Streamlit for the interactive dashboard

---

## Project Structure
- main.py - training pipeline, data cleaning, model comparison, model selection
- app.py - Streamlit interface for predictions
- FA-KES-Dataset.csv - dataset used for training and validation
- best_model.pkl - trained model saved after training
- vectorizer.pkl - fitted TF-IDF vectorizer saved for inference
- requirements.txt - required dependencies

---

## Model Performance
The project does not rely on a single model. Instead, it trains several models and compares them on the same validation set.

The evaluation metrics used are:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

Why F1-score matters:
- Fake-news detection is often imbalanced, so accuracy alone can be misleading.
- F1-score balances precision and recall, which makes it a better metric for identifying suspicious content without too many false positives or false negatives.

The code automatically selects the model with the best F1-score and stores it for prediction. This means the final performance depends on the actual dataset version and split, but the project is designed to choose the strongest model for that dataset automatically.

In short, the model is evaluated thoroughly and optimized for the right metric rather than just selecting the first model that appears to work.

---

## How to Run It
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python main.py
```
This script:
- loads the dataset
- preprocesses the text
- tests multiple models
- selects the best one
- saves the model and vectorizer as .pkl files

### 3. Launch the app
```bash
streamlit run app.py
```
Then open the local URL shown in the terminal, typically:
```bash
Local URL: http://localhost:8501
Network URL: http://10.251.201.13:8501
```

---

## Where to Test It
You can test the solution in several ways:

### Local testing
- Paste an article into the Streamlit text box
- Click Predict
- Review the prediction, confidence score, statistics, and explanation

### Sample testing
The app includes sample articles that represent likely real and likely fake cases. These can be used to quickly verify how the model responds to different writing styles.

### Real-world testing
Use the app with:
- short news snippets
- full article bodies
- blog posts or social media text
- headlines and summaries

This is useful for testing how the model behaves on different article lengths and writing patterns.

---

## How Well It Performs
The project is designed to be practical rather than just experimental. It combines data cleaning, model search, and deployment so that the final model is more than a notebook experiment.

Real-world performance depends on:
- quality of the dataset
- text preprocessing choices
- how balanced the labels are
- whether the article is short or long
- how much noisy or informal language appears in the content

Because the system compares multiple models and keeps the best one by F1-score, it is a strong baseline for text-based fake-news detection. It is especially effective when the article contains clear patterns of misinformation such as sensational wording, emotional tone, or repetitive claims.

---

## How to Tune It
There are several ways to improve or tune this project:

### Text preprocessing
- try different stop word settings
- use stemming instead of lemmatization
- add custom rules for suspicious words or phrases
- experiment with bigger n-grams

### Feature engineering
- increase TF-IDF max features
- use character-level n-grams
- experiment with unigrams and bigrams together

### Model tuning
- adjust Logistic Regression regularization
- tune Random Forest depth and number of estimators
- change SVM C parameter
- improve Naive Bayes smoothing parameters

### Evaluation strategy
- use cross-validation instead of a single train/test split
- compare performance across multiple random seeds
- monitor class-specific precision and recall

### Deployment improvements
- add explanation features for stronger transparency
- include a confidence threshold
- add a blacklist or flagged keywords section
- create a batch-processing mode for multiple articles

---

## Future Improvements
- add more labeled fake-news datasets for stronger generalization
- support multilingual text detection
- include source credibility checks
- add URL and author analysis
- deploy the app to a cloud platform such as Streamlit Community Cloud or Render

---

## Conclusion
This project demonstrates how machine learning can be applied to a practical real-world problem: detecting fake news in article text. It combines strong preprocessing, multiple model comparisons, and a user-friendly interface to make classification accessible and easy to test.

The result is a working fake-news detection system that can be used as a starting point for further research, deployment, and improvement.

---

## Quick Start
```bash
pip install -r requirements.txt
python main.py
streamlit run app.py
```

Then open the app in your browser and begin testing article text.
