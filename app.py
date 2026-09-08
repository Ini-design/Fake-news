import re

import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Fake News Detection", page_icon=":newspaper:", layout="wide")


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model_artifacts()


def get_article_statistics(article_text):
    cleaned = article_text.strip()
    if not cleaned:
        return {
            "word_count": 0,
            "character_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
            "reading_time_minutes": 0.0,
            "suspicious_markers": [],
            "unique_word_ratio": 0.0,
        }

    words = re.findall(r"\b\w+\b", cleaned)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    suspicious_markers = [
        marker for marker in ["urgent", "breaking", "shocking", "miracle", "secret", "hoax", "exposed"]
        if marker in cleaned.lower()
    ]

    word_count = len(words)
    char_count = len(cleaned)
    sentence_count = max(len([s for s in sentences if s.strip()]), 1)
    avg_sentence_length = word_count / sentence_count
    reading_time_minutes = max(word_count / 200.0, 0.1)
    unique_word_ratio = len(set(w.lower() for w in words)) / word_count if word_count else 0.0

    return {
        "word_count": word_count,
        "character_count": char_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "reading_time_minutes": round(reading_time_minutes, 2),
        "suspicious_markers": suspicious_markers,
        "unique_word_ratio": round(unique_word_ratio, 2),
    }


def get_explanation(article_text, prediction, probabilities):
    explanation_lines = []
    score_label = "likely real" if prediction == 1 else "likely fake"
    explanation_lines.append(f"The model classified this article as {score_label} with a confidence of {max(probabilities) * 100:.1f}%.")

    stats = get_article_statistics(article_text)
    if stats["suspicious_markers"]:
        explanation_lines.append(
            "This article contains attention-grabbing wording such as: "
            + ", ".join(stats["suspicious_markers"])
            + ". Such phrases often appear in misleading or sensational content."
        )
    else:
        explanation_lines.append("The article does not show strong sensational language markers commonly linked with misleading content.")

    if prediction == 1:
        explanation_lines.append("Balanced wording and a more neutral tone generally support a higher credibility score.")
    else:
        explanation_lines.append("Short, sensational, and highly polarized wording can reduce trustworthiness and increase fake-news likelihood.")

    if hasattr(model, "coef_"):
        feature_names = np.asarray(vectorizer.get_feature_names_out())
        coef_values = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        top_indices = np.argsort(np.abs(coef_values))[-10:][::-1]
        top_terms = [(feature_names[i], round(float(coef_values[i]), 4)) for i in top_indices]
        explanation_lines.append("Top terms influencing the prediction: " + ", ".join(f"{term} ({value})" for term, value in top_terms[:5]))

    return explanation_lines


st.title("AI-powered News Credibility Analysis")
st.caption("A production-ready fake news detection app that analyzes article content and explains prediction outcomes.")

with st.sidebar:
    st.header("System Status")
    st.success("Model and vectorizer loaded successfully")
    st.info("Use the article text box below to assess credibility in real time.")

sample_articles = {
    "Likely Real": "Officials from the city health department said the hospital expanded emergency services after reviewing patient data and increasing staffing during the flu season. The report noted that treatment times improved and officials planned to publish updated figures after the next review. The statement was supported by public records and comments from hospital administrators.",
    "Likely Fake": "Wed 05, April 2017 Syria attack symptoms consistent with nerve agent use WHO. Victims of a suspected chemical attack in Syria appeared to show symptoms consistent with reaction to a nerve agent the World Health Organization said on Wednesday. ""Some cases appear to show additional signs consistent with exposure to organophosphorus chemicals a category of chemicals that includes nerve agents"" WHO said in a statement putting the death toll at at least 70. The United States has said the deaths were caused by sarin nerve gas dropped by Syrian aircraft. Russia has said it believes poison gas had leaked from a rebel chemical weapons depot struck by Syrian bombs. Sarin is an organophosporus compound and a nerve agent. Chlorine and mustard gas which are also believed to have been used in the past in Syria are not. A Russian Defence Ministry spokesman did not say what agent was used in the attack but said the rebels had used the same chemical weapons in Aleppo last year. The WHO said it was likely that some kind of chemical was used in the attack because sufferers had no apparent external injuries and died from a rapid onset of similar symptoms including acute respiratory distress. It said its experts in Turkey were giving guidance to overwhelmed health workers in Idlib on the diagnosis and treatment of patients and medicines such as Atropine an antidote for some types of chemical exposure and steroids for symptomatic treatment had been sent. A U.N. Commission of Inquiry into human rights in Syria has previously said forces loyal to Syrian President Bashar al-Assad have used lethal chlorine gas on multiple occasions. Hundreds of civilians died in a sarin gas attack in Ghouta on the outskirts of Damascus in August 2013. Assads government has always denied responsibility for that attack. Syria agreed to destroy its chemical weapons in 2013 under a deal brokered by Moscow and Washington. But Russia a Syrian ally and China have repeatedly vetoed any United Nations move to sanction Assad or refer the situation in Syria to the International Criminal Court. ""These types of weapons are banned by international law because they represent an intolerable barbarism"" Peter Salama Executive Director of the WHO Health Emergencies Programme said in the WHO statement.    ",
    "Balanced Example": "Local authorities announced a new road safety campaign after reviewing traffic accident reports from the previous year. The initiative includes better road markings, increased driver education, and public awareness material for schools and businesses. Officials said the program would be reviewed after six months to assess its impact.",
}

st.subheader("Try a sample article")
example_cols = st.columns(len(sample_articles))
for idx, (label, text) in enumerate(sample_articles.items()):
    with example_cols[idx]:
        st.markdown(f"### {label}")
        st.code(text, language="text")
        if st.button(f"Use {label} example", key=f"sample_{idx}"):
            st.session_state.news_article = text
            st.rerun()

st.write("---")
article_area = st.text_area(
    "Enter the news article here:",
    height=300,
    key="news_article",
    placeholder="Paste the full article here...",
)

if st.button("Predict", type="primary"):
    if article_area.strip() == "":
        st.warning("Please enter a news article to predict.")
    else:
        clean_text = article_area.strip()
        transformed_article = vectorizer.transform([clean_text])
        prediction = model.predict(transformed_article)[0]
        probabilities = model.predict_proba(transformed_article)[0] if hasattr(model, "predict_proba") else [0.0, 0.0]

        if hasattr(model, "predict_proba"):
            prob_display = probabilities[1] if prediction == 1 else probabilities[0]
            st.metric("Model confidence", f"{prob_display * 100:.1f}%")

        st.subheader("Prediction Result")
        if prediction == 0:
            st.error("The news article is likely to be fake.")
        else:
            st.success("The news article is likely to be real.")

        st.subheader("Article Statistics")
        stats = get_article_statistics(clean_text)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Words", stats["word_count"])
        col2.metric("Characters", stats["character_count"])
        col3.metric("Sentences", stats["sentence_count"])
        col4.metric("Reading Time", f"{stats['reading_time_minutes']} min")

        st.write("")
        col5, col6, col7 = st.columns(3)
        col5.metric("Avg. sentence length", stats["avg_sentence_length"])
        col6.metric("Unique word ratio", f"{stats['unique_word_ratio'] * 100:.1f}%")
        col7.metric("Suspicious markers", len(stats["suspicious_markers"]))

        if stats["suspicious_markers"]:
            st.caption("Detected markers: " + ", ".join(stats["suspicious_markers"]))

        st.subheader("Explanation")
        explanation_lines = get_explanation(clean_text, prediction, probabilities)
        for line in explanation_lines:
            st.write("- " + line)

        if hasattr(model, "coef_"):
            feature_names = np.asarray(vectorizer.get_feature_names_out())
            coef_values = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            top_indices = np.argsort(np.abs(coef_values))[-8:][::-1]
            top_terms = [(feature_names[i], float(coef_values[i])) for i in top_indices]

            st.write("")
            st.caption("Most influential words in this prediction")
            for term, weight in top_terms:
                st.write(f"• {term}: {weight:.4f}")


