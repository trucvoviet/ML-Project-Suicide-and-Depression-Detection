import streamlit as st
import joblib
import re
import string
import nltk
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

# Load models
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

stop = set(stopwords.words("english"))


def expand_contractions(text):
    try:
        return contractions.fix(text)
    except:
        return text


def preprocess_text(text):

    wl = WordNetLemmatizer()

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    text = expand_contractions(text)

    emoji_clean = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)

    text = emoji_clean.sub(r"", text)

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\.(?=\S)", ". ", text)

    text = "".join(
        [char.lower() for char in text if char not in string.punctuation]
    )

    text = " ".join(
        [wl.lemmatize(word) for word in text.split()
         if word not in stop and word.isalpha()]
    )

    return text


# Streamlit UI
st.title("🧠 Suicide & Self-harm Intent Detection")
st.write("Enter a sentence to analyze suicidal and self-harm intenttion")

user_input = st.text_area("Enter text")

if st.button("Analyze"):

    cleaned = preprocess_text(user_input)

    vectorized = tfidf.transform([cleaned])

    probs = model.predict_proba(vectorized)[0]

    classes = label_encoder.classes_

    suicide_prob = round(probs[1] * 100, 2)
    nonsuicide_prob = round(probs[0] * 100, 2)

    prediction = label_encoder.inverse_transform([probs.argmax()])[0]

    st.subheader("Prediction")

    st.write(f"**Suicide probability:** {suicide_prob}%")
    st.write(f"**Non-suicide probability:** {nonsuicide_prob}%")

    st.success(f"Final prediction: **{prediction}**")