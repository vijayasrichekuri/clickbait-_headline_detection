import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Page config
st.set_page_config(page_title="AI Clickbait Detector", layout="wide")

# Custom CSS
st.markdown("""
<style>
body {
background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
color:white;
}

.big-title {
font-size:50px;
text-align:center;
font-weight:bold;
}

.subtitle {
text-align:center;
font-size:20px;
margin-bottom:40px;
}

.stButton>button {
background-color:#ff4b4b;
color:white;
font-size:18px;
border-radius:10px;
padding:10px 30px;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title
st.markdown("<div class='big-title'>🧠 AI Clickbait Headline Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Machine Learning + NLP Based News Analysis</div>", unsafe_allow_html=True)

# Input
headline = st.text_input("Enter News Headline")

if st.button("Detect Headline"):

    vector = vectorizer.transform([headline])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]

    clickbait_prob = probability[1]*100
    normal_prob = probability[0]*100

    # Result
    if prediction == 1:
        st.error("⚠️ CLICKBAIT HEADLINE DETECTED")
    else:
        st.success("✅ GENUINE HEADLINE")

    # Probability Chart
    st.subheader("Prediction Probability")

    labels = ["Clickbait","Not Clickbait"]
    values = [clickbait_prob, normal_prob]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Probability (%)")

    st.pyplot(fig)

# Word Cloud
st.subheader("Common Clickbait Words")

try:
    data = pd.read_csv("clickbait_data.csv")

    clickbait_text = " ".join(data[data['clickbait']==1]['headline'])

    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(clickbait_text)

    fig2, ax2 = plt.subplots()
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis("off")

    st.pyplot(fig2)

except:
    st.warning("Dataset not found for wordcloud.")

# Confusion Matrix
st.subheader("Model Performance")

try:

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    X = data['headline']
    y = data['clickbait']

    X_vec = vectorizer.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec,y,test_size=0.2)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)

    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    st.pyplot(fig3)

except:
    st.warning("Unable to generate confusion matrix.")