import streamlit as st
from predict import predict_sentiment

st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis Web App")
st.markdown("Classify any text as **Positive**, **Negative**, or **Neutral** using Machine Learning.")
st.markdown("---")

user_input = st.text_area("Enter your review or text below:", height=150,
                           placeholder="e.g. The product was absolutely amazing! Highly recommend.")

if st.button("Analyse Sentiment", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text before analysing.")
    else:
        with st.spinner("Analysing..."):
            label, confidence = predict_sentiment(user_input)

        color_map = {"Positive": "green", "Negative": "red", "Neutral": "orange"}
        emoji_map = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}

        st.markdown("### Result")
        col1, col2 = st.columns(2)
        col1.metric("Sentiment", f"{emoji_map[label]}  {label}")
        col2.metric("Confidence", f"{confidence:.1f}%")

        st.success(f"The sentiment of your text is **{label}** with **{confidence:.1f}%** confidence.")

st.markdown("---")
st.caption("Built with Python · Scikit-learn · NLTK · Streamlit")
