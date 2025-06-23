import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("📦 AI Text Classifier for Product Restrictions")

# Sample training data
data = {
    'Text': [
        "This product is restricted by Amazon",
        "Amazon does not allow this product",
        "WPC certificate is required for this item",
        "Telecom license is mandatory for use in India",
        "This item is banned from import to India",
        "Safe to use in India",
        "No restrictions, free import",
        "Complies with Indian telecom rules"
    ],
    'Label': [
        "Amazon.Restriction",
        "Amazon.Restriction",
        "Telecom.Restriction",
        "Telecom.Restriction",
        "Indian.Import.Ban",
        "Allowed",
        "Allowed",
        "Allowed"
    ]
}

# Train model
df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Text'])
model = MultinomialNB()
model.fit(X, df['Label'])

# User Input
st.subheader("🔍 Check Product Description or Title")
user_input = st.text_area("Enter product text here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid product description.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        # Display output with icon
        icon = {
            "Allowed": "✅",
            "Amazon.Restriction": "🚫",
            "Telecom.Restriction": "📡",
            "Indian.Import.Ban": "❌"
        }.get(prediction, "ℹ️")

        st.success(f"{icon} Prediction: **{prediction}**")
