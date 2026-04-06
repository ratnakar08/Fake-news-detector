import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set up page configurations for a modern, clean layout
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
)

# -------------------------------------------------------------
# 1. Text Preprocessing Function
# -------------------------------------------------------------
def clean_text(text):
    """Cleans the input text by converting to lowercase and keeping only letters."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# -------------------------------------------------------------
# 2. Cache Data Loading and Model Training for Performance
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_and_train_model():
    """Loads datasets, cleans text, and trains the TF-IDF Vectorizer and Logistic Regression Model."""
    try:
        # Load data
        fake = pd.read_csv("Fake.csv")
        real = pd.read_csv("True.csv")

        # Assign labels: 0 for Fake, 1 for Real
        fake["label"] = 0
        real["label"] = 1

        # Combine datasets
        data = pd.concat([fake, real], ignore_index=True)

        # Preprocess text
        data["text"] = data["text"].apply(clean_text)
        
        # Prepare Features and Target
        x = data["text"]
        y = data["label"]

        # Improve TF-IDF Vectorization
        # - stop_words='english': Removes common english words
        # - ngram_range=(1, 2): Considers individual words and pairs of adjacent words
        # - max_df=0.7: Ignores terms that appear in more than 70% of the documents
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.7
        )
        x_vec = vectorizer.fit_transform(x)

        # Train Logistic Regression Model
        model = LogisticRegression(max_iter=500)
        model.fit(x_vec, y)

        return vectorizer, model
    except FileNotFoundError:
        st.error("Error: Could not find 'Fake.csv' or 'True.csv'. Please ensure both files are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during model training: {e}")
        st.stop()

# -------------------------------------------------------------
# 3. Prediction Logic
# -------------------------------------------------------------
def predict_news(news, vectorizer, model):
    """Processes the text and returns whether it's real, along with confidence percentage."""
    # Clean the input text
    cleaned_news = clean_text(news)
    
    # Vectorize the text
    vector = vectorizer.transform([cleaned_news])
    
    # Make Prediction
    prediction = model.predict(vector)[0]
    
    # Calculate Confidence Score using predict_proba()
    # predict_proba returns probabilities for each class (Fake=0, Real=1), so we take the max
    confidence = model.predict_proba(vector)[0].max() * 100
    
    # Return boolean (True if Real, False if Fake) and confidence score
    return prediction == 1, confidence


# -------------------------------------------------------------
# 4. Streamlit UI Build
# -------------------------------------------------------------
# 4a. Header Section: Title & Description
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: gray;'>Verify the authenticity of news articles instantly using Machine Learning.</p>", unsafe_allow_html=True)
st.divider()

# Load the model directly upon script execution (st.cache_resource drastically reduces redraw times)
with st.spinner("Initializing Model & Loading Dataset... (This takes a moment on the first run)"):
    vectorizer, model = load_and_train_model()

# 4b. Body Section: Columns for Input (Left) and Result (Right)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input Article")
    # Input field for users to paste news text
    input_text = st.text_area("Paste your news text here...", height=250, label_visibility="collapsed")
    
    # Action button with a clear label spanning full width
    check_btn = st.button("Check Authenticity", use_container_width=True)

with col2:
    st.subheader("Analysis Result")
    
    # Wait for the button click
    if check_btn:
        # Warn user if input is empty
        if not input_text.strip():
            st.warning("⚠️ Please paste some text into the input field to verify.")
        else:
            # Show a loading spinner specifically for the prediction process
            with st.spinner("Analyzing text..."):
                # Run the prediction
                is_real, confidence = predict_news(input_text, vectorizer, model)
                
                # Render Colored Output based on prediction
                if is_real:
                    st.success(f"**Real News** (Confidence: {confidence:.0f}%)", icon="🟢")
                else:
                    st.error(f"**Fake News** (Confidence: {confidence:.0f}%)", icon="🔴")
    else:
        st.info("👈 Enter text on the left and click 'Check Authenticity' to see the result.")