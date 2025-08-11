import streamlit as st
import pandas as pd
import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

# Download necessary NLTK data (this might run multiple times on Streamlit Cloud,
# but it's fine as it only downloads if not present)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except nltk.downloader.DownloadError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')


# --- Text Preprocessing Functions (from your Jupyter Notebook) ---

def clean_text(input_text):
    """
    Cleans the input text by removing numbers, special characters,
    converting to lowercase, and handling specific patterns.
    """
    input_text = re.sub(r'\d+', '', input_text)  # Remove numbers
    # Remove unwanted patterns like (Reuters) -
    input_text = re.sub(r'.* \(Reuters\) - |\ (Reuters\) - ', '', input_text)
    # Remove words starting with @ and source links, timestamp patterns
    input_text = re.sub(r'( (@\w+ and @\w+)|@\w+| \[\d{4} EST\]| -+ |:|Source link: \(.+\))', '', input_text)
    input_text = unicodedata.normalize('NFKD', input_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    input_text = re.sub(r'[^a-zA-Z]', ' ', input_text)  # Keep only alphabetic characters
    input_text = re.sub(r'^\\s*|\\s\\s*', ' ', input_text).strip()  # Replace multiple spaces with single space
    input_text = re.sub(r"U S", "US", input_text)  # Correct "U S" to "US"
    input_text = input_text.lower()  # Convert to lowercase
    return input_text

def tokenize_text(input_text):
    """
    Tokenizes the input text and removes stopwords.
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(input_text)
    return [word for word in tokens if word not in stop_words]

def get_wordnet_pos(tag):
    """
    Maps NLTK POS tags to WordNet POS tags for lemmatization.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_tokens(token_list):
    """
    Lemmatizes a list of tokens based on their POS tags.
    """
    lemmatizer = WordNetLemmatizer()
    tagged_tokens = pos_tag(token_list)
    return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tagged_tokens]

def preprocess_text_pipeline(text):
    """
    Applies the full preprocessing pipeline.
    """
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    lemmatized = lemmatize_tokens(tokens)
    return " ".join(lemmatized)

# --- Model Loading and Prediction (Mocked for now) ---

# In a real application, you would load your trained tokenizer, model, and LabelEncoder here.
# Example:
# try:
#     with open('tokenizer.pkl', 'rb') as f:
#         tokenizer = pickle.load(f)
#     model = tf.keras.models.load_model('rnn_model.h5')
#     with open('label_encoder.pkl', 'rb') as f:
#         label_encoder = pickle.load(f)
#     MAX_SEQUENCE_LENGTH = model.input_shape[1] # Or a predefined constant
#     VOCAB_SIZE = len(tokenizer.word_index) + 1 # Or a predefined constant
#     MODEL_LOADED = True
# except Exception as e:
#     st.error(f"Error loading model components: {e}. Please ensure 'tokenizer.pkl', 'rnn_model.h5', and 'label_encoder.pkl' are in the same directory.")
#     MODEL_LOADED = False

# Mock data and model components for demonstration
# In a real scenario, you would train these from your dataset (True.csv)
# For demonstration purposes, let's create a dummy tokenizer and encoder
# based on a small sample of expected words.
dummy_texts = [
    "the head of a conservative republican faction in the us congress",
    "transgender people will be allowed for the first time to enlist in the us military",
    "the special counsel investigation of links between russia and president trumps election campaign",
    "trump campaign adviser george papadopoulos told an australian diplomat",
    "president donald trump called on the us postal service on friday",
    "nato allies on tuesday welcomed president donald trump",
    "lexisnexis a provider of legal regulatory and business information",
    "in the shadow of disused soviet era factories"
]
tokenizer = Tokenizer(num_words=10000, oov_token="<unk>") # A reasonable vocabulary size
tokenizer.fit_on_texts(dummy_texts) # Fit on some representative text
MAX_SEQUENCE_LENGTH = 100 # Example max sequence length

label_encoder = LabelEncoder()
label_encoder.fit(['politicsNews', 'worldnews']) # Fit on your actual classes

# Create a dummy RNN model structure for demonstration.
# In a real app, you'd load your actual trained model.
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=MAX_SEQUENCE_LENGTH),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
MODEL_LOADED = True # Set to True because we're mocking it

def predict_news_type(text):
    """
    Predicts the news type using the loaded model.
    """
    if not MODEL_LOADED:
        return "Model not loaded. Cannot make predictions."

    processed_text = preprocess_text_pipeline(text)
    
    # Convert text to sequence, pad, and predict
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
    prediction = model.predict(padded_sequence)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
    
    # Get confidence for all classes
    confidence = {label: prob for label, prob in zip(label_encoder.classes_, prediction)}
    
    return predicted_label, confidence

# --- Streamlit App Layout ---

st.set_page_config(page_title="Fake News Classifier", layout="centered")

st.markdown(
    """
    <style>
    .main-header {
        font-size: 3em;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .subheader {
        font-size: 1.5em;
        color: #333333;
        text-align: center;
        margin-bottom: 30px;
    }
    .stTextArea label {
        font-size: 1.2em;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 1.1em;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: #e0f2f7;
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
        border: 2px solid #2196F3;
    }
    .confidence-box {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-header">📰 Fake News Classifier 📰</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Enter a news article to classify it as Politics News or World News.</p>', unsafe_allow_html=True)

# Text input for the user
user_input = st.text_area(
    "Enter the news article text here:",
    height=250,
    placeholder="Paste your news article here...",
)

if st.button("Classify Article"):
    if user_input:
        # Preprocess the text
        with st.spinner("Preprocessing and classifying..."):
            predicted_label, confidence = predict_news_type(user_input)
            
        st.markdown(f'<div class="prediction-box"><h3>Prediction: <span style="color:#2196F3;">{predicted_label}</span></h3></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="confidence-box"><h4>Confidence Scores:</h4>', unsafe_allow_html=True)
        for label, prob in confidence.items():
            st.markdown(f'- **{label}**: {prob:.2%}')
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Preprocessed Text:")
        st.info(preprocess_text_pipeline(user_input))
    else:
        st.warning("Please enter some text to classify.")

st.markdown("---")
st.markdown("This app uses a text classification model. The prediction is based on the trained model and the preprocessing steps derived from your Jupyter Notebook.")
st.markdown("For a real-world application, you would train and save your actual TensorFlow/Keras model and tokenizer, then load them into this Streamlit app.")

