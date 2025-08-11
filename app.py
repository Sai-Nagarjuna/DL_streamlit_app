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

# --- Global variables for tokenizer, model, and label_encoder ---
tokenizer = None
model = None
label_encoder = None
MAX_SEQUENCE_LENGTH = 100 # Default, will be updated based on data
MODEL_INITIALIZED = False

# --- Model Initialization on App Startup ---
# This function will load data, fit the tokenizer/encoder, and define the model.
# In a production app, you would load pre-trained artifacts.
@st.cache_resource
def initialize_model_components():
    global tokenizer, model, label_encoder, MAX_SEQUENCE_LENGTH, MODEL_INITIALIZED

    st.write("Initializing model components... This may take a moment.")
    
    try:
        # Load the dataset
        df = pd.read_csv("True.csv")
        
        # Preprocess the text column
        df['processed_text'] = df['text'].apply(preprocess_text_pipeline)
        
        # Initialize and fit Tokenizer
        # num_words can be adjusted, 10000 is a common starting point
        tokenizer = Tokenizer(num_words=10000, oov_token="<unk>")
        tokenizer.fit_on_texts(df['processed_text'])

        # Determine MAX_SEQUENCE_LENGTH
        # Use the average or a percentile of sequence lengths for practical purposes
        # For simplicity, we'll keep a fixed value, but in real scenario, it should be
        # based on training data statistics
        # max_len = max([len(x.split()) for x in df['processed_text']])
        # MAX_SEQUENCE_LENGTH = min(max_len, 250) # Cap at 250 words or actual max_len

        # Initialize and fit LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(df['subject'].unique()) # Fit on all unique subjects

        VOCAB_SIZE = len(tokenizer.word_index) + 1
        NUM_CLASSES = len(label_encoder.classes_)

        # Define the RNN model structure (without training)
        # This structure matches the one implied by your notebook's classification report
        model = Sequential([
            Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_SEQUENCE_LENGTH),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        MODEL_INITIALIZED = True
        st.success("Model components initialized successfully! You can now classify articles.")

    except Exception as e:
        st.error(f"Error initializing model components. Please ensure 'True.csv' is in the same directory. Error: {e}")
        MODEL_INITIALIZED = False

initialize_model_components() # Call the initialization function

def predict_news_type(text):
    """
    Predicts the news type using the loaded model.
    """
    if not MODEL_INITIALIZED:
        return "Model not ready. Please wait for initialization or check for errors."

    processed_text = preprocess_text_pipeline(text)
    
    # Convert text to sequence, pad, and predict
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Predict with the model
    # Note: If the model hasn't been truly trained, predictions will be random.
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
    key="news_input" # Added a key for better re-rendering
)

if st.button("Classify Article"):
    if user_input:
        if MODEL_INITIALIZED:
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
            st.warning("Model components are still initializing or encountered an error. Please wait or refresh.")
    else:
        st.warning("Please enter some text to classify.")

st.markdown("---")
st.markdown("This app uses a text classification model. The tokenizer and label encoder are fitted on the provided `True.csv` dataset upon startup, and a Keras RNN model structure is defined. **Note: The model itself is *not* trained on the data within this Streamlit app due to performance considerations; only its structure is set up.** For production, you would pre-train and save your model and related artifacts.")

