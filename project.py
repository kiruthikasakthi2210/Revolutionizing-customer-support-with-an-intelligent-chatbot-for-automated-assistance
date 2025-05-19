"""
Intelligent Chatbot for Customer Support - Phase Two
Using Pandas, NumPy, Scikit-Learn, TensorFlow, NLTK, SpaCy
with visualizations via Seaborn, Matplotlib, and Plotly.

This example script covers:
- Loading sample intent dataset
- Text preprocessing with NLTK and SpaCy
- Feature extraction (TF-IDF)
- Intent classification with TensorFlow neural network
- Visualization of intent distribution and training history

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import string
import re

# Download required NLTK packages
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Spacy English model
nlp = spacy.load('en_core_web_sm')

# Sample intents dataset (in real case, replace with your data)
data = {
    'text': [
        "Hello, I need help with my order",
        "What is your return policy?",
        "I want to track my shipment",
        "Can you assist me with billing?",
        "Thank you for your help",
        "When will my package arrive?",
        "I want to cancel my order",
        "Do you offer international shipping?",
        "My refund hasn't arrived yet",
        "How do I reset my password?"
    ],
    'intent': [
        "greeting",
        "return_policy",
        "track_order",
        "billing_help",
        "thanks",
        "order_status",
        "order_cancel",
        "shipping_info",
        "refund_status",
        "password_reset"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Visualize intent distribution (Seaborn)
plt.figure(figsize=(8,4))
sns.countplot(x='intent', data=df, palette='viridis')
plt.title('Intent Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Text preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize using Spacy
    doc = nlp(text)
    # Lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(token.text) for token in doc if token.text not in stop_words and not token.is_space]
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)

# Features - TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['processed_text']).toarray()

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['intent'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple TensorFlow Neural Network for Intent Classification
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=4,
                    callbacks=[early_stop],
                    verbose=2)

# Plot training history using Plotly
history_df = pd.DataFrame(history.history)

fig = px.line(history_df, y=['accuracy', 'val_accuracy'], title='Training and Validation Accuracy')
fig.show()

fig_loss = px.line(history_df, y=['loss', 'val_loss'], title='Training and Validation Loss')
fig_loss.show()

# Sample prediction function
def predict_intent(text):
    processed = preprocess_text(text)
    vect = tfidf.transform([processed]).toarray()
    pred = model.predict(vect)
    intent = le.inverse_transform([np.argmax(pred)])[0]
    confidence = np.max(pred)
    return intent, confidence

# Test sample
sample_text = "Can you help me with tracking my package?"
predicted_intent, confidence = predict_intent(sample_text)
print(f"User Input: {sample_text}")
print(f"Predicted Intent: {predicted_intent} (Confidence: {confidence:.2f})")

"""
This code is a framework prototype to be extended with:
- Larger training datasets
- More advanced NLP with SpaCy pipelines/custom models
- Dialog management and conversation state handling
- Integration with chatbot platforms or APIs
"""

