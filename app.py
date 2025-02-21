import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
import nltk
import string

# Download stopwords and lemmatizer
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def preprocess_text(text):
    """Preprocess input text: lowercase, remove punctuation & stopwords, apply lemmatization."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in STOPWORDS]
    return " ".join(words)

def load_model():
    """Load trained embeddings and answers."""
    question_embeddings = np.load("model/question_embeddings.npy")
    with open("model/answers.pkl", "rb") as f:
        answers = pickle.load(f)
    return question_embeddings, answers

def find_best_match(user_input):
    """Find the best matching question for the user's input using semantic search & fuzzy matching."""
    user_input = preprocess_text(user_input)
    question_embeddings, answers = load_model()
    user_embedding = sbert_model.encode([user_input])[0]
    similarities = np.dot(question_embeddings, user_embedding)  # Cosine similarity
    best_match_index = np.argmax(similarities)

    if similarities[best_match_index] > 0.7:  # SBERT Confidence Threshold
        return answers[best_match_index]
    else:
        return "I'm sorry, I couldn't find a relevant answer. Could you rephrase your question?"

# Streamlit UI
st.title("AI-Powered Q&A Chatbot 🤖")
st.write("Enter a question below, and I'll try to find the best answer!")

user_question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if user_question.strip():
        answer = find_best_match(user_question)
        st.success(answer)
    else:
        st.warning("Please enter a valid question.")
