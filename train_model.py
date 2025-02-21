from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import json
import nltk
import string
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
import numpy as np

# Download stopwords
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # Using SBERT

def preprocess_text(text):
    """Preprocess input text: lowercase, remove punctuation & stopwords, apply lemmatization."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in STOPWORDS]
    return " ".join(words)

def load_dataset():
    """Load dataset from data/website_data.json and ensure it's a list of dictionaries."""
    with open("data/website_data.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if isinstance(dataset, dict):  # Adjust based on JSON structure
        dataset = dataset.get("data", [])  
    
    if not isinstance(dataset, list):
        raise ValueError("Dataset format is incorrect. Expected a list of dictionaries.")

    return dataset

def train_model():
    dataset = load_dataset()

    # Extract questions & answers
    try:
        questions = [preprocess_text(item["question"]) for item in dataset]
        answers = [item["answer"] for item in dataset]
    except KeyError:
        raise KeyError("Dataset items must contain 'question' and 'answer' keys.")

    # Convert questions to SBERT embeddings
    question_embeddings = sbert_model.encode(questions)

    # Save embeddings and answers for later retrieval
    np.save("model/question_embeddings.npy", question_embeddings)
    with open("model/answers.pkl", "wb") as f:
        pickle.dump(answers, f)

    print("✅ Model trained and saved successfully!")

def find_best_match(user_input):
    """Find the best matching question for the user's input using fuzzy matching & embeddings."""
    user_input = preprocess_text(user_input)

    # Load trained embeddings & answers
    question_embeddings = np.load("model/question_embeddings.npy")
    with open("model/answers.pkl", "rb") as f:
        answers = pickle.load(f)

    # Compute similarity using SBERT embeddings
    user_embedding = sbert_model.encode([user_input])[0]
    similarities = np.dot(question_embeddings, user_embedding)  # Cosine similarity
    best_match_index = np.argmax(similarities)

    # Use fuzzy matching as a backup
    dataset = load_dataset()
    questions = [preprocess_text(item["question"]) for item in dataset]
    fuzzy_match, score = process.extractOne(user_input, questions)
    fuzzy_index = questions.index(fuzzy_match)

    # Return the best-matching answer
    if similarities[best_match_index] > 0.7:  # SBERT Confidence Threshold
        return answers[best_match_index]
    elif score > 70:  # FuzzyWuzzy Confidence Threshold
        return answers[fuzzy_index]
    else:
        return "I'm sorry, I couldn't find a relevant answer. Could you rephrase your question?"

if __name__ == "__main__":
    train_model()
