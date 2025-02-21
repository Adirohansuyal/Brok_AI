from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from dataset_loader import load_dataset

def train_model():
    dataset = load_dataset()
    questions = [item["question"] for item in dataset]
    answers = [item["answer"] for item in dataset]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)

    model = LogisticRegression()
    model.fit(X, answers)

    with open("model/model.pkl", "wb") as f:
        pickle.dump((vectorizer, model), f)

    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()
