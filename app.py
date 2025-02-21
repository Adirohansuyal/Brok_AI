import json
import streamlit as st

# Load the JSON data
def load_data(json_path="data/website_data.json"):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["data"]

# Find answer from JSON
def get_answer(question, qa_data):
    for item in qa_data:
        if question.lower() in item["question"].lower():
            return item["answer"]
    return "Sorry, I couldn't find an answer to your question."

# Streamlit UI
def main():
    st.title("BIAS Q&A Bot by Adi🤖")
    
    st.write("Ask me anything about BIAS!")

    # Load data
    qa_data = load_data()

    # User Input
    user_question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_question.strip():
            answer = get_answer(user_question, qa_data)
            st.write(f"**Answer:** {answer}")
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
