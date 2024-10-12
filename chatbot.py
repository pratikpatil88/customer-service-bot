import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json

# Loads the model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Loads the training data
with open('training_data.json', 'r') as file:
    training_data = json.load(file)

# Function to encode sentences into an embedding
def encode_sentence(sentence):
    return model.encode(sentence, convert_to_tensor=True, clean_up_tokenization_spaces=True)

# Function to find best match by iterating through training data, calculating the cosine similarity score between the query and the question, and returns a best match
def find_best_match(query, training_data):
    query_embedding = encode_sentence(query)
    best_match = None
    best_score = float('-inf')

    for question, answer in training_data.items():
        question_embedding = encode_sentence(question)
        score = util.pytorch_cos_sim(query_embedding, question_embedding).item()
        if score > best_score:
            best_score = score
            best_match = question

    return best_match if best_score > 0.8 else None  # Adjusted threshold

# Defines the main chatbot response function
def chatbot_response(query):
    best_match = find_best_match(query, training_data)
    if best_match:
        return training_data[best_match]
    else:
        return "I'm not sure how to respond to that."

# Defines the submit function which handles user's input and generates a response
def submit():
    user_query = st.session_state.user_query
    response = chatbot_response(user_query)
    st.session_state.conversation.insert(0, ("You", user_query))
    st.session_state.conversation.insert(1, ("Assistant", response))
    st.session_state.user_query = ""  # Clear the input field

# Streamlit app configuration
st.set_page_config(page_title="Customer Support", page_icon="🤖", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        padding: 10px;
        font-size: 16px;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stMarkdown {
        font-size: 16px;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and Introduction
st.title("Welcome to Virtual Support Assistant")
st.markdown("### Hello! I'm here to help you with your queries. Type your question below:")

# Initialize conversation in session state if not already present
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Input field for user query
st.text_input("You: ", key="user_query", on_change=submit)

# Display conversation history
for speaker, message in st.session_state.conversation:
    st.markdown(f"**{speaker}:** {message}")

# Styling for input and markdown
st.markdown(
    """
    <style>
    .stTextInput > div > div > input {
        padding: 10px;
        font-size: 16px;
    }
    .stMarkdown {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
