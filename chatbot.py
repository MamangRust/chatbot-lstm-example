import re
import random
import numpy as np
import pandas as pd
import json
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# Loading dummy school-related data
with open("intens.json", 'r') as f:
    data = json.load(f)

# Creating DataFrame from dummy data
df_school = pd.DataFrame(data['intents'])

# Load tokenizer and label encoder used during training
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer_school = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    lbl_enc_school = pickle.load(handle)

# Loading pre-trained model
model_path_school = 'my_model.keras'  # Update with the correct path
loaded_model_school = load_model(model_path_school)

# Streamlit application
response_user_school = []
response_bot_school = []

st.title("School ChatBot")
st.write("Welcome to the School ChatBot. Ask me anything about the school!")

# Display chat history
if "messages_school" not in st.session_state:
    st.session_state.messages_school = []

for message in st.session_state.messages_school:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and chat processing
prompt_school = st.chat_input("Type your school-related question here")

if prompt_school:
    text_school = []
    txt_school = re.sub('[^a-zA-Z\']', ' ', prompt_school)
    txt_school = txt_school.lower()
    txt_school = txt_school.split()
    txt_school = " ".join(txt_school)
    text_school.append(txt_school)

    # Ensure that maxlen in pad_sequences matches the value used during training
    x_test_school = tokenizer_school.texts_to_sequences(text_school)
    x_test_school = pad_sequences(x_test_school, padding='post', maxlen=8)  # Update maxlen as needed
    y_pred_school = loaded_model_school.predict(x_test_school)
    y_pred_school = y_pred_school.argmax()
    tag_school = lbl_enc_school.inverse_transform([y_pred_school])[0]
    responses_school = df_school[df_school['tag'] == tag_school]['responses'].values[0]

    bot_response_school = random.choice(responses_school) if responses_school else "I don't have information on that topic."

    with st.chat_message("user"):
        st.write(prompt_school)
    with st.chat_message("assistant"):
        st.write(bot_response_school)
    st.session_state.messages_school.append({"role": "user", "content": prompt_school})
    st.session_state.messages_school.append({"role": "assistant", "content": "BOT: " + bot_response_school})
