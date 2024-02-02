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
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig
from transformers.generation.streamers import TextStreamer

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

# Loading pre-trained school chatbot model
model_path_school = 'my_model.keras'  # Update with the correct path
loaded_model_school = load_model(model_path_school)

# Loading LLM (Neural Chat) model
model_vendor, model_name = 'Intel', 'neural-chat-7b-v3'
model_precision = ['FP16', 'INT8', 'INT4'][2]

tokenizer_llm = AutoTokenizer.from_pretrained(f'{model_vendor}/{model_name}')
ov_model_llm = OVModelForCausalLM.from_pretrained(
    model_id=f'{model_name}/{model_precision}',
    device='CPU',
    ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""},
    config=AutoConfig.from_pretrained(f'{model_name}/{model_precision}')
)

# Streamlit application
st.title("Multi-ChatBot Interface")
st.write("Welcome! Ask me anything about the school or any general question.")

# Display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and chat processing
prompt = st.text_input("Type your question here")

if prompt:
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', prompt)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)

    # School ChatBot
    x_test_school = tokenizer_school.texts_to_sequences(text)
    x_test_school = pad_sequences(x_test_school, padding='post', maxlen=8)
    y_pred_school = loaded_model_school.predict(x_test_school)
    y_pred_school = y_pred_school.argmax()
    tag_school = lbl_enc_school.inverse_transform([y_pred_school])[0]
    responses_school = df_school[df_school['tag'] == tag_school]['responses'].values[0]

    bot_response_school = random.choice(responses_school) if responses_school else "I don't have information on that topic."

    # LLM ChatBot
    prompt_llm = f"""\
    ### System:
    You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    ### User:
    {prompt}
    ### Assistant (School ChatBot):
    {bot_response_school}
    ### Assistant (LLM ChatBot):
    """

    input_tokens_llm = tokenizer_llm(prompt_llm, return_tensors='pt', add_special_tokens=False)
    response_llm = ov_model_llm.generate(**input_tokens_llm, max_new_tokens=300, temperature=0.2, do_sample=True, top_k=5, top_p=0.8, repetition_penalty=1.2, num_return_sequences=1)
    response_text_llm = tokenizer_llm.decode(response_llm[0], skip_special_tokens=True)
    bot_response_llm = response_text_llm.split('### Assistant (LLM ChatBot):\n')[-1]

 
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st.write(f"BOT (School ChatBot): {bot_response_school}")
    with st.chat_message("assistant"):
        st.write(f"BOT (LLM ChatBot): {bot_response_llm}")

    st.session_state.messages.append({"role": "user", "content": prompt})


    st.session_state.messages.append({"role": "assistant", "content": f"BOT (LLM ChatBot): {bot_response_llm}"})
