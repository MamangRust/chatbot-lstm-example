from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import uvicorn
import re
import random
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index_2.html", {"request": request})

@app.post("/chat")
def process_chat(request: Request, prompt_school: str = Form(...)):
    text_school = []
    txt_school = re.sub('[^a-zA-Z\']', ' ', prompt_school)
    txt_school = txt_school.lower()
    txt_school = txt_school.split()
    txt_school = " ".join(txt_school)
    text_school.append(txt_school)

    x_test_school = tokenizer_school.texts_to_sequences(text_school)
    x_test_school = pad_sequences(x_test_school, padding='post', maxlen=8)
    y_pred_school = loaded_model_school.predict(x_test_school)
    y_pred_school = y_pred_school.argmax()
    tag_school = lbl_enc_school.inverse_transform([y_pred_school])[0]
    responses_school = df_school[df_school['tag'] == tag_school]['responses'].values[0]

    bot_response_school = random.choice(responses_school) if responses_school else "I don't have information on that topic."

    return JSONResponse(content={"bot_response_school": bot_response_school})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
