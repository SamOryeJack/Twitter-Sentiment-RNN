import re
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr

MAX_LEN = 40

model = tf.keras.models.load_model("gru_sentiment.keras", compile=False)
with open("tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

def clean_tweet(t):
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'http\S+', '', t)
    t = re.sub(r'[^a-zA-Z\s]', '', t)
    return t.lower().strip()

def predict(text):
    cleaned = clean_tweet(text or "")
    if not cleaned:
        return "Enter a tweet with at least one word."
    seq = tok.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    p = float(model.predict(pad, verbose=0)[0][0])
    label = "Positive" if p > 0.5 else "Negative"
    conf = p if p > 0.5 else 1 - p
    return f"{label} ({conf:.1%} confidence)"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Type a tweet..."),
    outputs="text",
    title="Twitter Sentiment (GRU)",
    description="A GRU trained on 1.6 million tweets (Sentiment140). Type a sentence and it predicts positive or negative.",
    examples=["I love this!", "worst day ever", "the movie was ok"],
)

demo.launch()
