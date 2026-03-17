import customtkinter as ctk
import json
import random
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer()

# -------------------------
# Load ML Model + Data
# -------------------------
model = tf.keras.models.load_model("chat_model.h5")
words = np.load("words.npy", allow_pickle=True)
labels = np.load("labels.npy", allow_pickle=True)

with open("intents.json") as file:
    intents = json.load(file)


# -------------------------
# NLP Functions
# -------------------------
def bag_of_words(s):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [ps.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat_response(msg):
    bow = bag_of_words(msg)
    result = model.predict(np.array([bow]))[0]
    result_index = np.argmax(result)
    tag = labels[result_index]

    if result[result_index] > 0.80:
        for tg in intents["intents"]:
            if tg["tag"] == tag:
                return random.choice(tg["responses"])
    return "I'm not sure I understand. Can you explain more?"


# -------------------------
# GUI Setup
# -------------------------
ctk.set_appearance_mode("dark")      # "light" / "dark"
ctk.set_default_color_theme("blue")  # "green" / "dark-blue"

app = ctk.CTk()
app.title("Mental Health Chatbot")
app.state("zoomed")   # Full screen
app.resizable(False, False)


# -------------------------
# Chat Display (Read-Only, Word Wrap)
# -------------------------
chat_frame = ctk.CTkTextbox(
    app,
    width=480,
    height=520,
    font=("Times New Roman", 14),
    wrap="word"          # Wrap at word boundaries
)
chat_frame.pack(padx=20, pady=20)
chat_frame.configure(state="disabled")  # Make it read-only


# -------------------------
# Send Message Function
# -------------------------
def send_message(event=None):
    user_msg = entry.get()
    if user_msg.strip() == "":
        return

    # Enable chat_frame temporarily to insert messages
    chat_frame.configure(state="normal")

    # Insert user message
    chat_frame.insert("end", f"You: {user_msg}\n")
    entry.delete(0, "end")  # Clear entry
    chat_frame.see("end")   # Auto-scroll

    # Get bot reply
    bot_reply = chat_response(user_msg)
    chat_frame.insert("end", f"ChatBot: {bot_reply}\n\n")

    # Make chat_frame read-only again
    chat_frame.configure(state="disabled")
    chat_frame.see("end")


# -------------------------
# Entry Box
# -------------------------
entry = ctk.CTkEntry(
    app,
    width=350,
    placeholder_text="Type your message...",
    font=("Times New Roman", 14)
)
entry.pack(side="left", padx=15, pady=10)

# Bind Enter key to send message
entry.bind("<Return>", send_message)


# -------------------------
# Send Button
# -------------------------
send_btn = ctk.CTkButton(
    app,
    text="Send",
    width=120,
    height=40,
    font=("Times New Roman", 14),
    command=send_message
)
send_btn.pack(side="right", padx=10, pady=10)


# -------------------------
# Start App
# -------------------------
app.mainloop()
