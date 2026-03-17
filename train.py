import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt')

ps = PorterStemmer()

# Load data
with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

# Prepare dataset
for intent in data['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        docs_x.append(w)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Stemming
words = [ps.stem(w.lower()) for w in words if w.isalnum()]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0] * len(labels)

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [ps.stem(w.lower()) for w in doc]

    for w in words:
        bag.append(1 if w in wrds else 0)

    training.append(bag)

    output_row = out_empty.copy()
    output_row[labels.index(docs_y[x])] = 1
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Model
model = Sequential()
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(labels), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
model.fit(training, output, epochs=300, batch_size=8, verbose=1)

# Save files
model.save("chat_model.h5")

np.save("words.npy", words)
np.save("labels.npy", labels)

print("Training Complete!")
