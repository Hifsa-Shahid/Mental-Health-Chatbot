# Mental Health Chatbot

A supportive mental health chatbot built in **Python**. It uses a small neural network (TensorFlow/Keras) to classify what the user is talking about (their *intent*) and replies with a matching supportive message through a desktop chat window built with **CustomTkinter**.

> **Disclaimer:** This chatbot is **not a substitute for professional therapy or medical advice**. If you are in crisis, please contact your local emergency services or a mental health helpline. A global directory is available at [findahelpline.com](https://findahelpline.com).

## Features

- **Intent classification:** a neural network trained on 85 intents and 6,282 example patterns (greetings, sadness, stress, anxiety, sleep, motivation, confidence, panic attacks, trauma, mental-health facts, and more).
- **Supportive responses:** each intent has a set of written responses (about 6,000 in total), and the bot picks one at random so replies do not feel repetitive.
- **Confidence threshold:** if the model is less than 80% confident, the bot asks the user to explain more instead of guessing.
- **Desktop GUI:** a dark-themed chat window built with CustomTkinter. Press **Enter** or click **Send**.
- **Runs locally:** after the one-time setup, no internet connection or API is needed.

## How It Works

1. **Preprocessing:** each pattern in `intents.json` is tokenized with NLTK and reduced to its stem with the Porter stemmer.
2. **Bag-of-words:** every sentence becomes a vector over a 1,497-word stemmed vocabulary (`words.npy`).
3. **Model:** a feed-forward network with a 128-unit dense layer, dropout (0.5), a 64-unit dense layer, dropout (0.5), and a softmax output over the 85 intents. It is trained with Adam and categorical cross-entropy for 300 epochs (batch size 8).
4. **Prediction:** the GUI converts the user's message to a bag-of-words vector, predicts the intent, and returns a random response from that intent (or a fallback message if confidence is 0.80 or lower).

> Each message is classified on its own. The bot does not remember earlier messages in the conversation.

## Project Structure

```
Mental-Health-Chatbot/
├── intents.json      # Intents, example patterns, and responses
├── train.py          # Builds the dataset, trains the model, saves the files below
├── chat_model.h5     # Trained Keras model
├── words.npy         # Stemmed vocabulary
├── labels.npy        # Intent labels
└── chatbot_gui.py    # CustomTkinter chat application
```

## Technologies Used

- **Language:** Python 3.x
- **Machine learning:** TensorFlow / Keras, NumPy
- **NLP:** NLTK (tokenization, Porter stemmer)
- **GUI:** CustomTkinter

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Hifsa-Shahid/Mental-Health-Chatbot.git
   cd Mental-Health-Chatbot
   ```

2. Install the dependencies:

   ```bash
   pip install tensorflow nltk numpy customtkinter
   ```

3. Download the NLTK tokenizer data (one time only):

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

## Usage

**Run the chatbot** (the trained model is already included):

```bash
python chatbot_gui.py
```

Type a message in the input box and press **Enter** or click **Send**.

**Retrain the model** (optional, after editing `intents.json`):

```bash
python train.py
```

This regenerates `chat_model.h5`, `words.npy`, and `labels.npy`.

## Example Conversations

| User says | Intent detected | Bot behavior |
| --- | --- | --- |
| "Hello" | greeting | Replies with a friendly greeting |
| "I'm feeling sad" | sad | Replies with a supportive message |
| "I'm so stressed" | stressed | Replies with a stress-related supportive message |
| Unclear or unrelated text | none (confidence ≤ 0.80) | "I'm not sure I understand. Can you explain more?" |

Responses are chosen randomly from each intent's list, so exact wording varies.

## Customizing

To add topics or responses, edit `intents.json`: add a new intent with a `tag`, example `patterns`, and `responses`, then run `python train.py` to retrain.

## Limitations

- The bot only recognizes topics that exist in `intents.json` and matches by vocabulary, so unusual phrasing or misspellings may be misunderstood.
- It has no memory of previous messages and does not store conversations.
- Model accuracy on unseen messages has not been formally measured.

## Contributing

Contributions are welcome. You can add new intents and responses, improve the model or its evaluation, or enhance the GUI. Please open an issue or submit a pull request.

## Author

**Hifsa Shahid**
