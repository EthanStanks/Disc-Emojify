import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re


EMBEDDING_DIM = 2000
NUM_EPOCHS = 15
BATCH_SIZE = 25

SKIP_WORDS = set(stopwords.words('english'))
SKIP_WORDS = SKIP_WORDS.union(['went','got','want','get',',','?','!','@','#','$','%','^','&','*','(',')','/','.',
                               '<','>',';',':','\"','[',']','{','}','\\','|','_','=','+', 'made'])

def DataPrep():
    data_folder = 'Disc-Emojify/data/'
    output_folder = 'Disc-Emojify/output/'
    data_file = 'emojis.csv'
    data_path = os.path.join(data_folder, data_file)

    df = pd.read_csv(data_path)

    unique_emojis = df['emoji'].unique()
    emoji_to_int = {emoji: i for i, emoji in enumerate(unique_emojis)}

    tokenized_descriptions = []
    emoji_labels = []
    for _, row in df.iterrows():
        emoji = row['emoji']
        descriptions = row['description'].split(',')

        for description in descriptions:
            words = description.strip().lower()
            words = re.findall(r'\b\w+\b|[.,;!?]', words)
            tokenized_description = [word for word in words if word not in SKIP_WORDS]
            tokenized_description = ' '.join(tokenized_description)
            tokenized_descriptions.append(tokenized_description)
            emoji_labels.append(emoji_to_int[emoji])
    
    # tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_descriptions)
    sequences = tokenizer.texts_to_sequences(tokenized_descriptions)
    
    # padding
    sequence_length = len(max(sequences, key=len))
    padded_sequences = pad_sequences(sequences, maxlen=sequence_length)

    return padded_sequences, emoji_labels, unique_emojis, tokenizer, sequence_length

def Train(padded_sequences, emoji_labels, unique_emojis, tokenizer, sequence_length):
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, emoji_labels, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, num_classes=len(unique_emojis))
    y_test = to_categorical(y_test, num_classes=len(unique_emojis))

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=EMBEDDING_DIM, input_length=sequence_length))
    model.add(LSTM(128))
    model.add(Dense(len(unique_emojis), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

    return model

def Keywords(message):
    words = re.findall(r'\b\w+\b|[.,;!?]', message)
    words = [word for word in words if word not in SKIP_WORDS]
    return words

def Predict(message, tokenizer, model, unique_emojis, sequence_length):
        message = message.strip().lower()
        keywords = Keywords(message)
        keyword_emojis = dict()
        for word in keywords:
            text_input = tokenizer.texts_to_sequences([word])
            text_input = pad_sequences(text_input, maxlen=sequence_length)
            predicted_label = model.predict(text_input)
            predicted_emoji_index = np.argmax(predicted_label)
            predicted_emoji = unique_emojis[predicted_emoji_index]
            keyword_emojis[word] = predicted_emoji
        return keyword_emojis