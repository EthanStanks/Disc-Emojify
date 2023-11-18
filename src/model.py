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
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.metrics import mean_absolute_error
from keras.layers import Dense
from keras.optimizers import Adam
from wordcloud import WordCloud

IS_GRAPHING = False
IS_PRINTING = False
EMBEDDING_DIM = 2000
NUM_EPOCHS = 20
BATCH_SIZE = 25

SKIP_WORDS = set(stopwords.words('english'))
SKIP_WORDS = SKIP_WORDS.union(['went','got','want','get',',','?','!','@','#','$','%','^','&','*','(',')','/','.',
                               '<','>',';',':','\"','[',']','{','}','\\','|','_','=','+', 'made','goes','really'])

def DataPrep():
    data_folder = 'data/'
    data_file = 'emojis.csv'
    data_path = os.path.join(data_folder, data_file)
    df = pd.read_csv(data_path)

    unique_emojis = df['emoji'].unique()
    emoji_to_int = {emoji: i for i, emoji in enumerate(unique_emojis)}

    if(IS_GRAPHING):
        df['description_length'] = df['description'].apply(lambda x: len(x.split(',')))
        sns.histplot(df['description_length'])
        plt.title('Distribution of Description Lengths')
        plt.xlabel('Description Length')
        plt.ylabel('Count of Emojis with Similar Lengths')
        plt.savefig(os.path.join('output/','DescriptionLength.png'))
        plt.close()

        all_descriptions = ' '.join(df['description'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Emoji Descriptions')
        plt.savefig(os.path.join('output/','WordCloud.png'))
        plt.close()

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
        
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_descriptions)
    sequences = tokenizer.texts_to_sequences(tokenized_descriptions)
    sequence_length = len(max(sequences, key=len))
    padded_sequences = pad_sequences(sequences, maxlen=sequence_length)

    return padded_sequences, emoji_labels, unique_emojis, tokenizer, sequence_length

def Train(padded_sequences, emoji_labels, unique_emojis, tokenizer, sequence_length):
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, emoji_labels, random_state=42)
    y_train = to_categorical(y_train, num_classes=len(unique_emojis))
    y_test = to_categorical(y_test, num_classes=len(unique_emojis))

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=EMBEDDING_DIM, input_length=sequence_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(56))
    model.add(Dense(len(unique_emojis), activation='softmax'))

    optimizer=Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    fit = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    if(IS_PRINTING):
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
        test_mae = mean_absolute_error(y_test, model.predict(X_test))
        train_mae = mean_absolute_error(y_train, model.predict(X_train))
        print("Test MAE:", test_mae)
        print("Train MAE:", train_mae)

    return model,fit

def Keywords(message):
    words = re.findall(r'\b\w+\b|[.,;!?]', message)
    words = [word for word in words if word not in SKIP_WORDS]
    return words

def Predict(message, tokenizer, model, unique_emojis, sequence_length):
        message = message.strip().lower()
        keywords = Keywords(message)
        keyword_emojis = dict()
        isPredicted = False
        for word in keywords:
            text_input = tokenizer.texts_to_sequences([word])
            text_input = pad_sequences(text_input, maxlen=sequence_length)
            if word in tokenizer.word_index:
                predicted_label = model.predict(text_input)
                predicted_emoji_index = np.argmax(predicted_label)
                predicted_emoji = unique_emojis[predicted_emoji_index]
                keyword_emojis[word] = predicted_emoji
                isPredicted = True

        return keyword_emojis, isPredicted

if __name__ == '__main__':
    padded_sequences, emoji_labels, unique_emojis, tokenizer, sequence_length = DataPrep()
    emojify,fit = Train(padded_sequences, emoji_labels, unique_emojis, tokenizer, sequence_length)

    if(IS_GRAPHING):
        output_folder = 'output/'

        plt.figure(figsize=(8,6))
        plt.title('Training and Validation Accuracy')
        plt.plot(fit.history['acc'], label='Training Accuracy')
        plt.plot(fit.history['val_acc'], label='Validation Accuracy')
        plt.legend(['accuracy', 'val_accuracy'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        accuracy_path = os.path.join(output_folder,'AccuracyScore.png')
        plt.savefig(accuracy_path)
        plt.close()

        plt.figure(figsize=(8,6))
        plt.title('Training and Validation Loss')
        plt.plot(fit.history['loss'], label='Training Loss')
        plt.plot(fit.history['val_loss'], label='Validation Loss')
        plt.legend(['loss', 'val_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        loss_path = os.path.join(output_folder,'LossScore.png')
        plt.savefig(loss_path)
        plt.close()

