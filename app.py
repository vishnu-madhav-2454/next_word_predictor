import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


faqs = """About the Program
What is the course fee for Data Science Mentorship Program (DSMP 2023)
... (shortened for brevity)"""


tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])


input_sequences = []
for sentence in faqs.split('\n'):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[:i+1])


max_len = max([len(x) for x in input_sequences])
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')


X = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1)


model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 100, input_length=max_len - 1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


st.title("Next Word Predictor")
user_input = st.text_input("Enter a word or phrase:")
if st.button("Predict Next Word"):
    token_text = tokenizer.texts_to_sequences([user_input])[0]
    padded_token_text = pad_sequences([token_text], maxlen=max_len-1, padding='pre')
    pos = np.argmax(model.predict(padded_token_text))
    predicted_word = ""
    for word, index in tokenizer.word_index.items():
        if index == pos:
            predicted_word = word
            break
    st.write(f"Predicted next word: {predicted_word}")
