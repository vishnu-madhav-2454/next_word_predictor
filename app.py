import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


faqs =  """Artificial intelligence has rapidly evolved over the
 past decade, revolutionizing numerous industries and aspects of daily life.
  One of the most significant advancements in AI is the development of natural language processing (NLP),
   which enables computers to understand, generate, and respond to human language in a meaningful way.
    From chatbots and virtual assistants to machine translation and sentiment analysis,
     NLP plays a crucial role in enhancing communication between humans and machines.
      The ability to predict the next word in a sentence is an essential component of many NLP applications
      , improving text generation, autocomplete features, and even speech recognition systems. 
      By analyzing vast amounts of text data, machine learning models can identify patterns, learn 
      grammatical structures, and develop contextual awareness. This predictive capability is
       particularly useful in mobile keyboards, where users rely on word suggestions to type messages more efficiently. 
       Similarly, it aids in search engines, where autocomplete suggestions help users find relevant information quickly.
        The underlying models behind next-word prediction are often trained on massive datasets, including books, news articles, and conversational data. 
        These models leverage techniques like deep learning, recurrent neural networks (RNNs), and transformers, such as the widely known GPT (Generative Pre-trained Transformer) architecture. 
        One of the main challenges in next-word prediction is maintaining coherence and context, especially in longer sentences where multiple possible words could logically follow. Ambiguity is another issue,
as the meaning of a sentence can change depending on the preceding words. To address this, advanced models use attention mechanisms, which help them focus on the most relevant parts of the input text while generating predictions.
Another crucial aspect is personalization, where AI systems adapt their predictions based on user behavior and writing style. For instance, a user who frequently types technical terms may receive suggestions that align with their specific domain of expertise. 
Ethical considerations also play a role in next-word prediction technology, as biased or inappropriate suggestions could lead to misinformation or offensive language. Researchers continuously work on improving fairness and reducing biases in AI models by carefully curating training data and implementing filtering mechanisms. 
In addition to mobile and web applications, next-word prediction has implications in creative writing, where authors use AI-assisted tools to brainstorm ideas and refine their drafts. Educational platforms also benefit from this technology by helping students improve their writing skills through intelligent grammar and vocabulary suggestions.
As AI continues to advance, the accuracy and sophistication of next-word prediction models will only improve, leading to even more seamless interactions between humans and machines. While challenges remain, the future of NLP and AI-driven text generation
looks promising, with ongoing innovations paving the way for smarter, more intuitive digital communication tools.

"""


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
