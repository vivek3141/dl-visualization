# Source: https://github.com/perkdrew/text-generation

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
import keras.utils as ku
import numpy as np


glove_path = "glove.twitter.27B/glove.twitter.27B.200d.txt"
tokenizer = Tokenizer()


def text_preprocess(text):
    # lowercase and split
    corpus = text.lower().split("\n")
    # tokenize
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    # create input sequences from token arrays
    input_seq = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_seq = token_list[: i + 1]
            input_seq.append(n_gram_seq)
    # pad sequences
    max_seq_len = max([len(x) for x in input_seq])
    input_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_len, padding="pre"))
    # create predictors and label
    predictors, label = input_seq[:, :-1], input_seq[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_seq_len, total_words


text = open("zagier.txt", encoding="latin1").read()
predictors, label, max_seq_len, total_words = text_preprocess(text)


def lstm_stack(predictors, label, max_seq_len, total_words):
    model = Sequential()
    model.add(
        Embedding(
            total_words, 200, weights=[embedding_matrix], input_length=max_seq_len - 1
        )
    )
    model.add(
        Bidirectional(
            LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
        )
    )
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    earlystop = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=5, verbose=0, mode="auto"
    )
    model.fit(predictors, label, epochs=25, verbose=1, callbacks=[earlystop])
    return model


# GloVe embeddings
embeddings_index = dict()
with open(glove_path, encoding="utf8") as glove:
    for line in glove:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    glove.close()

embedding_matrix = np.zeros((total_words, 200))
for word, index in tokenizer.word_index.items():
    if index > total_words - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


model = lstm_stack(predictors, label, max_seq_len, total_words)
model.save("zagier.h5")
