# Source: https://github.com/perkdrew/text-generation

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import keras.utils as ku
import numpy as np
import tqdm

tokenizer = Tokenizer()


def softmax(x, axis=1):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


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


def generate_text(seed_text, next_words, max_seq_len):
    for _ in tqdm.tqdm(range(next_words)):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding="pre")
        output = model.predict(token_list, verbose=0)
        dist = softmax(output, axis=1)[0]
        indices = np.argsort(dist)[-20:]
        probs = dist[indices] / np.sum(dist[indices])
        predicted = np.random.choice(indices, p=probs)
        # predicted = np.argmax(output, axis=1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


# Load model zagier.h5
model = load_model("zagier.h5")
print(generate_text("The most beautiful proof in math is", 50, max_seq_len))
