import numpy as np
import tiktoken

from n_gram.n_gram import train_trigram

enc = tiktoken.encoding_for_model("davinci")
tokenizer = enc.encode
trigram = train_trigram()

prompt = "The most beautiful proof in math is"
tokens = tokenizer(prompt)
next_word = ""

next_probs = []

for i in range(40):
    values, probabilities = [], []
    for word in range(enc.n_vocab):
        prob = trigram.n_gram_probability([tokens[-2], tokens[-1], word])
        values.append(word)
        probabilities.append(prob)

    top = sorted(zip(values, probabilities), key=lambda x: x[1], reverse=True)[:20]
    values, probabilities = zip(*top)

    # print("Top Predictions:")
    # for i in range(10):
    #     print(f"{enc.decode([values[i]]):<10}: {probabilities[i]:.4f}")
    # print([enc.decode([i]) for i in values[:10]], probabilities[:10])

    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    next_word = np.random.choice(values, p=probabilities)
    tokens.append(next_word)

    next_probs.append(top[:10])

# print(tokens)
# for t in tokens:
#     print(f"'{enc.decode([t])}'")
np.save("next_probs.npy", next_probs)
print(enc.decode(tokens))
