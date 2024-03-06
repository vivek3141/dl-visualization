# Source: https://machinelearningmastery.com/text-generation-with-lstm-in-pytorch/

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# load ascii text and covert to lowercase
filename = "gutenberg.txt"
raw_text = open(filename, "r", encoding="utf-8").read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)


class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)

    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x


model = CharModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generation using the trained model
best_model, char_to_int = torch.load("single-char.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
model.load_state_dict(best_model)

# randomly generate a prompt
filename = "gutenberg.txt"
seq_length = 100
raw_text = open(filename, "r", encoding="utf-8").read()
raw_text = raw_text.lower()
start = np.random.randint(0, len(raw_text) - seq_length)
prompt = raw_text[start : start + seq_length]
prompt = "alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought alice 'without pictures or conversation?'"
pattern = [char_to_int[c] for c in prompt]


model.eval()
print('Prompt: "%s"' % prompt)
with torch.no_grad():
    for i in range(1000):
        # format input array of int into PyTorch tensor
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(x.to(device))
        # convert logits into one character
        dist = F.softmax(prediction, dim=1).cpu().numpy()
        # Do top-k sampling with k=1
        top_k = 1
        top_k_indices = np.argsort(dist[0])[-top_k:]
        top_k_weights = dist[0][top_k_indices] / np.sum(dist[0][top_k_indices])
        index = np.random.choice(top_k_indices, p=top_k_weights)
        assert index == int(prediction.argmax())
        result = int_to_char[index]
        print(result, end="")
        # append the new character into the prompt for the next iteration
        pattern.append(index)
        pattern = pattern[1:]
print()
print("Done.")
