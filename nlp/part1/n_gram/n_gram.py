import tqdm
from collections import defaultdict, Counter
import tiktoken
from nltk.corpus import brown


class NGramModel:
    """
    An n-gram model, where alpha is the laplace smoothing parameter.
    """

    def __init__(self, train_text, n=2, alpha=3e-3, vocab_size=None):
        self.n = n
        if vocab_size is None:
            # Assume GPT tokenizer
            self.vocab_size = 50257

        self.smoothing = alpha
        self.smoothing_f = alpha * self.vocab_size

        self.c = defaultdict(lambda: [0, Counter()])
        for i in tqdm.tqdm(range(len(train_text) - n)):
            n_gram = tuple(train_text[i : i + n])
            self.c[n_gram[:-1]][1][n_gram[-1]] += 1
            self.c[n_gram[:-1]][0] += 1
        self.n_size = len(self.c)

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == self.n
        it = self.c[tuple(n_gram[:-1])]
        prob = (it[1][n_gram[-1]] + self.smoothing) / (it[0] + self.smoothing_f)
        return prob


class DiscountBackoffModel(NGramModel):
    """
    An n-gram model with discounting and backoff. Delta is the discounting parameter.
    """

    def __init__(self, train_text, lower_order_model, n=2, delta=0.9):
        super().__init__(train_text, n=n)
        self.lower_order_model = lower_order_model
        self.discount = delta

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == self.n
        it = self.c[tuple(n_gram[:-1])]

        if it[0] == 0:
            return self.lower_order_model.n_gram_probability(n_gram[1:])

        prob = (
            self.discount
            * (len(it[1]) / it[0])
            * self.lower_order_model.n_gram_probability(n_gram[1:])
        )
        if it[1][n_gram[-1]] != 0:
            prob += max(it[1][n_gram[-1]] - self.discount, 0) / it[0]

        return prob


class KneserNeyBaseModel(NGramModel):
    """
    A Kneser-Ney base model, where n=1.
    """

    def __init__(self, train_text, vocab_size=None):
        super().__init__(train_text, n=1, vocab_size=vocab_size)

        base_cnt = defaultdict(set)
        for i in range(1, len(train_text)):
            base_cnt[train_text[i]].add(train_text[i - 1])

        cnt = 0
        for word in base_cnt:
            cnt += len(base_cnt[word])

        self.prob = defaultdict(float)
        for word in base_cnt:
            self.prob[word] = len(base_cnt[word]) / cnt

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == 1
        ret_prob = self.prob[n_gram[0]]

        if ret_prob == 0:
            return 1 / self.vocab_size
        else:
            return ret_prob


class TrigramBackoff:
    """
    A trigram model with discounting and backoff. Uses a Kneser-Ney base model.
    """

    def __init__(self, train_text, delta=0.9):
        self.base = KneserNeyBaseModel(train_text)
        self.bigram = DiscountBackoffModel(train_text, self.base, n=2, delta=delta)
        self.trigram = DiscountBackoffModel(train_text, self.bigram, n=3, delta=delta)

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == 3
        return self.trigram.n_gram_probability(n_gram)


def train_trigram(verbose=True, return_tokenizer=False):
    """
    Trains and returns a trigram model on the brown corpus
    """

    enc = tiktoken.encoding_for_model("davinci")
    tokenizer = enc.encode
    vocab_size = enc.n_vocab

    # We use the brown corpus to train the n-gram model
    sentences = brown.sents()

    if verbose:
        print("Tokenizing corpus...")
    tokenized_corpus = []
    for sentence in tqdm.tqdm(sentences):
        tokens = tokenizer(" ".join(sentence))
        tokenized_corpus += tokens

    if verbose:
        print("\nTraining n-gram model...")

    if return_tokenizer:
        return TrigramBackoff(tokenized_corpus), tokenizer
    else:
        return TrigramBackoff(tokenized_corpus)
