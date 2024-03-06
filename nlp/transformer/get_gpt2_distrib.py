from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


def get_next_word_distribution(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    loss, logits = outputs[:2]

    predictions = logits[:, -1, :]
    probabilities = torch.softmax(predictions, dim=-1)
    probabilities = probabilities.cpu().numpy()

    idx_to_token = {idx: token for token, idx in tokenizer.get_vocab().items()}
    word_probabilities = {
        idx_to_token[idx]: prob for idx, prob in enumerate(probabilities[0])
    }

    return word_probabilities


prompt = "The sky is" 
word_probabilities = get_next_word_distribution(prompt)

top = sorted(word_probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
for word, prob in top:
    print(f"{word[1:]}: {prob:.4f}")
