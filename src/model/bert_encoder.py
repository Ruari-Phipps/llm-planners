import torch
from transformers import BertTokenizer, BertModel, logging

logging.set_verbosity_error()


from scipy.spatial.distance import cosine 

model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_hidden_states=True,  # Whether the model returns all hidden-states.
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.eval()


def bert_process_sentence(sentence):
    marked_text = "[CLS] " + sentence + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokens_tensor, segments_tensors


def bert_encode(sentence, output="word"):
    indexed_tokens, segments_ids = bert_process_sentence(sentence)
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = model(indexed_tokens, segments_ids)
        hidden_states = outputs[2]

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1, 0, 2)

    if output == "word":
        # Get sum embedding for each word
        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []

        # For each token in the sentence...
        for token in token_embeddings:

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)

            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        return token_vecs_sum

    # Return sentence embedding
    return torch.mean(token_embeddings[:, -1], dim=0)


if __name__ == '__main__':
    a = bert_encode(sentence="place next to red cube", output="s")
    b = bert_encode(sentence="pick up the red cube", output="s")
    c = bert_encode(sentence="place on top of red cube", output="s")

    print(cosine(a, b))
    print(cosine(a, c))
    print(cosine(b, c))

