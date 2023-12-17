import torch
from torch import nn

from .resnet_18_film import ResNet18BlockFiLM

from transformers import AutoTokenizer, AutoModel, logging

logging.set_verbosity_error()


from scipy.spatial.distance import cosine 


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

# def bert_process_sentences(tokenizer, sentences):
#     indexed_tokens = []
#     segments_ids = []
#     for sentence in sentences:
#         marked_text = "[CLS] " + sentence + " [SEP]"

#         # Split the sentence into tokens.
#         tokenized_text = tokenizer.tokenize(marked_text)

#         # Map the token strings to their vocabulary indeces.
#         indexed_tokens.append(tokenizer.convert_tokens_to_ids(tokenized_text))

#         segments_ids.append([1] * len(tokenized_text))

#     print(indexed_tokens)
#     tokens_tensors = torch.tensor(indexed_tokens)
#     segments_tensors = torch.tensor(segments_ids)

#     return tokens_tensors, segments_tensors

class ResNet18FiLMBert(nn.Module):
    def __init__(
        self, c_in=3, c_out=3, activation="linear",  bert="bert-base-uncased", bert_size = 768, bert_init=None
    ) -> None:
        super().__init__()
        self.c_out = c_out

        self.bert_model = AutoModel.from_pretrained(
            bert,
            output_hidden_states=True,  # Whether the model returns all hidden-states.
        )
        self.tokenizer = AutoTokenizer.from_pretrained(bert)

        if bert_init is not None:
            self.bert_model.to(torch.device("cpu"))
            self.bert_model.load_state_dict(torch.load(bert_init, map_location=torch.device('cpu')))
            self.bert_model.to(device)
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.c1 = nn.Conv2d(c_in, 64, 7)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [
                ResNet18BlockFiLM(c_in=64, c_out=64),
                ResNet18BlockFiLM(c_in=64, c_out=64),
                ResNet18BlockFiLM(c_in=64, c_out=128, stride=2, conv3=True),
                ResNet18BlockFiLM(c_in=128, c_out=128),
                ResNet18BlockFiLM(c_in=128, c_out=256, stride=2, conv3=True),
                ResNet18BlockFiLM(c_in=256, c_out=256),
                ResNet18BlockFiLM(c_in=256, c_out=512, stride=2, conv3=True),
                ResNet18BlockFiLM(c_in=512, c_out=512),
            ]
        )
        self.avgpool = nn.AvgPool2d(5)
        self.conv_out_1 = nn.Conv2d(512, 256, 1)
        self.conv_out_2 = nn.Conv2d(256, c_out, 1)

        self.channels = [
            res_block.get_channels() for res_block in self.res_blocks
        ]

        film_count = 2 * sum(self.channels)
        self.film_generator = nn.Linear(bert_size, bert_size)
        self.film_generator2 = nn.Linear(bert_size, bert_size)
        self.film_generator3 = nn.Linear(bert_size, bert_size)
        self.film_generator4 = nn.Linear(bert_size, film_count)

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax()
        else:
            self.activation = lambda x: x

    def forward(self, X, input):
        tokenised = self.tokenizer.batch_encode_plus(
            input,
            max_length = 25,
            padding='max_length',
            truncation=True
        )
        film = self.bert_model(torch.tensor(tokenised['input_ids']).to(device), attention_mask = torch.tensor(tokenised['attention_mask']).to(device), return_dict=False)[1]
        film = self.relu(self.film_generator(film))
        film = self.relu(self.film_generator2(film))
        film = self.relu(self.film_generator3(film))
        film = self.film_generator4(film)
        X = self.c1(X)
        X = self.bn1(X)
        X = self.maxpool(X)

        tot = 0
        for i, res_block in enumerate(self.res_blocks):
            channel = self.channels[i]
            g_b = film[:, tot : tot + 2 * channel]
            gamma, beta = g_b[:, :channel], g_b[:, channel:]

            tot += channel

            X = res_block(X, gamma, beta)
            X = self.relu(X)

        X = self.avgpool(X)
        X = self.conv_out_1(X)
        X = self.conv_out_2(X)
        X = self.activation(X)

        return X.view(-1, self.c_out)

if __name__ == "__main__":
    x = torch.ones((4, 1, 128, 128))
    sentences = ["hello", "Whats up my fellow humans", "my name is ruari phipps", "yoda"]

    model = ResNet18FiLMBert(c_in=1, c_out=1, bert="prajjwal1/bert-small", bert_size = 512)
    print(model.forward(x, sentences))
