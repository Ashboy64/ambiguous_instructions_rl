from telnetlib import X3PAD
import torch
import torch.nn as nn
import torch
import json
from classify import TextDataset
import random
from torchtext.data.utils import get_tokenizer
from classify import plot
import torchmetrics


def create_vocab(data_dirs):
    vocab = {}
    all_samples = []
    for data_dir in data_dirs:
        with open(data_dir, "r") as f:
            samples = [json.loads(line) for line in f] 
            all_samples += samples
    tokenizer = get_tokenizer(None, language='en')
    idx = 1
    print(len(all_samples))
    for sample in all_samples:
        sample = tokenizer(sample["text"])
        for word in sample:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    vocab["<PAD>"] = 0
    return vocab


class LSTM_classifier(nn.Module):
    def __init__(self, vocab_len, embed_dim=256, input_dim=156, output_dim=1):
        super().__init__()

        self.input_dim = input_dim
        self.embedding = nn.Embedding(vocab_len, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=input_dim, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, labels):
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        x = self.lstm(x)
        # print(x[1][0].shape)
        x = self.dropout(x[1][0].squeeze_())
        # print(x.shape)
        x = self.linear(x)
        x = self.sigmoid(x).clone()
        loss = self.bceloss(x.squeeze_(), labels)

        return x, loss


def train(model, num_epochs, data_loader, valid_dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    validate(model, valid_dataloader)

    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.
        metric = torchmetrics.Accuracy()
        i = 1
        for batch in data_loader:
            # if i == 10:
            #     break
            text, labels = batch
            # labels = labels.unsqueeze(1)
            model.zero_grad()
            # print(type(text), type(labels), text.keys())
            output, loss = model(text, labels.float())
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            acc = metric(output, labels)
            print("Batch", i, ", Accuracy:", acc.item(), "Loss:", loss.item())
            if i == 1:
                # print("logits", logits.data)
                print("pred", (output > 0.5).type(torch.int8))
                print("target", labels.data)
            i += 1
        acc = metric.compute()
        print("******************\nEpoch", epoch, ", Accuracy:", acc.item(), "Loss:", epoch_loss / i, "\n*********************")
        with open("log.txt", "a") as f:
            f.write(str(acc.item()) + "," + str(epoch_loss) + "\n")
        validate(model, valid_dataloader)
        if epoch % 1== 0:
            torch.save(model.state_dict(), "finetuned_models/lstm.pth")
            plot("log.txt", "log_val.txt", "accuracy")
            plot("log.txt", "log_val.txt", "loss")

    return model


def validate(model, data_loader):
    model.eval()
    metric = torchmetrics.Accuracy()
    with torch.no_grad():
        i = 1
        epoch_loss = 0.
        for batch in data_loader:
            # if i == 10:
            #     break
            text, labels = batch
            # labels = labels.unsqueeze(1)
            model.zero_grad()
            output, loss = model(text, labels.float())
            epoch_loss += loss.item()
            # pred = F.softmax(logits, dim=-1)
            acc = metric(output, labels)
            print("Validation: Batch", i, ", Accuracy:", acc.item(), "Loss:", loss.item())
            i += 1
        acc = metric.compute()
        print("******************\nOverall Validation Accuracy:", acc.item(), "Loss:", epoch_loss, "\n*********************")
        with open("log_val.txt", "a") as f:
            f.write(str(acc.item()) + "," + str(epoch_loss) + "\n")


def test():
    batch_size = 16
    vocab = create_vocab(['data/train.jsonl', 'data/valid.jsonl'])
    model = LSTM_classifier(len(vocab))
    model.load_state_dict(torch.load("finetuned_models/lstm.pth"))
    test_dataset = TextDataset('data/test.jsonl', get_tokenizer("spacy", language='en'), "<PAD>", vocab=vocab)
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    metric = torchmetrics.Accuracy()
    with torch.no_grad():
        i = 1
        epoch_loss = 0.
        for batch in data_loader:
            # if i == 10:
            #     break
            text, labels = batch
            # labels = labels.unsqueeze(1)
            model.zero_grad()
            output, loss = model(text, labels.float())
            epoch_loss += loss.item()
            # pred = F.softmax(logits, dim=-1)
            acc = metric(output, labels)
            print("Test: Batch", i, ", Accuracy:", acc.item(), "Loss:", loss.item())
            i += 1
        acc = metric.compute()
        print("******************\nOverall Test Accuracy:", acc.item(), "Loss:", epoch_loss, "\n*********************")
        with open("log_test.txt", "a") as f:
            f.write(str(acc.item()) + "," + str(epoch_loss) / i + "\n")



def main():
    torch.autograd.set_detect_anomaly(True)
    # random.seed(17)
    num_epochs = 10
    batch_size = 16
    vocab = create_vocab(['data/train.jsonl', 'data/valid.jsonl'])
    train_dataset = TextDataset('data/train.jsonl', get_tokenizer("spacy", language='en'), "<PAD>", vocab=vocab)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset =  TextDataset('data/valid.jsonl', get_tokenizer("spacy", language='en'), "<PAD>", vocab=vocab)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # print(model)
    model = LSTM_classifier(len(vocab))
    model = train(model, num_epochs, train_dataloader, valid_dataloader)

    plot("log.txt", "log_val.txt", "accuracy")
    plot("log.txt", "log_val.txt", "loss")


if __name__ == "__main__":
    # main()
    test()