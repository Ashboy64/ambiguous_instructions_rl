from transformers import GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification
import torch
import random
import json
import torchmetrics
import torch.nn.functional as F
import matplotlib.pyplot as plt


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer, pad_id, vocab=None):
        with open(data_dir, "r") as f:
            self.samples = [json.loads(line) for line in f]
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        # self.vocab = []
        # for sample in self.samples:
        #     sample = self.tokenizer(sample["text"])
            # print(sample)
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.vocab is not None:
            sample = self.tokenizer(self.samples[idx]["text"])
        else:
            sample = self.tokenizer(self.samples[idx]["text"])["input_ids"]
        while len(sample) < 156:
            sample.append(self.pad_id)

        if self.vocab is not None:
            text = []
            for word in sample:
                text.append(self.vocab[word])
            text = torch.Tensor(text).long()
        else:
            text = torch.Tensor(sample).long()
        # print(sample)

        if self.samples[idx]["label"]:
            label = 1
        else:
            label = 0

        return text, label


def train(model, num_epochs, data_loader, valid_dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

    model.train()

    validate(model, valid_dataloader)

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
            output = model(text, labels=labels)
            loss, logits = output.loss, output.logits
            # print(torch.argmax(logits, dim=-1))
            # print(labels)
            # print(logits[0], labels[0])
            # loss = F.binary_cross_entropy_with_logits(torch.argmax(logits, dim=-1).float(), labels.float())
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # loss.requires_grad = True
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            acc = metric(logits, labels)
            print("Batch", i, ", Accuracy:", acc.item(), "Loss:", loss.item())
            if i == 1:
                # print("logits", logits.data)
                print("pred", torch.argmax(logits, dim=-1))
                print("target", labels.data)
            i += 1
        acc = metric.compute()
        print("******************\nEpoch", epoch, ", Accuracy:", acc.item(), "Loss:", epoch_loss / i, "\n*********************")
        with open("log.txt", "a") as f:
            f.write(str(acc.item()) + "," + str(epoch_loss) + "\n")
        validate(model, valid_dataloader)
        if epoch % 1== 0:
            torch.save(model.state_dict(), "finetuned_models/gpt2.pth")
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
            output = model(text, labels=labels)
            loss, logits = output.loss, output.logits
            epoch_loss += loss.item()
            # pred = F.softmax(logits, dim=-1)
            acc = metric(logits, labels)
            print("Validation: Batch", i, ", Accuracy:", acc.item(), "Loss:", loss.item())
            i += 1
        acc = metric.compute()
        print("******************\nOverall Validation Accuracy:", acc.item(), "Loss:", epoch_loss / i, "\n*********************")
        with open("log_val.txt", "a") as f:
            f.write(str(acc.item()) + "," + str(epoch_loss) + "\n")


def plot(filepath_train, filepath_val, var_name):
    plt.figure()
    for filepath in [filepath_train, filepath_val]:
        with open(filepath, "r") as f:
            data = f.readlines()
        accuracy = []
        loss = []
        for d in data:
            a, l = d.split(",")
            a = float(a)
            l = float(l)
            accuracy.append(a)
            loss.append(l)
        if "val" in filepath:
            label = "validation"
        else:
            label = "training"
        if var_name == "accuracy":
            plt.plot(range(len(accuracy)), accuracy, label=label)
            # plt.xticks(range(len(accuracy)))
        if var_name == "loss":
            plt.plot(range(len(loss)), loss, label=label)
            # plt.xticks(range(len(loss)))
    if var_name == "accuracy":
        plt.ylabel("Accuracy")
        plt.title("Accuracy Plot over Epochs")
    else:
        plt.ylabel("Loss")
        plt.title("Loss Plot over Epochs")
    plt.xlabel("Number of Epochs")
    plt.legend()
    plt.savefig(var_name + ".png")
    plt.close()


def test():
    batch_size = 16
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=2)
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model.load_state_dict(torch.load("finetuned_models/gpt2.pth"))
    test_dataset = TextDataset('data/test.jsonl', tokenizer, model.config.eos_token_id)
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    metric = torchmetrics.Accuracy()
    with torch.no_grad():
        i = 1
        epoch_loss = 0.
        for batch in data_loader:
            text, labels = batch
            # labels = labels.unsqueeze(1)
            model.zero_grad()
            output = model(text, labels=labels)
            loss, logits = output.loss, output.logits
            epoch_loss += loss.item()
            # pred = F.softmax(logits, dim=-1)
            acc = metric(logits, labels)
            print("Test: Batch", i, ", Accuracy:", acc.item(), "Loss:", loss.item())
            i += 1
        acc = metric.compute()
        print("******************\nOverall Test Accuracy:", acc.item(), "Loss:", epoch_loss / i, "\n*********************")
        with open("log_test.txt", "a") as f:
            f.write(str(acc.item()) + "," + str(epoch_loss) + "\n")


# reference: https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/gpt2_finetune_classification.ipynb#scrollTo=OlXROUWu5Osq
def main():
    random.seed(17)
    num_epochs = 5
    batch_size = 16

    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=2)
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    train_dataset = TextDataset('data/train.jsonl', tokenizer, model.config.eos_token_id)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset =  TextDataset('data/valid.jsonl', tokenizer, model.config.eos_token_id)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # print(model)
    model = train(model, num_epochs, train_dataloader, valid_dataloader)

    plot("log.txt", "log_val.txt", "accuracy")
    plot("log.txt", "log_val.txt", "loss")


if __name__ == "__main__":
    # main()
    test()
