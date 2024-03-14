import json
from nltk_utils import tokenize, lemmatize, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)


class ChatDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, index):
        try:
            x = self.x_data[index]
            y = self.y_data[index]
            return x, y
        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {e}")
            raise e

    def __len__(self):
        return self.n_samples


def main():
    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '!', '.', ',']
    all_words = [lemmatize(w) for w in all_words if w not in ignore_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    x_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        x_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)  # CrossEntropyLoss

    x_train = np.array(x_train)
    y_train = np.array(y_train)


    #Hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(x_train[0])
    learning_rate = 0.001
    num_epochs = 1000

    dataset = ChatDataset(x_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training arc
    i = 1
    for epochs in range(num_epochs):
        print(f'{i} iteration')
        i += 1
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device).long()  # Convert labels to Long type

            # forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            # backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epochs+1) % 100 == 0:
            print(f'epoch {epochs+1}/{num_epochs}, loss={loss.item():.4f}')

    print(f'final loss, loss={loss.item():.4f}')


if __name__ == '__main__':
    main()
