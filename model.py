import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import tqdm

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO
        self.c1 = nn.Conv2d()
        self.bn1 = nn.BatchNorm2d()
        self.c2 = nn.Conv2d()
        self.bn2 = nn.BatchNorm2d()
        self.output = nn.Linear()
    
    def forward(self, x):
        # go through layers
        x = self.c1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = f.relu(x)

        # return prediction
        x = torch.flatten(x, 1)
        pred = self.output(x)
        return pred

    def inference(self, x):
        return self.forward(x) # gets and returns prediction

    def loss(self, predictions, labels):
        return nn.CrossEntropyLoss(predictions, labels)


def train(model, optimizer, train_loader, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(train_loader)):
        # get prediction
        optimizer.zero_grad()
        output = model(inputs)

        # calculate losses
        loss = model.loss(output, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # print losses
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # get prediction
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)

            # calculate loss
            test_loss += model.loss(output, labels, reduction='mean').item()
            num_correct = (predicted == labels).sum().item()
            correct += num_correct

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset) * test_loader.dataset.sequence_length,
        100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)))
    return test_loss, test_accuracy

def train_acc(model, train_dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total