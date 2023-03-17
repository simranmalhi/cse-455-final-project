import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        RGB = 3 # rgb channel count
        LETTER_OUTPUT = 29 # output = 26 letters + 3 other things
        IMAGE_SIZE = 200 # pixel width of image
        filter_size = 5

        # define layers
        self.c1 = nn.Conv2d(RGB, 6, filter_size)
        self.c2 = nn.Conv2d(6, 12, filter_size)
        self.c3 = nn.Conv2d(12, 24, filter_size)
        self.c4 = nn.Conv2d(24, 48, filter_size)
        self.c5 = nn.Conv2d(48, 64, filter_size)

        # in_features is hardcoded and should be updated if layer output changes
        self.output = nn.Linear(2073600, LETTER_OUTPUT)

    def forward(self, x):
        # go through layers
        x = self.c1(x)
        x = f.relu(x)

        x = self.c2(x)
        x = f.relu(x)

        x = self.c3(x)
        x = f.relu(x)

        x = self.c4(x)
        x = f.relu(x)

        x = self.c5(x)
        x = f.relu(x)

        # return prediction
        x = torch.flatten(x, 1)
        pred = self.output(x)
        return pred

    def inference(self, x):
        return self.forward(x) # gets and returns prediction

    def loss(self, predictions, labels):
        return nn.CrossEntropyLoss()(predictions, labels)


# returns mean loss for single epoch
def train(model, optimizer, train_loader, epoch, log_interval, device):
    model.train()
    losses = []
    for batch_idx, batch in enumerate(train_loader, 0):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        # zero parameter gradients
        optimizer.zero_grad()

        #  get prediction (forward)
        output = model(inputs)

        # calculate losses
        loss = model.loss(output, labels)

        # backward propagate
        loss.backward()

        # take a step in gradient direction
        optimizer.step()

        # print losses
        losses.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # get prediction
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)

            # calculate loss
            test_loss += model.loss(output, labels).item()
            num_correct = (predicted == labels).sum().item()
            correct += num_correct
            total += labels.size(0)

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, test_accuracy))
    return test_loss, test_accuracy

def train_acc(model, train_dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total
