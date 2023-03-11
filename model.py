import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
#import tqdm

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # init

        # TODO
        RGB = 3 # rgb channel count
        LETTER_OUTPUT = 29 # output 26 letters + 3 other things
        IMAGE_SIZE = 200 # pixel width of image
        filter_size = 5
        c1_out_size = 6
        c2_out_size = 12

        # 3 channel input, 6 channel output, applies 5x5 filters
        self.c1 = nn.Conv2d(RGB, c1_out_size, filter_size)
        # TODO: uncomment
        # self.bn1 = nn.BatchNorm2d()
        self.c2 = nn.Conv2d(c1_out_size, c2_out_size, filter_size)
        # TODO: uncomment
        # self.bn2 = nn.BatchNorm2d()

        ## we can do the math or just run and see what size it needs to be
        # new_image_size = IMAGE_SIZE-(2*(filter_size-1))
        # self.output = nn.Linear(442368, LETTER_OUTPUT)
        self.output = nn.Linear(230496, LETTER_OUTPUT)

    def forward(self, x):
        # go through layers
        x = self.c1(x)
        # # x = self.bn1(x) # TODO: uncomment
        # x = f.relu(x)
        # x = self.c2(x)
        # # x = self.bn2(x) # TODO: uncomment
        # x = f.relu(x)

        # return prediction
        x = torch.flatten(x, 1)
        pred = self.output(x)
        return pred

    def inference(self, x):
        return self.forward(x) # gets and returns prediction

    # def loss(self):
    #     return nn.CrossEntropyLoss()
    def loss(self, predictions, labels):
        return nn.CrossEntropyLoss()(predictions, labels)


# returns mean loss for single epoch
def train(model, optimizer, train_loader, epoch, log_interval, device):
    model.train()
    losses = []
    # for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
    for batch_idx, batch in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch[0], batch[1]
        # TODO: GPU
        # inputs, labels = batch[0].to(device), batch[1].to(device)

        # zero parameter gradients
        optimizer.zero_grad()
        #  get prediction (forward)
        output = model(inputs)

        # calculate losses
        # loss = model.loss(output, labels)
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
        break # TODO remove break to train
    return np.mean(losses)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs, labels = batch[0], batch[1]
            # TODO: GPU
            # inputs, labels = batch[0].to(device), batch[1].to(device)

            # get prediction
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)

            # calculate loss
            test_loss += model.loss(output, labels).item()
            num_correct = (predicted == labels).sum().item()
            correct += num_correct
            total += labels.size(0)
            break # TODO: REMOVE
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total

    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
        test_loss, test_accuracy))
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
            break # TODO: remove
    return correct / total