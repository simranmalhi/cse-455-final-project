import torch
import model as n
import data_parsing as dp
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print("loading data...")
train_loader = dp.data['train']
test_loader = dp.data['test']
print("done")
print()

print("making model...")
TRAIN_BATCH_SIZE = dp.BATCH_SIZE
TEST_BATCH_SIZE = dp.BATCH_SIZE
EPOCHS = 5
LEARNING_RATE = 0.002
MOMENTUM = 0.9
WEIGHT_DECAY = 0.005
PRINT_INTERVAL = 10
m = n.ConvNet()
print("done")
print()

print("training...")
optimizer = optim.SGD(m.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m.to(device)

try:
    for epoch in range(1, EPOCHS + 1):
        train_loss = n.train(m, optimizer, train_loader, epoch, PRINT_INTERVAL, device)
        test_loss, test_accuracy = n.test(m, test_loader, device)
        train_accuracy = n.train_acc(m, train_loader, device)
        train_losses.append((epoch, train_loss))
        train_accuracies.append((epoch, train_accuracy))
        test_losses.append((epoch, test_loss))
        test_accuracies.append((epoch, test_accuracy))
    print("done")
    print()
except KeyboardInterrupt as ke:
        print('Interrupted')
except:
    import traceback
    traceback.print_exc()
finally:
    print("evaluating model...")

    ep, val = zip(*train_losses)
    plt.plot(val)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epoch')
    plt.savefig('train_loss.png')
    plt.clf()


    ep, val = zip(*train_accuracies)
    plt.plot(val)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracies vs Epoch')
    plt.savefig('train_acc.png')
    plt.clf()

    ep, val = zip(*test_losses)
    plt.plot(val)
    plt.xlabel('Epoch')
    plt.ylabel('Testing Loss')
    plt.title('Testing Loss vs Epoch')
    plt.savefig('test_loss.png')
    plt.clf()


    ep, val = zip(*test_accuracies)
    plt.plot(val)
    plt.xlabel('Epoch')
    plt.ylabel('Testing Accuracy')
    plt.title('Testing Accuracies vs Epoch')
    plt.savefig('test_acc.png')
    plt.close()

    print("done")
    print()
