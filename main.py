import torch
import model as n
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print("loading data...")
train_data = [] # TODO
test_data = [] # TODO
print("done")
print()

print("making model...")
# TODO
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0005
PRINT_INTERVAL = 10
m = n.ConvNet()
print("done")
print()

print("training...")
optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []

try:
    for epoch in range(1, EPOCHS + 1):
        train_loss = n.train(m, optimizer, train_loader, epoch, PRINT_INTERVAL)
        test_loss, test_accuracy = n.test(m, test_loader)
        train_accuracy = n.train_acc(m. train_loader)
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
    plt.show()

    ep, val = zip(*train_accuracies)
    plt.plot(val)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracies vs Epoch')
    plt.show()

    ep, val = zip(*test_losses)
    plt.plot(val)
    plt.xlabel('Epoch')
    plt.ylabel('Testing Loss')
    plt.title('Testing Loss vs Epoch')
    plt.show()

    ep, val = zip(*test_accuracies)
    plt.plot(val)
    plt.xlabel('Epoch')
    plt.ylabel('Testing Accuracy')
    plt.title('Testing Accuracies vs Epoch')
    plt.show()

    print("done")
    print()