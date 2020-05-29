import torch as t
import torch.nn as nn
import torch.optim as optim


def fit(model, dataloader, epochs, criterion=nn.CrossEntropyLoss()):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1, epochs+1):
        for i, [x, y] in enumerate(dataloader):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("the {} eoch {} iteration loss: {}".format(epoch, i, loss.detach().numpy()))


def test(model, dataloader):
    pass


