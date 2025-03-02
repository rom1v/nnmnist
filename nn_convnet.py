#!/usr/bin/env python3
import torch
import torchvision
import matplotlib.pyplot as plt


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                stride=(1, 1),
            ),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(576, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10),
        )

    def forward(self, x):
        logits = self.model(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 19:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def main():
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)

    training_data = torchvision.datasets.MNIST(
        root="torchvision_data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    train_dataloader = torch.utils.data.DataLoader(
        training_data, batch_size=64, drop_last=True, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=64, shuffle=False
    )

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")


if __name__ == "__main__":
    main()
