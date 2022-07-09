# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
rng_seed = 507
torch.manual_seed(rng_seed)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(255 - img.squeeze(), cmap="gray")
# plt.show()


def train_loop(model, transform_fn, loss_fn, optimizer, dataloader, num_epochs):
    tbar = tqdm(range(num_epochs))
    for _ in tbar:
        loss_total = 0.
        for i, (x, y) in enumerate(dataloader):
            x = transform_fn(x)

            pred = model(x)
            loss = loss_fn(pred, y.squeeze(-1))
            # print(pred)
            # print(y.squeeze(-1))
            ## Parameter updates
            model.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
        tbar.set_description(f"Train loss: {loss_total / len(dataloader)}")

    return loss_total / len(dataloader)


def calculate_test_accuracy(model, transform_fn, test_dataloader):
    y_true = []
    y_pred = []
    tf = nn.Flatten()
    for (xi, yi) in test_dataloader:
        xi = transform_fn(xi)
        pred = model(xi)
        yi_pred = pred.argmax(-1)
        y_true.append(yi)
        y_pred.append(yi_pred)
    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    accuracy = (y_true == y_pred).float().mean()
    return accuracy


class MultiClassMLP(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        """
        Arguments:
            num_features: The number of features in the input.
            num_hidden: Number of hidden features in the hidden layer:
            num_classes: Number of possible classes in the output
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.Sigmoid(),
            nn.Linear(num_hidden, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
train_dataloader = DataLoader(training_data, batch_size=100, shuffle=True, num_workers=0)



#####################################################################
#####################################################################
def conv_out_size(slen, kernel_size, stride):
    return int((slen - kernel_size) / stride + 1)


class MultiClassConvNet(torch.nn.Module):
    def __init__(self, side_length, conv_channels_1, conv_channels_2, linear_hidden, num_classes):
        """
        Arguments:
            side_length: Side-length of input images (assumed to be square)
            conv_channels_1: Number of channels output from first conv layer
            conv_channels_2: Number of channels output from second conv layer
            linear_hidden: Number of hidden units in linear layer
            num_classes: Number of classes in output
        """
        super().__init__()
        self.kernel_size = 3

        self.conv1 = conv_out_size(side_length, self.kernel_size, 1)
        self.maxpool1 = conv_out_size(self.conv1, self.kernel_size, 1)
        self.conv2 = conv_out_size(self.maxpool1, self.kernel_size, 1)
        self.maxpool2 = conv_out_size(self.conv2, self.kernel_size, 1)

        self.net = nn.Sequential(
            nn.Conv2d(1,
                      conv_channels_1,
                      kernel_size=self.kernel_size,
                      stride=1
                      ),
            nn.MaxPool2d(kernel_size=self.kernel_size, stride=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels_1, conv_channels_2, kernel_size=self.kernel_size),
            nn.MaxPool2d(kernel_size=self.kernel_size, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_channels_2 * (self.maxpool2 ** 2), linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

convnet = MultiClassConvNet(side_length=28, conv_channels_1=8, conv_channels_2=16, linear_hidden=256, num_classes=10)

convnet_optimizer = torch.optim.Adam(convnet.parameters(), lr=0.0002)


def s(x):
    return x
train_loop(convnet, s, nn.NLLLoss(), convnet_optimizer, train_dataloader, 10)


test_data = datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor())
logistic_test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=True, num_workers=0)
calculate_test_accuracy(convnet, nn.Identity(), logistic_test_dataloader)

pass


