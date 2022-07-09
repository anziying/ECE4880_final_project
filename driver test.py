import torch
import numpy as np
import cv2
import os
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt

rng_seed = 507
torch.manual_seed(rng_seed)


def load_images_from_folder(folder):
    """
    :param folder: The folder that loads images
    :return: a list of ndarray-type images

    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def load_images_from_folder_10_class(folder):
    """
    :param folder: The folder that loads images
    :return: a list of ndarray-type images
    """
    images = []
    labels = []
    class_list = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    for idx in range(len(class_list)):
        sub_folder = os.path.join(folder, class_list[idx])
        for filename in os.listdir(sub_folder):
            img = cv2.imread(os.path.join(sub_folder, filename))
            if img is not None:
                img_frame = np.zeros((3, 480, 640))
                img_frame[0, :, :] = img[:, :, 0]
                img_frame[1, :, :] = img[:, :, 1]
                img_frame[2, :, :] = img[:, :, 2]
                images.append(img_frame)
                labels.append(np.array([idx]))
    return images, labels

# train_dataset_folder = r"distraction_data/imgs/train"
train_dataset_folder = r"small_dataset/imgs/train"

#
# train_imgs, train_labels = load_images_from_folder_10_class(train_dataset_folder)
# tensor_x_train = torch.Tensor(train_imgs) # transform to torch tensor
# tensor_y_train = torch.Tensor(train_labels)
#
# torch.save(tensor_x_train, 'training_image_tensor_7_8.pt')
# torch.save(tensor_y_train, 'training_label_tensor_7_8.pt')

tensor_x_train = torch.load('training_image_tensor_7_8.pt')
tensor_y_train = torch.load('training_label_tensor_7_8.pt')


my_dataset = TensorDataset(tensor_x_train,tensor_y_train) # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=100, shuffle=True) # create your dataloader


def conv_out_size(slen, kernel_size, stride):
    return int((slen - kernel_size) / stride + 1)

class MultiClassConvNet(torch.nn.Module):
    def __init__(self, image_size_tup, num_classes):
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

        self.conv1 = (conv_out_size(image_size_tup[0], self.kernel_size, 1),
                      conv_out_size(image_size_tup[1], self.kernel_size, 1))
        self.maxpool1 = (conv_out_size(self.conv1[0], self.kernel_size, 1),
                         conv_out_size(self.conv1[1], self.kernel_size, 1))
        self.conv2 = (conv_out_size(self.maxpool1[0], self.kernel_size, 1),
                      conv_out_size(self.maxpool1[1], self.kernel_size, 1))
        self.maxpool2 = (conv_out_size(self.conv2[0], self.kernel_size, 1),
                         conv_out_size(self.conv2[1], self.kernel_size, 1))

        self.net = nn.Sequential(
            nn.Conv2d(3,
                      3,
                      kernel_size=self.kernel_size,
                      stride=1
                      ),
            nn.MaxPool2d(kernel_size=self.kernel_size, stride=1),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=self.kernel_size),
            nn.MaxPool2d(kernel_size=self.kernel_size, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * (self.maxpool2[0] * self.maxpool2[1]), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

def train_loop(model, transform_fn, loss_fn, optimizer, dataloader, num_epochs):
    tbar = tqdm(range(num_epochs))
    for _ in tbar:
        loss_total = 0.
        for i, (x, y) in enumerate(dataloader):
            x = transform_fn(x)

            pred = model(x)
            y = y.type(torch.LongTensor)
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


convnet = MultiClassConvNet(image_size_tup=(480, 640),  num_classes=10)

convnet_optimizer = torch.optim.Adam(convnet.parameters(), lr=0.0002)


def s(x):
    return x

train_loop(convnet, s, nn.NLLLoss(), convnet_optimizer, my_dataloader, 10)

acc = calculate_test_accuracy(convnet, s, my_dataloader)

pass

