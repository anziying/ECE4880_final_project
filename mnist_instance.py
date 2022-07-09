import torch
import numpy as np
import cv2
import os
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
    You can use this function to load all images in a certain folder into a list of np.ndarray
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def load_images_from_folder_10_class(folder):
    """
    :param folder: The folder that loads images
    :return: image, labels
    Set "folder" as the path of training dataset (e.g. "distraction_data/imgs/train").
    There should be ten folders (c0-c9) under the path.
    The function returns images in a list of np.ndarray and a list of classification labels.
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


"""
Set your path of training dataset here
"""

# train_dataset_folder = r"distraction_data/imgs/train"
train_dataset_folder = r"small_dataset/imgs/train"

"""
Comment out the following snippet if you have your.pt files of images and labels in the root of directory. 
"""
############################Training set#############################
#
# train_imgs, train_labels = load_images_from_folder_10_class(train_dataset_folder)
# tensor_x_train = torch.Tensor(train_imgs) # transform to torch tensor
# tensor_y_train = torch.Tensor(train_labels)
#
# torch.save(tensor_x_train, 'training_image_tensor_7_8.pt')
# torch.save(tensor_y_train, 'training_label_tensor_7_8.pt')

#############################Testing set#############################

test_dataset_folder = r"small_test"

test_imgs, test_labels = load_images_from_folder_10_class(test_dataset_folder)
tensor_x_test = torch.Tensor(test_imgs) # transform to torch tensor
tensor_y_test = torch.Tensor(test_labels)

torch.save(tensor_x_test, 'testing_image_tensor_7_8.pt')
torch.save(tensor_y_test, 'testing_label_tensor_7_8.pt')


"""
The following code loads the saved torch.Tensor file into a Dataloader
"""

tensor_x_train = torch.load('training_image_tensor_7_8.pt')
tensor_y_train = torch.load('training_label_tensor_7_8.pt')
tensor_x_test = torch.load('testing_image_tensor_7_8.pt')
tensor_y_test = torch.load('testing_label_tensor_7_8.pt')

train_dataset = TensorDataset(tensor_x_train, tensor_y_train)  # create your datset
train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)  # create your dataloader

test_dataset = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)  # create your dataloader

def conv_out_size(slen, kernel_size, stride):
    """
    :param slen: Size length of the image. Should be an int.
    :param kernel_size: Int
    :param stride: Int
    :return: The size length of output after convolution
    This function considers 1-dim case.
    """
    return int((slen - kernel_size) / stride + 1)


class MultiClassConvNet(torch.nn.Module):
    def __init__(self, image_size_tup, num_classes):
        """
        :param image_size_tup: should be (480, 640) in this project
        :param num_classes: 10
        This is a naive example of CNN. Please feel free to modify its stucture.
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
    """

    :param model:
    :param transform_fn:
    :param loss_fn:
    :param optimizer:
    :param dataloader:
    :param num_epochs:
    :return:

    Use this function to train your model.
    """
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
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    accuracy = (y_true.squeeze(-1) == y_pred).float().mean()
    return accuracy


convnet = MultiClassConvNet(image_size_tup=(480, 640), num_classes=10)

convnet_optimizer = torch.optim.Adam(convnet.parameters(), lr=0.0002)


def s(x):
    return x


loss_functions = [torch.nn.CrossEntropyLoss(), nn.NLLLoss()]

train_loop(convnet, s, loss_functions[0], convnet_optimizer, train_dataloader, 10)

acc = calculate_test_accuracy(convnet, s, test_dataloader)

print(acc)
pass

pass