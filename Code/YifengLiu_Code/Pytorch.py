{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf100
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from PIL import Image\
import numpy as np\
import os\
import cv2\
import glob\
import torch\
from torchvision import transforms, datasets\
import torch.nn as nn\
from torch.autograd import Variable\
import matplotlib.pyplot as plt\
from termcolor import *\
import torch.optim.lr_scheduler as auto\
\
train_dir = '/home/ubuntu/home/ubuntu/final_pytorch/data_2/train'\
test_dir = '/home/ubuntu/home/ubuntu/final_pytorch/data_2/test'\
\
a = transforms.Compose([\
        transforms.RandomResizedCrop(75),\
        transforms.RandomHorizontalFlip(),\
        transforms.ToTensor(),\
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\
    ])\
\
b = transforms.Compose([\
        transforms.RandomResizedCrop(75),\
#        transforms.RandomHorizontalFlip(),\
        transforms.ToTensor(),\
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\
    ])\
\
train_datasets = datasets.ImageFolder(os.path.join(train_dir),\
                                       transform = a)\
\
test_datasets = datasets.ImageFolder(os.path.join(test_dir),\
                                       transform = b)\
\
print(len(train_datasets))\
\
\
num_epochs = 130\
batch_size = 32\
learning_rate = 0.0003\
# -----------------------------------------------------------------------------------\
\
\
# Data Loader (Input Pipeline)\
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size,shuffle=True, num_workers=4)\
\
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False,  num_workers=4)\
# -----------------------------------------------------------------------------------\
# CNN Model (4 conv layer)\
\
class CNN(nn.Module):\
    def __init__(self):\
        super(CNN, self).__init__()\
        self.layer1 = nn.Sequential(\
            nn.Conv2d(3, 64, kernel_size=4),\
            nn.BatchNorm2d(64),\
            nn.ReLU(),\
            nn.MaxPool2d(kernel_size=2, stride=2),\
            nn.Dropout2d(p=0.2)\
        )\
        self.layer2 = nn.Sequential(\
            nn.Conv2d(64, 128, kernel_size=5),\
            nn.BatchNorm2d(128),\
            nn.ReLU(),\
            nn.MaxPool2d(kernel_size=2, stride=2),\
            nn.Dropout2d(p=0.2)\
        )\
        self.layer3 = nn.Sequential(\
            nn.Conv2d(128, 256, kernel_size=5),\
            nn.BatchNorm2d(256),\
            nn.ReLU(),\
            nn.MaxPool2d(kernel_size=2, stride=2),\
            nn.Dropout2d(p=0.2)\
        )\
        self.layer4 = nn.Sequential(\
            nn.Conv2d(256, 512, kernel_size=4),\
            nn.BatchNorm2d(512),\
            nn.ReLU(),\
            nn.MaxPool2d(kernel_size=2, stride=2),\
            nn.Dropout2d(p=0.2)\
        )\
#        self.drop1 = nn.Dropout2d(p=0.2)\
        self.fc1 = nn.Sequential(\
            nn.Linear(512,256),\
            nn.ReLU(),\
            nn.Dropout(p=0.2)\
        )\
        self.fc2 = nn.Sequential(\
            nn.Linear(256,2),\
            nn.ReLU(),\
            nn.Dropout(p=0.2)\
        )\
        self.sigmoid = nn.Sigmoid()\
\
    def forward(self, x):\
        out = self.layer1(x)\
        out = self.layer2(out)\
        out = self.layer3(out)\
        out = self.layer4(out)\
        out = out.view(out.size(0), -1)\
        out = self.fc1(out)\
        out = self.fc2(out)\
        out = self.sigmoid(out)\
        return out\
# -----------------------------------------------------------------------------------\
cnn = CNN()\
cnn.cuda()\
# -----------------------------------------------------------------------------------\
# Loss and Optimizer\
criterion = nn.CrossEntropyLoss()\
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)\
#scheduler = auto.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)\
\
train_loss_epoch = []\
Accuracy_epoch = []\
# -----------------------------------------------------------------------------------\
# Train the Model\
for epoch in range(num_epochs):\
    loss_mini = []\
    correct_epoch = []\
    for i, (images, labels) in enumerate(train_loader):\
        total = 0\
        correct = 0\
        images = Variable(images).cuda()\
        target = labels\
        labels = Variable(labels).cuda()\
\
        # Forward + Backward + Optimize\
        optimizer.zero_grad()\
        outputs = cnn(images)\
        loss = criterion(outputs, labels)\
        loss.backward()\
        optimizer.step()\
#        scheduler.step()\
#        train_loss.append(loss.data[0])\
        loss_mini.append(loss.data[0])\
        _, predicted = torch.max(outputs.data, 1)\
        total += labels.size(0)\
        correct += (predicted == target.cuda()).sum()\
        correct_epoch.append(100*correct/total)\
        if (i + 1) % 10 == 0:\
            print(colored('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Accu: %d %%'\
                % (epoch + 1, num_epochs, i + 1, len(train_datasets) // batch_size, loss.data[0],100 * correct / total),"grey"))\
        # -----------------------------------------------------------------------------------\
    Accuracy_epoch.append((np.array(correct_epoch).mean()))\
    train_loss_epoch.append((np.array(loss_mini).mean()))\
\
# -----------------------------------------------------------------------------------\
# Test the Model\
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).\
\
correct = 0\
total = 0\
\
for images, labels in test_loader:\
    images = Variable(images).cuda()\
    outputs = cnn(images)\
    _, predicted = torch.max(outputs.data, 1)\
    total += labels.size(0)\
    correct += (predicted == labels.cuda()).sum()\
# -----------------------------------------------------------------------------------\
print(colored('Test Accuracy of the model on the %d test images: %d %%' % (len(test_datasets),100 * correct / total),"red"))\
# -----------------------------------------------------------------------------------\
# Save the Trained Model\
torch.save(cnn.state_dict(), 'cnn.pt')\
\
plt.figure(0)\
plt.plot(np.arange(len(train_loss_epoch)), np.array(train_loss_epoch))\
plt.xlabel('Number of Iteration')\
plt.ylabel('Training Loss Values')\
plt.title('Training Loss of each epoch')\
\
plt.figure(1)\
plt.plot(np.arange(len(Accuracy_epoch)), np.array(Accuracy_epoch))\
plt.xlabel('Number of Iteration')\
plt.ylabel('Training Accuracy percentage')\
plt.title('Training Accuracy of each epoch')\
\
plt.show()\
}