import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import math
from PIL import Image
from torchsummary import summary
from numpy import asarray

image_dir = "/content/Images"
SAVE_DIR = "/content/Results"
piece_dir = {}
batch_size = 128
image_size = (128, 128)  # Subject to change
epochs = 50

def popPeiceDir():
    for dir in os.listdir(image_dir):
        if dir.startswith('.'):
            continue
        piece_dir[dir] = os.path.join(image_dir, dir)


popPeiceDir()


# Empty Square = 0, Pawn = 1, King = 2, Queen = 3, Rook = 4, Bishop = 5, Knight = 6
def getFilesAndLabels():
    fl = []
    for key in piece_dir:
        if key == "Testimg" or key == "Testimg2":
            continue
        for file in os.listdir(piece_dir[key]):
            if file.startswith('.'):
                continue
            label = -1
            if key == "BB" or key == "WB":
                label = 5
            if key == "BK" or key == "WK":
                label = 6
            if key == "BKi" or key == "WKi":
                label = 2
            if key == "BP" or key == "WP":
                label = 1
            if key == "BQ" or key == "WQ":
                label = 3
            if key == "BR" or key == "WR":
                label = 4
            if key == "Empty":
                label = 0
            assert label != -1
            fl.append([os.path.join(piece_dir[key], file), label])
    return fl


def transformImg(img):
    preproc = transforms.Compose([transforms.Grayscale(),
                                  transforms.Resize(image_size), transforms.ToTensor(),
                                  transforms.Normalize((0.5), (0.5))])
    return preproc(img)


def getArr(image_path):
    img = Image.open(image_path)
    img = transformImg(img)
    return img


class DataSet(Dataset):
    def __init__(self, files_labels):
        self.fl = files_labels
        self.count = 0

    def __len__(self):
        return len(self.fl)

    def __getitem__(self, index):
        img_arr = getArr(self.fl[index][0])
        #label = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        #label[self.fl[index][1]] = 1
        label = self.fl[index][1]
        return img_arr.float(), label


# Starting Image dims, Maxpooling Kernal, number of times maxpool is called in CNN, number of conv channels
def findFCFeatures(images_size, maxpooling, num_maxpool, num_channels):
    x, y = images_size
    mx, my = maxpooling
    for i in range(num_maxpool):
        x = math.floor(((x - (mx - 1) - 1) / mx) + 1)
        y = math.floor(((y - (my - 1) - 1) / my) + 1)
    return x * y * num_channels


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_down1 = double_conv(1, 32)
        self.conv_down2 = double_conv(32, 64)
        self.conv_down3 = double_conv(64, 128)
        self.conv_down4 = double_conv(128, 256)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(findFCFeatures(image_size, (2, 2), 3, 256), 7)

    def forward(self, x):
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        x = self.conv_down4(x)
        x = x.view(-1, findFCFeatures(image_size, (2, 2), 3, 256))
        x = self.fc1(x)
        return x

def train(GPU=True):
    data = getFilesAndLabels()
    # print(data)
    dataset = DataSet(data)
    model = Net()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if GPU:
        print(device)
        model.to(device)
    w = torch.tensor(np.array([1, 1, 1, 1, 1, 1, 1]))
    loss_func = nn.CrossEntropyLoss()

    tess = math.floor(len(data) * 0.2)
    trss = len(data) - tess

    train_subsamples, test_subsamples = torch.utils.data.random_split(range(len(data)), [trss, tess])


    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=0, sampler=train_subsamples)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=0, sampler=test_subsamples)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    summary(model, (1, 128, 128))
    for epoch in range(epochs):
        model.train()
        train_num_correct = 0
        test_num_correct = 0
        for i, (inp, label) in enumerate(train_loader):
            if GPU:
                inp = inp.to(device)
                label = label.to(device)
            optimizer.zero_grad()
            cur_pre = model(inp)
            loss = loss_func(cur_pre, label.long())
            train_loss.append([loss, epoch])

            for i in range(cur_pre.shape[0]):
                if label[i] == torch.argmax(cur_pre[i]):
                    train_num_correct += 1
            # print('ML Started')
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for i, (inp, label) in enumerate(test_loader):
                if GPU:
                    inp = inp.to(device)
                    label = label.to(device)
                cur_pre = model(inp)
                loss = loss_func(cur_pre, label.long())
                test_loss.append([loss, epoch])

                for i in range(cur_pre.shape[0]):
                    if label[i] == torch.argmax(cur_pre[i]):
                        test_num_correct += 1

        train_acc.append(train_num_correct / trss)
        test_acc.append(test_num_correct / tess)
        print(f'Finished Epoch {epoch + 1}')
    print(f'Train Acc: {train_acc}')
    print(f'Test Acc: {test_acc}')
    print(f'Train Loss: {train_loss}')
    print(f'Test Loss: {test_loss}')
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    torch.save(model.state_dict(), SAVE_DIR)
    np.savetxt(os.path.join(SAVE_DIR, 'trainAcc.csv'), train_acc, delimiter=',')
    np.savetxt(os.path.join(SAVE_DIR, 'testAcc.csv'), test_acc, delimiter=',')

train(GPU=True)