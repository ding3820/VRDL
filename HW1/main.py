import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils import data


def train(epoch):
    print('Training...')
    model.train()

    train_loss = 0
    total = 0
    correct = 0

    for index, (images, labels) in enumerate(train_loader):
        images = Variable(images, requires_grad=True).float().cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    torch.save(model.state_dict(),
               os.path.join("checkpoint", "c_" + para_name + "_" + str(epoch) + '.pth'))

    print("Loss: ", (train_loss / total), "Acc:", (correct / total))


def val(epoch):
    print('Validating...')
    model.eval()

    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for index, (images, labels) in enumerate(val_loader):
            images = Variable(images, requires_grad=True).float().cuda()
            labels = Variable(labels).cuda()

            outputs = model(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print("Loss: ", (test_loss / total), "Acc:", (correct / total))

    with open("val_loss.txt", "a") as f_loss:
        f_loss.write("%s\n" % (test_loss / total))
    with open("val_acc.txt", "a") as f_acc:
        f_acc.write("%s\n" % (correct / total))


def cv2_loader(path):
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return np.expand_dims(img_gray, axis=2)


def adjust_learning_rate(optimizer, epoch):
    if epoch % 8 == 0 and epoch != 0 and epoch != 8:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


if __name__ == '__main__':

    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
    model.fc = nn.Linear(2048, 13)
    model = model.cuda()

    # model = models.densenet201(pretrained=True)
    # model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
    #                         padding=3, bias=False)
    # model.classifier = nn.Linear(1920, 13)
    # model = model.cuda()

    # saved_state_dict = torch.load("save.pth")
    # model.load_state_dict(saved_state_dict)

    cudnn.benchmark = True

    para_name = "res50_sgd_l5e-3_b32"
    batch_size = 32
    lr = 5e-3
    epochs = 50

    print('Loading data set...')
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.4525,), (0.2204,)),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize((256, 256)),
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4525,), (0.2204,)),
    ])

    train_loader = data.DataLoader(
        datasets.DatasetFolder(root='./dataset/train', loader=cv2_loader, extensions=['.jpg'],
                               transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=5,
        pin_memory=True)

    val_loader = data.DataLoader(
        datasets.DatasetFolder(root='./dataset/val', loader=cv2_loader, extensions=['.jpg'], transform=transform_test),
        batch_size=1,
        shuffle=False, num_workers=5,
        pin_memory=True)

    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        train(epoch)
        val(epoch)
        adjust_learning_rate(optimizer, epoch)
        print("...Done!")
        print("-" * 20)
