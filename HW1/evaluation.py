import csv
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils import data


def test():
    print('Testing...')

    model.eval()

    total = 0
    pred = []
    with torch.no_grad():
        for index, (images, labels) in enumerate(test_loader):
            # print(images.shape)
            images = Variable(images, requires_grad=True).float().cuda()
            labels = Variable(labels).cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            pred.append(predicted.item())

    # print(total)
    return pred


def cv2_loader(path):
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return np.expand_dims(img_gray, axis=2)


if __name__ == '__main__':
    para_name = "c_res50_sgd_l5e-3_b32_47.pth"

    # model = models.densenet201()
    # model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
    #                         padding=3, bias=False)
    # model.classifier = nn.Linear(1920, 13)
    # model = model.cuda()

    model = models.resnet50()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
    model.fc = nn.Linear(2048, 13)
    model = model.cuda()

    saved_state_dict = torch.load(os.path.join("checkpoint", para_name))
    model.load_state_dict(saved_state_dict)

    print('Loading data set...')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4525,), (0.2204,)),
    ])

    test_loader = data.DataLoader(
        datasets.DatasetFolder(root='./dataset/test', loader=cv2_loader, extensions=['.jpg'], transform=transform_test),
        batch_size=1,
        shuffle=False, num_workers=5,
        pin_memory=True)

    classes = [
        "bedroom",
        "coast",
        "forest",
        "highway",
        "insidecity",
        "kitchen",
        "livingroom",
        "mountain",
        "office",
        "opencountry",
        "street",
        "suburb",
        "tallbuilding"]

    pred = test()

    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['id', 'label'])

        for i, p in enumerate(pred):
            writer.writerow(['image_' + str(format(i, '04d')), classes[p]])
