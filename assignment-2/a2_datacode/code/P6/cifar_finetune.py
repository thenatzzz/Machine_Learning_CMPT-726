'''
This is starter code for Assignment 2 Problem 6 of CMPT 726 Fall 2019.
The file is adapted from the repo https://github.com/chenyaofo/CIFAR-pretrained-models
'''

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from math import exp
import numpy as np
import json
NUM_EPOCH = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

######################################################
####### Do not modify the code above this line #######
######################################################
import torch.nn.functional as F
USE_NEW_ARCHITECTURE = False

class cifar_resnet20(nn.Module):
    def __init__(self):
        super(cifar_resnet20, self).__init__()
        ResNet20 = CifarResNet(BasicBlock, [3, 3, 3])
        url = 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet20-8412cc70.pth'
        ResNet20.load_state_dict(model_zoo.load_url(url))
        modules = list(ResNet20.children())[:-1]
        backbone = nn.Sequential(*modules) # * is to unpack argument lists
        self.backbone = nn.Sequential(*modules)

        if USE_NEW_ARCHITECTURE:
            hidden_nodes = 256
            self.fc1 = nn.Linear(in_features = 64, out_features = hidden_nodes) #
            self.fcbn1 = nn.BatchNorm1d(hidden_nodes)#
            self.fc = nn.Linear(in_features = hidden_nodes, out_features = 10)#
            self.dropout_rate =0.0  #To set dropout rate
        else:
            self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.shape[0], -1)

        if USE_NEW_ARCHITECTURE:
            out = F.dropout(F.relu(self.fcbn1(self.fc1(out))),
                            p=self.dropout_rate, training=self.training)
            out = self.fc(out)
            return F.log_softmax(out, dim=1)
        else:
            return F.log_softmax(self.fc(out),dim=1)
            # return self.fc(out)

def accuracy(out, labels):
    total =0
    for index in range(0,len(out)):
        if out[index] == labels[index]:
            total += 1
    return total, total/len(out)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# BATCH_SIZE = 128
BATCH_SIZE = 64


if __name__ == '__main__':
    model = cifar_resnet20()
    model.to(device) #.cuda() to use GPU

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10('./data', download=True, transform=transform) #Number of datapoint = 50000
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)#num_workers=2
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc.parameters()), lr=0.001, momentum=0.9,weight_decay=0.0)
    # optimizer = optim.SGD(list(model.backbone.parameters()), lr=0.001, momentum=0.9,weight_decay=0.0)
    # optimizer = optim.SGD(list(model.fc.parameters())+list(model.backbone.parameters()), lr=0.001, momentum=0.9,weight_decay=0.0)

    ############################################################
    ### Do the training
    DO_TRAINING = True
    DO_TESTING = True
    NUM_EPOCH_TRAINING = 1

    if DO_TRAINING:
        model.train()
        for epoch in range(NUM_EPOCH_TRAINING):  # loop over the dataset multiple times
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs.to(device)) #.cuda() to use GPU
                loss = criterion(outputs.to(device), labels.to(device)) #.cuda() to use GPU
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 20 == 19:    # print every 20 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0
        print('Finished Training')

    with torch.no_grad():
        TEST_BATCH_SIZE = 20
        SAVING_JSON = False #True
        if DO_TESTING:
        ### DO TESTING/EVALUATION
            model.eval()
            testset = datasets.CIFAR10(root='./data', train=False,
            download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE,
            shuffle=False, num_workers=0)

            running_test_loss = 0
            match= running_match = 0
            total_test_loss,total_accuracy = 0,0
            score_dict = {}
            loss_dict ={}
            for i, test_data in enumerate(testloader, 0):
                # get the inputs
                test_inputs, test_labels = test_data
                test_outputs = model(test_inputs.to(device))
                test_loss = criterion(test_outputs.to(device), test_labels.to(device))
                pred = test_outputs.argmax(dim=1, keepdim=True) # get the index of the max log-probability

                running_test_loss += test_loss.item()
                total_test_loss += test_loss

                if TEST_BATCH_SIZE==1:
                    loss_dict['image_loss'+str(i)] = test_loss.item()
                    score = test_outputs[0][pred].to('cpu')
                    score_dict['image_score'+str(i)] = score.item()

                    if pred == test_labels.to(device) :
                        match += 1
                        total_accuracy += 1
                        running_match +=1
                        # print("found MATCH !!!!!!!!!!!!!!!")

                if TEST_BATCH_SIZE>1:
                    temp_match , accuracy_batch = accuracy( pred.to('cpu'),test_labels.to('cpu') )
                    match += temp_match
                    running_match += temp_match
                    total_accuracy += accuracy_batch

                if i % 20 == 19:    # print every 20 mini-batches
                    print('[At %d] loss: %.3f, accuracy: %.3f, cumulative match: %i' %( i + 1, running_test_loss / 20, running_match/20,match))
                    running_match =0.0
                    running_test_loss = 0.0

            if TEST_BATCH_SIZE==1 and SAVING_JSON:
                json.dump(loss_dict,open("test_img_loss1.json",'w'))
                json.dump(score_dict,open("test_img_score1.json",'w'))
                print('Save: 2 json files successfully!')

            print("total_accuracy:",total_accuracy)
            print("total_test_loss:",total_test_loss)
            print("Number of match found: ",match) #500 match for no train , 6500 match for 5 epoch train
            print('Finished Testing')
