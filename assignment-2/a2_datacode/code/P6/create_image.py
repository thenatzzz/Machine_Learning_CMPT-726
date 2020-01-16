import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch
import os
import json
import math

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

    fig = plt.gcf()
    fig.set_size_inches(1.4, 1.4)

    plt.show()

def save_image(index,img,num,image_name,is_score=True):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

    fig = plt.gcf()
    fig.set_size_inches(1.4, 1.4)

    if is_score:
        name = "Score:"
    else:
        name = 'Loss:'
    fig.suptitle(str(index)+")"+name+str(num), fontsize=8)
    fig.savefig("sample_test_image/"+image_name, dpi=80,bbox_inches='tight')


TEST_BATCH_SIZE = 1
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                     std=(0.2023, 0.1994, 0.2010))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE,
shuffle=False, num_workers=0)

if __name__ == '__main__':
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    SHOW_IMAGE = False
    SAVE_IMAGE = True
    SOFTMAX_OUTPUT = True
    USE_SCORE = True

    if SHOW_IMAGE:
        # get some random training images
        testiter_show = iter(testloader)
        images, labels = testiter_show.next()
        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(TEST_BATCH_SIZE)))
        # show images
        imshow(torchvision.utils.make_grid(images))

    if SAVE_IMAGE:
        NUM_IMAGE_SAVED = 10000
        #REQUIREMENTS
        # NEED TO LOAD JSON file for loss score of each test image (cifar_finetune.py)
        # NEED TEST_BATCH_SIZE = 1 before using this function
        if USE_SCORE:
            score_dict = json.load(open('test_img_score.json'))
            input_dict = score_dict
            key="image_score"
        else:
            loss_dict = json.load(open('test_img_loss.json'))
            input_dict = loss_dict
            key="image_loss"

        testiter = iter(testloader)
        for index in range(0,10000):
            test_image,test_label = testiter.next()

            image_torch = torchvision.utils.make_grid(test_image)
            single_class = classes[test_label]
            image_name = "test_image"+str(index)
            if SOFTMAX_OUTPUT and USE_SCORE:
                number = round(math.exp(input_dict[key+str(index)])*100,2)
                number = str(number)+"%"
            else:
                number = round(input_dict[key+str(index)],3)

            save_image(index,image_torch,number,image_name,is_score=USE_SCORE)
            if NUM_IMAGE_SAVED == index:
                break
            print("index: ",index)
