from prototree.prototree import ProtoTree
from util.log import Log

from util.args import get_args, save_args, get_optimizer
from util.data import get_dataloaders
from util.init import init_tree
from util.net import get_network, freeze
from util.visualize import gen_vis
from util.analyse import *
from util.save import *
from prototree.train import train_epoch, train_epoch_kontschieder
from prototree.test import eval, eval_fidelity
from prototree.prune import prune
from prototree.project import project, project_with_class_constraints
from prototree.upsample import upsample

import torch
from shutil import copy
from copy import deepcopy
import torchvision.transforms as transforms
import torchvision

import torch.nn as nn
import torch.optim as optim

def get_nih(
    augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size=224
):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose(
        [transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), normalize]
    )
    if augment:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    size=(img_size + 32, img_size + 32)
                ),  # resize to 256x256
                transforms.RandomOrder(
                    [
                        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                        transforms.ColorJitter(brightness=0.25, contrast=0.25),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomAffine(
                            15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                        ),
                    ]
                ),
                transforms.RandomCrop(size=(img_size, img_size)),  # crop to 224x224
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transform_no_augment
    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(
        project_dir, transform=transform_no_augment
    )
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes
    return trainset, projectset, testset, classes, shape

def get_dataloaders(batch_size = 32):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, testset, classes, shape = get_nih(True,
            "./data/NIH_CHEST_XRAYS/dataset/train_corners",
            "./data/NIH_CHEST_XRAYS/dataset/train_crop",
            "./data/NIH_CHEST_XRAYS/dataset/test_full", img_size=224)
    c, w, h = shape
    # Determine if GPU should be used
    
    # Uncomment for CUDA
    cuda = torch.cuda.is_available()

    # Uncomment for MPS
    # cuda = not args.disable_cuda and torch.backends.mps.is_available()
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, pin_memory=cuda
    )
    projectloader = torch.utils.data.DataLoader(
        projectset,
        #    batch_size=args.batch_size,
        batch_size=int(
            batch_size / 4
        ),  # make batch size smaller to prevent out of memory errors during projection
        shuffle=False,
        pin_memory=cuda,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, pin_memory=cuda
    )
    print("Num classes (k) = ", len(classes), flush=True)
    return trainloader, projectloader, testloader, classes, c

if torch.cuda.is_available():
        # device = torch.device('cuda')
    device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
else:
    device = torch.device('cpu')


trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(
        batch_size=32
    )

EPOCHS = 100

# Assuming `trainloader` is already defined and contains the training data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define ResNet-50 model
resnet50 = torchvision.models.resnet50(pretrained=False)
resnet50.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)

# Training loop
resnet50.train()  # Set model to training mode
for epoch in range(EPOCHS):  # Loop over the dataset multiple times
    print("Epoch: ", epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Save the trained model
torch.save(resnet50, 'runs/run_prototree/checkpoints/best_test_model/resnet50.pth')

