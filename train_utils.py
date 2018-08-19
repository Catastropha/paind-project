import argparse
import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', help='input path to directory containing 2 folders: train and valid.')
    parser.add_argument('--save_dir', type=str, default='', help='path to folder of saved checkpoints')
    parser.add_argument('--arch', type=str, default='vgg19', choices=['vgg19','alexnet','resnet152','densenet201', 'inception_v3'], help='choose architecture')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[1024, 1024, 512, 256], help='number of hidden units')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='learning rate for training')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to execute')
    parser.add_argument('--gpu', default=False, action='store_true', help='activate gpu if available')
    
    return parser.parse_args()


def data_loader(path):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(path+'train/', transform=train_transforms)
    valid_dataset = datasets.ImageFolder(path+'valid/', transform=valid_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    validloader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
    
    return train_dataset, trainloader, valid_dataset, validloader


def load_pretrained_model(name):
    if name == 'vgg19':
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[0].in_features
    elif name == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif name == 'resnet152':
        model = models.resnet152(pretrained=True)
        in_features = model.fc.in_features
    elif name == 'densenet201':
        model = models.densenet201(pretrained=True)
        in_features = model.classifier.in_features
    elif name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        in_features = model.fc.in_features

    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model, in_features



def create_classifier(model, in_features, units):
    skeleton = OrderedDict()
    
    # Input layer
    count = 0
    skeleton.update({'fc'+str(count): nn.Linear(in_features, units[count])})
    skeleton.update({'relu'+str(count): nn.ReLU()})
    skeleton.update({'drop'+str(count): nn.Dropout(p=0.05)})
    
    # Hidden layers
    count = 1
    while count < len(units):
        skeleton.update({'fc'+str(count): nn.Linear(units[count-1], units[count])})
        skeleton.update({'relu'+str(count): nn.ReLU()})
        skeleton.update({'drop'+str(count): nn.Dropout(p=0.05)})
        count += 1
    
    # Last layer
    skeleton.update({'fc'+str(count): nn.Linear(units[count-1], 102)})
    skeleton.update({'output': nn.LogSoftmax(dim=1)})
    
    return nn.Sequential(skeleton)
    
    
def save_the_checkpoint(model, train_dataset, arch, save_dir):
    torch.save({
        'arch': arch,
        'class_to_idx': train_dataset.class_to_idx,
        'classifier': model.classifier,
        }, save_dir+'model-checkpoint.pth')
    
    return save_dir+'model-checkpoint.pth' 