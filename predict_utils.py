import argparse
import torch
import json
from torchvision import transforms, models
from collections import OrderedDict
from torch import nn
from PIL import Image
import numpy as np


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', help='input image path')
    parser.add_argument('checkpoint_path', help='input checkpoint path')
    parser.add_argument('--top_k', type=int, default=5, help='top K probabilities')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='path to category names json file')
    parser.add_argument('--gpu', default=False, action='store_true', help='activate gpu if available')
    
    return parser.parse_args()


def process_image(image_path):
    image_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    try:
        im = Image.open(image_path)
    except:
        print('Could not open image file')
        raise

    width, height = im.size
    ratio = width/height
    shortest_is_width = True if width < height else False
    
    if shortest_is_width:
        im = im.resize((256, int(256/ratio)), Image.ANTIALIAS)
    else:
        im = im.resize((int(256*ratio), 256), Image.ANTIALIAS)
    
    new_size = 224
    new_width, new_height = im.size
    
    left = (new_width - new_size)/2
    top = (new_height - new_size)/2
    right = (new_width + new_size)/2
    bottom = (new_height + new_size)/2
    im = im.crop((left, top, right, bottom))
    
    im = np.array(im)
    im = im.transpose((2,0,1))
    im = image_transforms(im).float()
    im = im.transpose(0,1)
    im = im.unsqueeze(0)
    return im


def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)
    
    if chpt['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif chpt['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif chpt['arch'] == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif chpt['arch'] == 'squeezenet1_1':
        model = models.squeezenet1_1(pretrained=True)
    elif chpt['arch'] == 'densenet201':
        model = models.densenet201(pretrained=True)
    elif chpt['arch'] == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = chpt['class_to_idx']
    
    in_features = model.classifier[0].in_features
    skeleton = OrderedDict()
    
    units = chpt['units']
    
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
    
    # Put the classifier on the pretrained network
    model.classifier = nn.Sequential(skeleton)
    
    model.load_state_dict(chpt['state_dict'])
    
    return model


def load_category_names(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name