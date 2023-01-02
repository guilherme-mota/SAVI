#!/usr/bin/env python3


#-----------------
# Imports
#-----------------
import torch
from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(self, image_filenames):

        super().__init__()

        self.image_filenames = image_filenames
        self.num_images = len(self.image_filenames)
        self.labels = []

        for image_filename in self.image_filenames:
            self.labels.append(self.getClassFromFilename(image_filename))
        
        # Create a set of transformations
        self.transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

    def __getitem__(self, index):  # return a specific element x,y given the index, of the dataset

        # Load the image
        image_pil = Image.open(self.image_filenames[index])

        image_t = self.transforms(image_pil)

        return image_t, self.labels[index]

    def __len__(self):  # return the length of the dataset
        return self.num_images

    def getClassFromFilename(self, filename):

        parts = filename.split('/')
        part = parts[-1]

        parts = part.split('.')

        class_name = parts[0]
        
        if class_name == 'dog':
            label = 0  # use the idx of the outputs vector where the 1 should be
        elif class_name == 'cat':
            label = 1
        else:
            raise ValueError('Unknown class')

        return label