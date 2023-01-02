#!/usr/bin/env python3


#-----------------
# Imports
#-----------------
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms


class ClassificationVisualizer():

    def __init__(self, title):
        
        # Initial Parameters
        self.handles = {} # dictionary of handles per layer
        self.title = title
        self.tensor_to_PIL_image = transforms.ToPILImage()

    def draw(self, inputs, labels, outputs):

        # Setup figure
        self.figure = plt.figure(self.title)
        plt.axis('off')
        self.figure.canvas.manager.set_window_title(self.title)
        self.figure.set_size_inches(8,6)
        plt.suptitle(self.title)
        plt.legend(loc='best')

        inputs = inputs
        batch_size,_,_,_ = list(inputs.shape)

        output_probabilities = F.softmax(outputs, dim=1).tolist()
        output_probabilities_dog = [x[0] for x in output_probabilities]

        random_idxs = random.sample(list(range(batch_size)), k=5*5)

        for plot_idx, image_idx in enumerate(random_idxs, start=1):

            label = labels[image_idx]  # index da imagem
            output_probability_dog = output_probabilities_dog[image_idx]  # probabilidade da imagem ser de um cão

            is_dog = True if output_probability_dog > 0.5 else False

            # Verificar se o valor obtido corresponde ao valor do label da imagem
            success = True if (label.data.item() == 0 and is_dog) or (label.data.item() == 1 and not is_dog) else False

            image_t = inputs[image_idx,:,:,:]
            image_PIL = self.tensor_to_PIL_image(image_t)

            # Define a 5x5 subplot matrix
            ax = self.figure.add_subplot(5,5,plot_idx)
            plt.imshow(image_PIL)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            color = 'green' if success else 'red'
            title = 'dog' if is_dog else 'cat'
            title += ' ' + str(image_idx)

            ax.set_xlabel(title, color=color)

        # Draw Plot
        plt.draw()

        # Wait key
        key = plt.waitforbuttonpress(0.05)
        if not plt.fignum_exists(1):
            print('Terminating')
            exit(0)