"""Defines the neural network, losss function and metrics"""
import warnings
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiOutputMobileNetModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        n_insect_classes = params.n_insect_classes
        n_bird_classes = params.n_bird_classes
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before classifier

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.insect = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_insect_classes)
        )
        self.bird = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_bird_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'insect': self.insect(x),
            'bird': self.bird(x)
        }



def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    insect_loss = F.cross_entropy(outputs['insect'], labels['insect_label'])
    bird_loss = F.cross_entropy(outputs['bird'], labels['bird_label'])
    loss = insect_loss + bird_loss
    return loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    _, predicted_insect = outputs['insect'].cpu().max(1)
    gt_insect = labels['insect_label'].cpu()

    _, predicted_bird = outputs['bird'].cpu().max(1)
    gt_bird = labels['bird_label'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_insect = accuracy_score(y_true=gt_insect.numpy(), y_pred=predicted_insect.numpy())
        accuracy_bird = accuracy_score(y_true=gt_bird.numpy(), y_pred=predicted_bird.numpy())

    return accuracy_insect, accuracy_bird

def accuracy_bird(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    _, predicted_bird = outputs['bird'].cpu().max(1)
    gt_bird = labels['bird_label'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_bird = accuracy_score(y_true=gt_bird.numpy(), y_pred=predicted_bird.numpy())

    return accuracy_bird

def accuracy_insect(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    _, predicted_insect = outputs['insect'].cpu().max(1)
    gt_insect = labels['insect_label'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_insect = accuracy_score(y_true=gt_insect.numpy(), 
                                         y_pred=predicted_insect.numpy())

    return accuracy_insect


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'accuracy_bird': accuracy_bird,
    'accuracy_insect': accuracy_insect
    # could add more metrics
}
