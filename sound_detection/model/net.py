"""
Defines the neural network, losss function and metrics to evaluate the model's output
"""

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BinaryMobileNetModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.base_model = models.mobilenet_v2(pretrained=True).features  # take the model without classifier

        # freeze some layers
        for num_layer, child in enumerate(self.base_model.children()):
            if num_layer < 0:  # 0 equals no freezed layers
                for param in child.parameters():
                    param.requires_grad = False

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=1), # test out_features=2
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)
        
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        return x


class BinaryResNetModel(nn.Module):
    """
    ResNet model using only feature layers. Fully connected layers are replaced.
    
    """
    def __init__(self, params):
        super().__init__()
        # take the resnet model without fully connected layers
        self.base_model = nn.Sequential(*list(models.resnet18(pretrained=True).children())[0:8])
        
        # freeze layers
        for num_layer, child in enumerate(self.base_model.children()):
            if num_layer < 0:  # 0 equals no freezed layers
                for param in child.parameters():  
                    param.requires_grad = False            
                    
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # create separate classifiers for our outputs
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=1),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)
        
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        return x

def loss_fn(outputs, labels):
    """
    Compute the binary cross entropy loss given outputs and labels.

    Args:
        outputs: (Tensor) dimension batch_size x 1 - output of the model
        labels: (Tensor) dimension batch_size, where each element is a value in [0, 1]

    Returns:
        loss (Tensor): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    loss = F.binary_cross_entropy(outputs.flatten(), labels.to(torch.float32))
    return loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all samples.

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]

    Returns: (float) accuracy in [0,1]
    """
    outputs = (outputs>0.5).astype(int)
    return accuracy_score(labels, outputs)


def f1(outputs, labels):
    """
    Compute the f1 score, given the outputs and labels for all samples.

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]

    Returns: (float) accuracy in [0,1]
    """
    outputs = (outputs>0.5).astype(int)
    return f1_score(labels, outputs, zero_division=1)

def precision(outputs, labels):
    """
    Compute the precision, given the outputs and labels for all samples.

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]

    Returns: (float) accuracy in [0,1]
    """
    outputs = (outputs>0.5).astype(int)
    return precision_score(labels, outputs, zero_division=1)

def recall(outputs, labels):
    """
    Compute the recall, given the outputs and labels for all samples.

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]

    Returns: (float) accuracy in [0,1]
    """
    outputs = (outputs>0.5).astype(int)
    return recall_score(labels, outputs, zero_division=1)


def average_precision(outputs, labels):
    """
    Compute the average prevision (AP), given the outputs and labels for all samples.
    TODO: Rises nans when batch has only 0's.

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]

    Returns: (float) accuracy in [0,1]
    """
    return average_precision_score(labels, outputs)

# maintain all metrics required in this dictionary
metrics = {
    'accuracy': accuracy,
    'f1_score': f1,
    #'avg precision': average_precision,
    'precision': precision,
    'recall': recall
    # add more metrics as needed
}