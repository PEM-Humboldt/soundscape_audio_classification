#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference - Use a pretrained model to make inference on new data
"""
import argparse
import logging
import os

import torch
import pandas as pd
import utils
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


# Make inference
def inference(model, dataloader, params):
    output = pd.DataFrame()
    # compute model output
    for i, (data_batch, labels_batch) in enumerate(dataloader):
        # forward pass
        output_batch = model(data_batch)
        # organize data
        aux = pd.DataFrame({'fname': labels_batch,
                            'output': output_batch.detach().numpy()[:,0]})
        output = output.append(aux, ignore_index=True)
        
    return output

if __name__ == '__main__':
    """
        Make inference using a trained model on new data.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'inference.log'))

    # Create the input data pipeline
    logging.info("Loading data...")
    dataloaders = data_loader.fetch_dataloader(['infer'], args.data_dir, params)
    infer_dl = dataloaders['infer']
    logging.info("- done.")

    # Define the model and optimizer
    if params.model_name == 'mobilenet':
        model = net.BinaryMobileNetModel(params).cuda() if params.cuda else net.BinaryMobileNetModel(params)
    elif params.model_name =='resnet':
        model = net.BinaryResNetModel(params).cuda() if params.cuda else net.BinaryResNetModel(params)
    else:
        raise NameError('Model name not found')
    
    logging.info("Starting inference on new data")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)
    logging.info('Model: {}'.format(model.__class__.__name__))
    
    # Make inference
    predictions = inference(model, infer_dl, params)
    predictions['output'] = predictions['output'].round(3)
    predictions.sort_values('fname').to_csv(args.data_dir+'predictions.csv', index=False)
    
