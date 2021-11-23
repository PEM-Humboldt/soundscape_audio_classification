#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference - Use a pretrained model to make inference on new data
"""
import argparse
import logging
import os

import torch
import numpy as np
import pandas as pd
import utils
import model.net as net
from maad import sound, util
from skimage import io
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def load_data():
    """
    Transform an audio file into multiple spectrogram images according to simple
    parameters: target sampling frequency (target_fs), window length (wl), and
    spectrogram parameters (number of samples and overlap)
    
    TODO: This should be done as transformations on the dataloader.py script
    
    """
    fname_audio = '/Volumes/Audiolib/ML_datasets/Putumayo_2018/test_audio_putumayo/audio/ALFE/VEG01_20190903_170000.wav'
    target_fs = 24000  # target fs of project
    wl = 10  # Window length for formated rois
    spectro_nperseg = 512
    spectro_noverlap = 0
    
    # Load data
    s, fs = sound.load(fname_audio)
    s = sound.resample(s, fs, target_fs)
    s = sound.normalize(s, max_db=0)
    Sxx, tn, fn, ext = sound.spectrogram(s, fs=target_fs, nperseg=spectro_nperseg,
                                         noverlap=spectro_noverlap)
    Sxx = util.power2dB(Sxx, db_range=60, db_gain=20)
    Sxx = np.flipud(Sxx)
    img_shape = [len(fn), np.floor(target_fs * wl / (spectro_nperseg - spectro_noverlap)).astype(int)]
    # pad
    pad_width = (img_shape[1]-(Sxx.shape[1]%img_shape[1])).astype(int)
    Sxx = np.pad(Sxx, pad_width=[(0,0), (0, pad_width)])
    
    # reshape
    n_img = (Sxx.shape[1]/img_shape[1]).astype(int)
    img_list = np.reshape(Sxx, (img_shape[0], img_shape[1], n_img))
    
    # save
    for idx in range(n_img):
        path_save = '/Users/jsulloa/Downloads/tmp/'
        fname_save = path_save+'img_'+str(idx).zfill(3)
        io.imsave(fname_save+'.jpg', Sxx[:,idx*img_shape[1]:(idx+1)*img_shape[1]])#%%  
    
    return


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

    # fetch data
    dataloaders = data_loader.fetch_dataloader(['infer'], args.data_dir, params)
    infer_dl = dataloaders['infer']

    logging.info("- done.")

    # Define the model
    model = net.BinaryMobileNetModel(params).cuda() if params.cuda else net.BinaryMobileNetModel(params)

    metrics = net.metrics

    logging.info("Starting inference on new data")

    # Reload weights from the saved file
    logging.info("Loading model ...")
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)
    logging.info("- done")
    
    # Make inference
    predictions = inference(model, infer_dl, params)
    predictions.sort_values('fname').to_csv(args.data_dir+'predictions.csv', index=False)
    
