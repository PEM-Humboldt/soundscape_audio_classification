# Multi label transferlearning for soundscapes 
 
This repository compiles scripts to tag multiple soundmarks in audio recordings using
transfer learning. For training, the model receives spectrograms and multiple labels 
with presence and absence of sounds. This script uses torchvision pre-trained models on 
ImageNet, allowing have decent classification performance with few samples.

The main structure of the project was adatped from the Stanford CS230 course 
[repository](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision).

## Getting Started

### Prerequisites
All scripts have been tested in Python (v3.7.11), using torch (v1.9.0) and torchvision (v0.10.0).
Other dependencies are specified in the file `requirements.txt`. 

### Installing and Running

To train the model, run on the console:
```
python train.py --data_dir <location of data directory> --model_dir experiments/base_model
```

- The file *config.json* in the `experiments` directory has all the configuration options required for training.
- The example dataset is splitted into train and validation dataset and tree main 
columns: file_path, insects, birds. The file_path has the full path where the spectrogram
is located (*.jpg). The columns insects and birds have a binary value, 0 for absence and 1
for presence.

### Adapt to new soundmarks

You will need at least 2 datasets (train and validation), usually in csv format 
with the names `train_dsname.csv` and `val_dsname.csv`. Then:

    1. Format the dataset and create a custom dataloader class. `data_loader.py`
    2. Adapt the torchvision pre-trained model to have multiple outputs. `net.py`
    3. Modify loss function and metrics to include more outputs. `net.py`.
    4. Finally, run training script:
    
```
python train.py --data_dir <location of data directory> --model_dir experiments/base_model
```
To make further adaptation, read the CS230 tutorial to understand the structure of the project.

## TODO

- Implement configuration for: input_size, frequency for validation, frequency for checkpoints.
- Facilitate choosing between dataset and models from the configuration file.
- Add a simple dataset to test the implementation.


## Authors

Juan Sebastián Ulloa compiled this repository based on previous work (see references).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References

These scripts were developed based on:
- Most of the ideas for multilabel image classification using transfer learning where based on [this example](https://learnopencv.com/multi-label-image-classification-with-pytorch/) by Dmitry Retinskiy.
- Part of the transfer learning is based on this [pytorch tutorial](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#finetuning-torchvision-models) by Nathan Inkawhich.
- Pytorch main structure from Stanford [CS230 course](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision).
