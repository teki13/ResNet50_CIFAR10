# ResNet50_CIFAR10

## Overview
The following project involves implementing an image classification task on the CIFAR10 dataset using ResNet50 and evaluates the impact of changes to the network through an ablation study

## Python Files

This repository consists of three python files:

- Minibatch_SGD.py: this Python script containst implementation of a minimibatch stochastic gradient decent for linear models
- ResNet50.py: This Python script utilizes ResNet to perform image classification on the CIFAR10 dataset. In addition, a mix-up technique is applied during the training phase to enhance generalization and augment the training data with additional images.
- Ablation_Study.py: In this Python script, a comparison is conducted between two optimizers, namely Adam and SGD, to evaluate their performance on an image dataset. Additionally, the script incorporates the use of mix-up.

There is also ine image included:

- mix_up.jpg: This image demonstrates how the mix up techniques aguments the images

  
## How to run the scripts

Each script is independent of one another and can be run on its one. 

To run the Minibatch_SGD.py script: 

```bash
python Minibatch_SGD.py
```

To run the ResNet50_Mix_Up.py script:

```bash
python ResNet50_Mix_Up.py
```

To run the Ablation_Study.py script:

```bash
python RAblation_Study.py
```

## Requirements

- Python
- PyTorch
- GPU recommended (CPU is also supported but may result in longer execution time)

## Output

Upon execution, both the ResNet50_Mix_Up.py and Ablation_Study.py scripts save the trained models as output. In total, there will be three models saved. Furthermore, these scripts also generate output files that include the images along with their respective classifications, as well as evaluation metrics.
