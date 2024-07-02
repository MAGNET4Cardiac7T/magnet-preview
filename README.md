# Physical knowledge improves prediction of EM Fields

This repository contains the code for reproducing the experiments in the paper "Physical knowledge improves prediction of EM Fields" submitted to the *Machine Learning Meets Differential Equations: From Theory to Applications @ ECAI 2024* Workshop


## Installation

To install the required packages, run the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Donwloading the data and pre-trained models

Data used in the experiments can be downloaded from [this Link](https://drive.google.com/drive/folders/15vHLDuAYg6aiCl3LmCaJwfsyEtDBGVlb?usp=sharing)

Pre-trained models can be downloaded from [this Link](https://drive.google.com/drive/folders/1lOkAxg123P4ui9UoFY771_MTkdtofSXj?usp=sharing)

To use the pre-trained models, extract the contents of the downloaded file to the `output` directory. 
The data should be extracted to the `data` directory.

## Running the experiments

### Train the models

To run the experiments, use the following commands. To train the plain U-Net run:

```bash
python train.py experiment_name=u-zero dataset=train model.model.physics_loss._target_=magnet.utils.zero_loss.ZeroLoss
```

to train the physics informed U-Net with Gauss's law of magnetism run:

```bash
python train.py experiment_name=u-theo dataset=train model.model.physics_loss._target_=magnet.utils.divergence_loss.DivergenceLoss
```


### Evaluate the models

To reproduce the test results from pretrained models, use the following command:

```bash
python test.py experiment_name=#EXP_NAME dataset=test
```

where `#EXP_NAME` is either `u-zero` for the plain U-Net or `u-theo` for the physics informed U-Net with Gauss's law of magnetism.