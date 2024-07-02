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

Data used in the experiments can be downloaded from [this Link](https://drive.google.com/file/d/1Q6Q6)

Pre-trained models can be downloaded from [this Link](https://drive.google.com/file/d/1Q6Q6)

To use the pre-trained models, extract the contents of the downloaded file to the `output` directory. 
The data should be extracted to the `data` directory.

## Running the experiments

To run the experiments, use the following command:

```bash
python src/train.py experiment_name=#EXP_NAME
```

where `#EXP_NAME` is either `u-zero` for the plain U-Net or `u-theo` for the physics informed U-Net with Gauss's law of magnetism.
