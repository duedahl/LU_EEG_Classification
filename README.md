# Few-Shot Classification of EEG with Quasi-Inductive Transfer Learning
## BSc Project - Lund University 2022.

This project employed inductive transfer learning to classify Time-Frequency transformed EEG data. The pre-trained models used, were trained on a vastly different domain, resulting in less transferrable learned weights. In the end, the project didn't yield notable results.

This branch contains the code used for the Inter-Subject Transfer Learning section, along with the significance testing of these.

This repository includes:
* DataExploration.ipynb - Containins the code used to investigate the matlab data files.
* DataPreparation.ipynb - Contains the code used to convert matlab data to MNE FIF format.
* NN_EEGNET.ipynb - Code for the CNN called EEGNET, not contained in project, this was abandoned in favor of deep pre-trained CNNs.
* NN_EFFNET.ipynb - Code for training of EfficientNet V2.
* NN_ResNet.ipynb - Code for training ResNet18.
* TFR_Gen.py - Code to generate all the TFR visuals and save them.
* Visuals2.ipynb - Code generating MNE visuals.
* Visuals_Res.ipynb - Code generating visuals of results.

Directories whose names start with "res_" contain results, images, loss and validation accuracy data.
