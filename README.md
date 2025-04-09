# Few-Shot Classification of EEG with Quasi-Inductive Transfer Learning
## BSc Project - Lund University 2022.

This project employed inductive transfer learning to classify Time-Frequency transformed EEG data. The pre-trained models used, were trained on a vastly different domain, resulting in less transferrable learned weights. In the end, the project didn't yield notable results.

The InterSubTL branch contains the code used for the Inter-Subject Transfer Learning section, along with the significance testing of these.

## Repository Structure

```bash
.
├── res_DataVisualization/    # Results and visualizations of EEG data
├── res_EEGNET/               # Results from EEGNET model experiments
├── res_EffNet/               # Results from EfficientNet experiments
├── res_ResNet/               # Results from ResNet experiments
├── DataExploration.ipynb     # Code for exploring the MATLAB data files
├── DataPreparation.ipynb     # Code for converting MATLAB data to MNE FIF format
├── NN_EEGNET.ipynb           # Implementation of the EEGNET model (deprecated)
├── NN_EffNet.ipynb           # Implementation and training of EfficientNet V2
├── NN_ResNet.ipynb           # Implementation and training of ResNet18
├── Report.pdf                # The project report compiled from Overleaf
├── TFR_Gen.py                # Script for generating Time-Frequency Representation visuals
├── Visuals2.ipynb            # Code for generating MNE visualizations
├── Visuals_Res.ipynb         # Code for visualizing experimental results
```
