from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import mne
from tqdm import tqdm # For progress bar

# Load data
subject = 15
filepath = join("neuro_data","raw_epofif","Subject"+str(subject),"dataSubj"+str(subject)+"-epo.fif")
epochs = mne.read_epochs(filepath)

# Downsample data, smallest power of 2 satisfying nyquist for frequencies<=30
newfreq = 64
epochs_resampled = epochs.resample(newfreq)

# We use EfficientNetV2
# https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.EfficientNet_V2_S_Weights
# Generate TFR to fit the 384x384 format ^

def genMorletColored(dir, chan, epochs):
    """Generates colored TFR for all samples in epochs

    Args:
        dir (str): location for images
        chan (str): channel to generate TFR for
        epochs (MNE.Epochs): MNE epochs object containing all samples
    """
    freqs = np.linspace(1,30,300) # Frequencies of interest for spectrogram
    n_cycles = freqs/1.
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    # Experimentally determined to yield 384x384 image...
    pixr = (384*1.293)*px
    pixc = (384*1.3)*px
    for i in range(len(epochs)):
        power = mne.time_frequency.tfr_morlet(epochs[i], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=1)
        power.data = np.log(power.data)
        fig, ax = plt.subplots(1,1,figsize=(pixr, pixc))
        power.plot([chan], baseline=(-2,0), tmin=0.6, tmax=3.6, colorbar=False, axes=ax, show=False, verbose=False)
        plt.axis('off')
        plt.savefig(join(dir,str(i))+".png",bbox_inches='tight',pad_inches=0)
        plt.close("all")

genMorletColored(join("TFR_plots","T8Colored","Subject"+str(subject)),"T8",epochs_resampled)