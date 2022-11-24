from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import mne
from tqdm import tqdm # For progress bar

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


# Load data
def LoadData(subject,newfreq=None):
    """Loads -epo.fif data for subject number, resamples to rewfreq

    Args:
        subject (int): Subject number to load
        newfreq (int, optional): new frequency to resample. Defaults to None.

    Returns:
        MNE.Epochs: (potentially resampled) epochs of subject
    """
    filepath = join("neuro_data","raw_epofif","Subject"+str(subject),"dataSubj"+str(subject)+"-epo.fif")
    epochs = mne.read_epochs(filepath)
    if newfreq:
        epochs = epochs.resample(newfreq)
    return epochs

def genGC3(dir, label, picks, epochs, newfreq):
    freqs = np.linspace(1,30,300) # Frequencies of interest for spectrogram
    startindex = int(np.ceil(newfreq*2.6))
    endindex = int(np.ceil(newfreq*5.6))
    width = endindex-startindex
    n_cycles = freqs/1.
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    # Experimentally determined to yield 384x384 image...
    pixr = (384*1.293)*px
    pixc = (384*1.3)*px
    for i, lab in enumerate(label):
        filepath = join(dir, lab, str(i)+".png")
        # Morlet transform
        power = mne.time_frequency.tfr_morlet(epochs[i],picks=picks, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=1)
        # Extract data and convert to image
        rgb = np.log(power.data[:,:,startindex:endindex])
        rgb = np.transpose(rgb,(1,2,0))
        rgb -= rgb.min()
        rgb *= 255.0/rgb.max()
        rgb=np.uint8(rgb)
        fig, ax = plt.subplots(1,1,figsize=(pixr, pixc))
        plt.axis('off')
        ax.imshow(rgb,extent = [0, width, 0, 300], aspect=width/300)
        plt.savefig(filepath,bbox_inches='tight',pad_inches=0)
        plt.close("all")

# We use EfficientNetV2
# https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.EfficientNet_V2_S_Weights
# Generate TFR to fit the 384x384 format ^

# Loop for generating all GC3 images, (grayscale in 3 channels)

for subject in range(10,16):
    epochs = LoadData(subject,newfreq=64)
    eventdict = {1:"left", 2:"right"}
    label = [eventdict[i] for i in epochs.events[:,2]]
    rootdir = join("TFR_plots","GC3","Subject"+str(subject))
    genGC3(rootdir,label,["T8","Cz","T7"],epochs,64)


#genMorletColored(join("TFR_plots","T8Colored","Subject"+str(subject)),"T8",epochs_resampled)