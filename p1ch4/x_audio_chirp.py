# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Audio
# ====

import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50)

# Sound can be seen as fluctuations of pressure of a medium, air for instance, at a certain location in time. There are other representations that we'll get into in a minute, but we can think about this as the _raw_, time-domain representation. In order for the human ear to appreciate sound, pressure must fluctuate with a frequency between 20 and 20000 oscillations per second (measured in Hertz, Hz). More oscillations per second will lead to a higher perceived pitch.
#
# By recording pressure fluctuations in time using a microphone and converting every pressure level at each time point into a number (e.g. a 16-bit integer), we can now represent sound as a vector of numbers. This is known as Pulse Code Modulation (PCM), where a continuous signal is both sampled in time and quantized in amplitude. If we want to make sure we hear the highest possible pitch in a recording, we'll have to record our samples at slightly more than twice the maximum audible frequency, i.e. just over 40000 times per second. It is not by chance that audio CD's have a sampling frequency of 44100 Hz. This means that a one-hour stereo (i.e. 2 channels) CD track where samples are recorded at 16-bit precision will amount to `2 * 16 * 44100 * 3600 = 5080320000 bit = 605.6 MB` if stored without compression.
#
# There are a plethora of audio formats, WAV, AIFF, MP3, AAC being the most popular, where raw audio signals are typically encoded in compressed form by leveraging on both correlation between successive samples in the time series, between the two stereo channels as well as elimination of barely audible frequencies. This can result in dramatic reduction of storage requirements (a one-hour audio file in AAC format takes less than 60 MB). In addition, audio players can decode these formats on the fly on dedicated hardware, consuming a tiny amount of power.
#
# In our data scientist role we may have to feed audio samples to our network and classify them, or generate captions, for instance. In that case, we won't work with compressed data, rather we'll have to find a way to load an audio file in some format and lay it out as an uncompressed time series in a tensor. Let's do that now.
#
# We can download a fair number of environmental sounds at the ESC-50 repository (https://github.com/karoldvl/ESC-50) in the `audio` directory. Let's get `1-100038-A-14.wav` for instance, containing the sound of a bird chirping.
#
# In order to load the sound we resort to SciPy, specifically `scipy.io.wavfile.read`, which has the nice property to return data as a NumPy array:

# +
import scipy.io.wavfile as wavfile

freq, waveform_arr = wavfile.read('../data/p1ch4/audio-chirp/1-100038-A-14.wav')
freq, waveform_arr
# -

# The `read` function returns two outputs, namely the sampling frequency and the waveform as a 16-bit integer 1D array. It's a single 1D array, which tells us that it's a mono recording - we'd have two waveforms (two channels) if the sound were stereo.
#
# We can convert the array to a tensor and we're good to go. We might also want to convert the waveform tensor to a float tensor since we're at it.

waveform = torch.from_numpy(waveform_arr).float()
waveform.shape

# In a typical dataset, we'll have more than one waveform, and possibly over more than one channel. Depending on the kind of network employed for carrying out a task, for instance a sound classification task, we would be required to lay out the tensor in one of two ways.
#
# For architectures based on filtering the 1D signal with cascades of learned filter banks, such as convolutional networks, we would need to lay out the tensor as `N x C x L`, where `N` is the number of sounds in a dataset, `C` the number of channels and `L` the number of samples in time.
#
# Conversely, for architectures that incorporate the notion of temporal sequences, just as recurrent networks we mentioned for text, data needs to be laid out as `L x N x C` - sequence length comes first. Intuitively, this is because the latter architectures take one set of `C` values at a time - the signal is not considered as a whole, but as an individual input changing in time.
#
# Although the most straightforward, this is only one of the ways to represent audio so that it is digestible by a neural network. Anther way is turning the audio signal into a _spectrogram_.
#
# Instead of representing oscillations explicitly in time, we can characterize what at frequencies those oscillations occur for short time intervals. So, for instance, if we pluck the fifth string of our (hopefully tuned) guitar and we focus on 0.1 seconds of that recording, we will see that the waveform oscillates at 440 cycles per second, plus smaller spurious oscillations at different frequencies that make up the timbre of the sound. If we move on to subsequent 0.1 second intervals, we now see that the frequency content doesn't change, but the intensity does, as the sound of our string fades. If we now decide to pluck another string, we will observe new frequencies fading in time.
#
# We could indeed build a plot having time in the X-axis, frequencies heard at that time in the Y-axis and encode intensity of those frequencies as a value at that X and Y. Or color. Ok, that starts to look like an image, right?
#
# That's correct, spectrograms are a representation of the intensity at each frequency at each point in time. It turns out that one can train convolutional neural networks built for analyzing images (we'll see about those in a couple of chapters) on sound represented as a spectrogram.
#
# Let's see how we can turn the sound we loaded earlier into a spectrogram. To do that, we need to resort to a method for converting a signal in the time domain into its frequency content. This is known as the Fourier transform, and the algorithm that allows us to compute it efficiently is the Fast Fourier Trasform (FFT), which is one of the most widespread algorithms out there. If we do that consecutively for short bursts of sound in time, we can build out spectrogram column by column.
#
# This is the general idea and we won't go into too many details here. Luckily for us SciPy has a function that gets us a shiny spectrogram given an input waveform. We import the `signal` module from SciPy,
# then provide the `spectrogram` function with the waveform and the sampling frequency that we got previously.
# The return values are all NumPy arrays, namely frequency `f_arr` (values along the Y axis), time `t_arr` (values along the X axis) and the actual spectrogra `sp_arr` as a 2D array. Turning the latter into a PyTorch tensor is trivial:
#

# +
from scipy import signal

f_arr, t_arr, sp_arr = signal.spectrogram(waveform_arr, freq)

sp_mono = torch.from_numpy(sp_arr)
sp_mono.shape
# -

#
# Dimensions are `F x T`, where `F` is frequency and `T` is time.
#
# As we mentioned earlier, stereo sound has two channels, which will lead to a two-channel spectrogram. Suppose we have two spectrograms, one for each channel. We can convert the two channels separately:
#

sp_left = sp_right = sp_arr
sp_left_t = torch.from_numpy(sp_left)
sp_right_t = torch.from_numpy(sp_right)
sp_left_t.shape, sp_right_t.shape

#
# and stack the two tensors along the first dimension to obtain a two channels image of size `C x F x T`, where `C` is the number channels:

sp_t = torch.stack((sp_left_t, sp_right_t), dim=0)
sp_t.shape

# If we want to build a dataset to use as input for a network, we will stack multiple spectrograms representing multiple sounds in a dataset along the first dimension, leading to a `N x C x F x T` tensor.
#
# Such tensor is indistinguishable from what we would build for a dataset set of images, where `F` is represents rows and `T` columns of an image. Indeed, we would tackle a sound classification problem on spectrograms with the exact same networks.
