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

# Video
# ====

import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50)

# When it comes to the shape of tensors, video data can be seen as equivalent to volumetric data, with `depth` replaced by the `time` dimension. The result is again a 5D tensor with shape `N x C x T x H x W`.
#
# There are several formats for video, especially geared towards compression by exploiting redundancies in space and time. Luckily for us, `imageio` reads video data as well. Suppose we'd like to retain 100 consecutive frames in our 512 x 512 RBG video for classifying an action using a convolutional neural network. We first create a reader instance for the video, that will allow us to get information about the video and iterate over the frames in time.
# Let's see what the meta data for the video looks like:

# +
import imageio

reader = imageio.get_reader('../data/p1ch4/video-cockatoo/cockatoo.mp4')
meta = reader.get_meta_data()
meta
# -

# We now have all the information to size the tensor that will store the video frames:

# +
n_channels = 3
n_frames = meta['nframes']
video = torch.empty(n_channels, n_frames, *meta['size'])

video.shape
# -

# Now we just iterate over the reader and set the values for all three channels into in the proper `i`-th time slice.
# This might take a few seconds to finish!

for i, frame_arr in enumerate(reader):
    frame = torch.from_numpy(frame_arr).float()
    video[:, i] = torch.transpose(frame, 0, 2)

# In the above, we iterate over individual frames and set each frame in the `C x T x H x W` video tensor, after transposing the channel. We can then obtain a batch by stacking multiple 4D tensors or pre-allocating a 5D tensor with a known batch size and filling it iteratively, clip by clip, assuming clips are trimmed to a fixed number of frames.
#
# Equating video data to volumetric data is not the only way to represent video for training purposes. This is a valid strategy if we deal with video bursts of fixed length. An alternative strategy is to resort to network architectures capable of processing long sequences and exploiting short and long-term relationships in time, just like for text or audio.
# // We'll see this kind of architectures when we take on recurrent networks.
#
# This next approach accounts for time along the batch dimension. Hence, we'll build our dataset as a 4D tensor, stacking frame by frame in the batch:
#

# +
time_video = torch.empty(n_frames, n_channels, *meta['size'])

for i, frame in enumerate(reader):
    frame = torch.from_numpy(frame).float()
    time_video[i] = torch.transpose(frame, 0, 2)

time_video.shape
