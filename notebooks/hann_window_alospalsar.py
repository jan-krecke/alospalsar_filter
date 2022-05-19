# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.signal import windows

import utils

# %% [markdown]
# # Applying a Hann-Window to ALOS-PALSAR L1.1 Data

# %% [markdown]
# The purpose of this notebook is to visualize a single-look complex (SLC) SAR dataset collected by ALOS PALSAR. In addition, the dataset will be filtered in the frequency domain using a Hann-window.
#
# The PALSAR product used for this analysis is *ALPSRP139670660*. It can be downloaded using the [ASF Vertex](https://search.asf.alaska.edu/#/) tool.

# %% [markdown]
# ## 1. Read the complex SLC data

# %% [markdown]
# First, the dataset has to be read into a Numpy-array. To do that, I use a code snipped provided to me by Synspective (imported here from `utils.py` as `read_alospalsar_image`). The path specified in `product_path` has to be changed according to where the product is.

# %%
product_path = Path(
    "/home/debian/data_share/alos_palsar/ALPSRP139670660-H1.1__A"
)  # noqa
fname = "IMG-HH-ALPSRP139670660-H1.1__A"

slc = utils.read_alospalsar_image(product_path / fname)
n_lines = slc.shape[0]
n_pixels = slc.shape[1]

# %% [markdown]
# # 2. Create an overview image

# %% [markdown]
# After reading the dataset, we visualize it using its power/ intensity on a logarithmic scale.

# %%
slc_intensity_dB = 20 * np.log10(np.abs(slc))

# %% [markdown]
# Visualize intensity image.

# %%
scale = 10

fig, ax = plt.subplots(1, 1, figsize=(scale, scale * n_lines / n_pixels))

img = ax.imshow(
    slc_intensity_dB,
    origin="lower",
    vmin=80,
    vmax=120,
    cmap="Greys_r",
    aspect="auto",
)

ax.set_ylabel("Lines")
ax.set_xlabel("Pixels")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(img, cax=cax, label="Intensity in dB")

# %% [markdown]
# # 3. Implement frequency based windowing

# %% [markdown]
# ## 3.1 For each line, compute the Fourier transform with a size of 16384

# %% [markdown]
# Numpy's fft function offers the possibility to compute the FFT for all lines at once (`axis=1`)

# %%
n_fft = 16384
slc_fft = np.fft.fftshift(np.fft.fft(slc, n=n_fft, axis=1), axes=1)

# %% [markdown]
# Visualize the line spectra

# %%
fig, ax = plt.subplots(1, 1, figsize=(scale, scale * n_lines / n_fft))

slc_fft_dB = 20 * np.log10(np.abs(slc_fft))
img = ax.imshow(
    slc_fft_dB,
    origin="lower",
    cmap="Greys_r",
    vmin=80,
    vmax=150,
    aspect="auto",
    extent=[-0.5,0.5,0,n_lines],
)

ax.set_ylabel("Lines")
ax.set_xlabel("Normalized Spatial Frequency")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(img, cax=cax, label="Spectrum Intensity in dB")

# %% [markdown]
# Visualize spectrum of single line

# %%
fig, ax = plt.subplots(1, 1, figsize=(scale * 1, scale * 1))

index_sample_line = 1000

freq = np.linspace(-0.5, 0.5, n_fft)
ax.plot(freq, slc_fft_dB[index_sample_line,:])
ax.grid(True)

ax.set_xlabel("Normalized Frequency")
ax.set_ylabel("Intensity in dB")

plt.xticks(np.arange(-0.5, 0.6, 0.1));

# %% [markdown]
# ## 3.2 Apply the Hanning window to the central part of the frequency spectrum

# %% [markdown]
# Comput Hann-window using `scipy`

# %%
n_hann = int(n_fft / 2)

ind_hann_start = int(n_fft / 4)
ind_hann_end = ind_hann_start + n_hann

hann_window = np.zeros(n_fft)
hann_window[ind_hann_start:ind_hann_end] = windows.hann(n_hann)

# %% [markdown]
# Plot Hann-window

# %%
fig, ax = plt.subplots(1, 1, figsize=(scale * 1, scale * 1))

ax.plot(hann_window)
ax.grid(True)

ax.set_xlabel("Sample Index")
ax.set_ylabel("Value")

plt.xticks(np.arange(0, n_fft, 2000));

# %% [markdown]
# Apply Hann-window to the range spectra

# %%
slc_fft_filtered = slc_fft * hann_window[None, :]

# %% [markdown]
# Visualize spectra after application of Hann-window

# %%
fig, ax = plt.subplots(1, 1, figsize=(scale, scale * n_lines / n_fft))

slc_fft_filtered_dB = 20 * np.log10(np.abs(slc_fft_filtered), where=(slc_fft_filtered != 0))

img = ax.imshow(
    slc_fft_filtered_dB,
    origin="lower",
    cmap="Greys_r",
    vmin=30,
    vmax=160,
    aspect="auto",
    extent=[-0.5,0.5,0,n_lines],
)

ax.set_ylabel("Lines")
ax.set_xlabel("Normalized Frequency")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(img, cax=cax, label="Spectrum Intensity in dB")

# %% [markdown]
# Visualize single line of filtered spectrum

# %%
fig, ax = plt.subplots(1, 1, figsize=(scale * 1, scale * 1))

index_sample_line = 1000

ax.plot(slc_fft_filtered_dB[index_sample_line,:])
ax.grid(True)

ax.set_xlabel("Frequency Index")
ax.set_ylabel("Value in dB")

plt.xticks(np.arange(0, n_fft, 2000));

# %% [markdown]
# ## 3.3 Compute Inverser Fourier Transform

# %%
slc_filtered = np.fft.ifft(slc_fft_filtered)
slc_filtered = slc_filtered[:, :n_pixels]

# %% [markdown]
# Visualize filtered image

# %%
fig, ax = plt.subplots(1, 1, figsize=(scale, scale * n_lines / n_pixels))

slc_filtered_dB = 20 * np.log10(np.abs(slc_filtered))

img = ax.imshow(
    slc_filtered_dB,
    origin="lower",
    vmin=80,
    vmax=120,
    cmap="Greys_r",
    aspect="auto",
)

ax.set_ylabel("Lines")
ax.set_xlabel("Pixels")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(img, cax=cax, label="Intensity in dB")

# %% [markdown]
# # 4. Compare Original and Filtered Image above Matsuyama Airport

# %%
fig, axes = plt.subplots(1, 2, figsize=(2 * scale, scale))

line_min = 13600
line_max = 14700
pixel_min = 7500
pixel_max = 8050

extent = (pixel_min, pixel_max, line_min, line_max)

ax = axes[0]
img = ax.imshow(
    slc_intensity_dB[line_min:line_max, pixel_min:pixel_max],
    origin="lower",
    vmin=80,
    vmax=120,
    cmap="Greys_r",
    aspect="auto",
    extent=extent,
)

ax.set_ylabel("Lines")
ax.set_xlabel("Pixels")
ax.set_title("Original")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(img, cax=cax, label="Intensity in dB")

ax = axes[1]
img = ax.imshow(
    slc_filtered_dB[line_min:line_max, pixel_min:pixel_max],
    origin="lower",
    vmin=80,
    vmax=120,
    cmap="Greys_r",
    aspect="auto",
    extent=extent,
)

ax.set_ylabel("Lines")
ax.set_xlabel("Pixels")
ax.set_title("Bandlimited with Hann-Window")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(img, cax=cax, label="Intensity in dB")

# %%