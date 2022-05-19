# Filtering an ALOS-PALSAR Dataset with Hanning-Window

This repo contains the code I wrote as part of my application process at Synspective (`Radar_engineer_test_assignment_4.pdf`). All results and some explanations can be found in the notebook `hann_window_alospalsar.py`.

## Summary

I wrote a notebook to visualize an ALOS-PALSAR L1.1 dataset, and to demonstrate the effect of applying a filter in the frequency domain. In short, the filter (i.e., the Hanning-window) decreases the noise level and---more importantly---reduces the level of the sidelobes caused by the PSF in range. This effect is shown in the figure below.

![alos_palsar_filter_demonstration](results/palsar_hanning_window_comparison.png)*Fig. 1: Demonstration of the effect of applying the Hanning window in the range frequency domain. The figure shows the area around Matsuyama airport without (left) and with filtering (right).*
