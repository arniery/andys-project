# andys-project

This script can be used to synthesize vowels, outputting plots of their frequency responses and time-domain responses, and contains fully manipulable variables for both formants and bandwidths. The script guides the user through firstly the creation of individual formant resonators, the excitation signal, and a low-pass filter, and then puts them together in a cascade model, after which gain factors are applied to each resonator for its reconfiguration as a parallel-modelled vowel. Each model's responses are plotted, along with the low-pass response, impulse signal, and formant resonator-filtered signal.

## Description

This is the final assignment for the SLP course "Speech Processing 2: Acoustic Modelling": Cascade and parallel formant synthesis. The task was to a) implement a standard cascade formant synthesizer using 5 discrete-time formant resonators connected in series and b) design and implement a formant synthesizer using 3 discrete-time formant resonators connected in parallel, the end goal being to generate vowel waveforms in the style of Klatt (1979). The aim was also to match the amplitudes of the first three formants of
the cascade synthesiser, by applying appropriate, frequency dependent, formant amplitude control (the use of spectral shaping filters was optional). For both synthesisers, a sampling frequency of 10 kHz was used.

## Getting Started

### Dependencies
This script was written on a MacBook Air M2 running Sonoma 14.6.1, and originally in a Jupyter notebook where the kernel environment was running Python 3.11.11.

To use, the following libraries need to be installed and imported:
* import numpy as np
* import scipy.signal as sig
* import soundfile as sf
* import matplotlib.pyplot as plt

### Installing / Execution

* This program is best run as a Jupyter notebook, available for download from the .ipynb file. It was not run or tested as a full .py file.
* Please download formantsandfilters.ipynb to run it this way.
* You can try downloading and running formantsandfilters.py in the environment outlined above if you like.
* Other .wav files provided are the audio outputs of the program.

## Authors

Anna DeLotto  
email: delottoa@tcd.ie

## Acknowledgments

Thanks Prof Murphy for providing the template for writing formant resonators, filters, and plots! 
