# this is the SERIES/CASCADE

import numpy as np
import scipy.signal as sig
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# gain incl spectral shaping filter
# spectral shaping doesn't change overall gain of resonator (relative measure, shifting up or down)


# prevent clicks in files
def apply_fade(signal):
    window = sig.windows.hann(8192)
    fade_length = window.shape[0] // 2
    signal[:fade_length] *= window[:fade_length]
    signal[-fade_length:] *= window[fade_length:]
    return signal

# function to create a second-order bandpass filter (formant resonator)
# uses iirpeak method from scipy.signal, ideal for creating RESONATORS with a sharp peak at the desired formant frequency
# bw of 100 hz for each formant
def create_formant_filter(frequency, bandwidth, fs):
    # normalize to the nyquist frequency
    w0 = frequency / (fs / 2)
    bw = bandwidth / (fs / 2)
    
    # Q factor
    Q = w0 / bw
    
    # create second-order bandpass filter
    b, a = sig.iirpeak(w0, Q)  # IIR filter for bandpass
    return b, a

def generate_impulse_train(F0, fs, duration):
    # generates an impulse train
    N = int(fs * duration)
    signal = np.zeros(N)
    phase = 0.0
    for n in range(N):
        phase += F0 / fs
        if phase >= 1.0:
            signal[n] = 1.0
            phase -= 1.0
    return signal

# creating a lowpass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# formant frequencies and bandwidths for different vowels
formants_a = [730, 1100, 2540, 3400, 4000]  # formants for vowel "a"
formants_i = [280, 2250, 2890, 3400, 4000]  # formants for vowel "i"
formants_u = [310, 870, 2250, 3200, 4000]  # formants for vowel "u"

# choose the vowel sound to synthesize
vowel = 'i'  # Choose from 'a', 'i', 'u'

if vowel == 'a':
    formants = formants_a
elif vowel == 'i':
    formants = formants_i
else:
    formants = formants_u

if __name__ == '__main__':
    # sampling rate and duration
    fs = 16000   # hz
    duration = 1 # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    F0 = 100              # fundamental frequency for the waveform
    waveform = "impulse"
    
    # generates the input excitation
    # could instead replace this with a white noise signal (in the else)
    if waveform == "impulse":
        excitation_signal = generate_impulse_train(F0, fs, duration)
    else:
        excitation_signal = np.random.randn(len(t)) # white noise excitation

    # initializes the output signal
    output_signal = excitation_signal

    """ normalizes the output signal to avoid clipping
    output_signal = output_signal / np.max(np.abs(output_signal))"""

    # make signal a little quieter
    amplitude = 0.5
    output_signal *= amplitude

    # apply the fade to our signal
    output_signal = apply_fade(output_signal)

    # Apply each formant filter in cascade
    for f in formants:
        b, a = create_formant_filter(f, 100, fs)  # get filter coefficients
        output_signal = sig.lfilter(b, a, output_signal)  # apply to signal

    # apply the final lowpass filter
    b_lp, a_lp = butter_lowpass(500, fs)
    output_signal = sig.lfilter(b_lp, a_lp, output_signal)

    # normalize the output signal to avoid silence
    output_signal /= np.max(np.abs(output_signal))

    # Save the sound file
    import soundfile as sf
    sf.write(f"{vowel}_vowel.wav", output_signal, fs)

    # choose a representative formant frequency (e.g., the first formant)
    formant_freq = formants[0]  # first formant for the vowel

    # get formant filter coefficients
    b_formant, a_formant = create_formant_filter(formant_freq, 100, fs)

    # get lowpass filter coefficients
    b_lp, a_lp = butter_lowpass(500, fs)

    # compute frequency responses
    freq_formant, h_formant = sig.freqz(b_formant, a_formant, fs=fs)
    freq_lp, h_lp = sig.freqz(b_lp, a_lp, fs=fs)

    # plot the individual transfer functions
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))

    # formant resonator amplitude response
    ax[0, 0].plot(freq_formant, 20 * np.log10(np.maximum(abs(h_formant), 1e-5)), color='blue')
    ax[0, 0].set_title("formant resonator amp/phase response")
    ax[0, 0].set_ylabel("amplitude [dB]", color='blue')
    ax[0, 0].set_xlim([100, 5000])
    ax[0, 0].set_ylim([-50, 20])
    ax[0, 0].grid(True)

    # formant resonator phase response
    ax[1, 0].plot(freq_formant, np.unwrap(np.angle(h_formant)) * 180 / np.pi, color='green')
    ax[1, 0].set_ylabel("phase [degrees]", color='green')
    ax[1, 0].set_xlabel("frequency [Hz]")
    ax[1, 0].set_xlim([100, 5000])
    ax[1, 0].set_ylim([-360, 360])
    ax[1, 0].grid(True)

    # lowpass filter amplitude response
    ax[0, 1].plot(freq_lp, 20 * np.log10(np.maximum(abs(h_lp), 1e-5)), color='red')
    ax[0, 1].set_title("lowpass filter amp/phase response")
    ax[0, 1].set_ylabel("amplitude [dB]", color='red')
    ax[0, 1].set_xlim([100, 5000])
    ax[0, 1].set_ylim([-50, 20])
    ax[0, 1].grid(True)

    # lowpass filter phase response
    ax[1, 1].plot(freq_lp, np.unwrap(np.angle(h_lp)) * 180 / np.pi, color='purple')
    ax[1, 1].set_ylabel("phase [degrees]", color='purple')
    ax[1, 1].set_xlabel("frequency [Hz]")
    ax[1, 1].set_xlim([100, 5000])
    ax[1, 1].set_ylim([-360, 360])
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # add plots for:
    # original signal
    # formant resonator output
    # low pass filter output