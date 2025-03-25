# big vowel synth
# this is the cascade

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


# prevent clicks in files
def apply_fade(signal):
    window = sig.windows.hann(8192)
    fade_length = window.shape[0] // 2
    signal[:fade_length] *= window[:fade_length]
    signal[-fade_length:] *= window[fade_length:]
    return signal

# function to create a second-order bandpass filter (formant resonator)
# uses iirpeak method from scipy.signal, ideal for creating 
# resonators with a sharp peak at the desired formant frequency
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

def generate_impulse_train(F0, sample_rate, duration):
    """Generates an impulse train."""
    N = int(sample_rate * duration)
    signal = np.zeros(N)
    phase = 0.0
    for n in range(N):
        phase += F0 / sample_rate
        if phase >= 1.0:
            signal[n] = 1.0
            phase -= 1.0
    return signal

# formant frequencies and bandwidths for different vowels
formants_a = [730, 1090, 2440, 3400, 4000]  # formants for vowel "a"
formants_i = [270, 2290, 3010, 3400, 4000]  # formants for vowel "i"
formants_u = [325, 500, 2400, 3200, 4000]  # formants for vowel "u"

# Choose the vowel sound to synthesize
vowel = 'a'  # Choose from 'a', 'i', 'u'

if vowel == 'a':
    formants = formants_a
elif vowel == 'i':
    formants = formants_i
else:
    formants = formants_u


# np.random.randn(len(t)) = white noise excitation

if __name__ == '__main__':
    # sampling rate and duration
    fs = 16000   # hz
    duration = 1 # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    F0 = 100              # fundamental frequency for the waveform
    
    # generates the input excitation
    # could instead replace this with impulse signal from andy's code
    excitation_signal = generate_impulse_train(F0, fs, duration)

    # initializes the output signal
    output_signal = excitation_signal


    # Apply Filters (each function returns the time–domain output and its coefficients)
    """ formant_out, formant_coeffs = formant_resonator(signal, formant_frequency=500, sample_rate=sample_rate, bandwidth=100)
    low_pass_out, lp_coeffs = low_pass_filter(signal, cutoff_frequency=500, sample_rate=sample_rate)
    diff_out, diff_coeffs = differentiator(signal, sample_rate=sample_rate)"""

    # applies each formant resonator (filter) in series
    for f in formants:
    # assume a bandwidth of 100 hz for each formant
    # filter the excitation signal with the corresponding bandpass filter
        b, a = create_formant_filter(f, 100, fs)
        output_signal = sig.lfilter(b, a, output_signal)

    # normalizes the output signal to avoid clipping
    output_signal = output_signal / np.max(np.abs(output_signal))

    # make signal a little quieter
    amplitude = 0.5
    output_signal *= amplitude

    # apply the fade to our signal
    output_signal = apply_fade(output_signal)

    # generate wav files !!!
    # writes to a .wav file or plays it directly
    import soundfile as sf
    sf.write(f"{vowel}_vowel.wav", output_signal, fs)

    # plots the waveform of the synthesized vowel sound
    plt.figure(figsize=(10, 4))
    plt.plot(t[:1000], output_signal[:1000])  # Plot a small part of the signal
    plt.title(f"Synthesized Vowel {vowel.upper()}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
