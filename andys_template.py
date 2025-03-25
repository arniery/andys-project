# andys template

import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import sawtooth

# ==============================
# Signal Generators
# ==============================
# source signals, add more if needed

# finding out what periodicity of wf and giving impulse per period
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

def generate_sawtooth(F0, sample_rate, duration):
    # Create a time vector with the proper number of samples.
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate the sawtooth waveform. The function returns values in the range [-1, 1].
    signal = sawtooth(2 * np.pi * F0 * t)
    return signal

# ==============================
# Filters
# ==============================

# --- Formant Resonator ---
# y[n] = A*x[n-1] + B*y[n-1] + C*y[n-2]
# with A = 1 – B – C, B = 2 exp(–π F_B T_s) cos(2π F T_s), C = exp(–2π F_B T_s)
# building block for resonators
def formant_resonator(signal, formant_frequency, sample_rate, bandwidth):
    T_s = 1 / sample_rate
    B_coef = 2 * np.exp(-np.pi * bandwidth * T_s) * np.cos(2 * np.pi * formant_frequency * T_s)
    C_coef = -1*np.exp(-2 * np.pi * bandwidth * T_s)
    A_coef = 1 - B_coef - C_coef

    y = np.zeros_like(signal)
    # For n = 0, no previous input; set y[0] = 0.
    y[0] = 0
    # For n = 1, use x[0]
    if len(signal) > 1:
        y[1] = A_coef * signal[0] + B_coef * y[0]
    # For n >= 2, follow the difference equation.
    for n in range(2, len(signal)):
        y[n] = A_coef * signal[n-1] + B_coef * y[n-1] + C_coef * y[n-2]
    # Return both the time response and the coefficients (for TF plotting)
    return y, (A_coef, B_coef, C_coef)

# --- First-Order Low-Pass Filter ---
# y[n] = A*x[n] + B*y[n-1], with A = 1 – B, B = exp(–2π f_c T_s)
# from discrete time pdf formulas
def low_pass_filter(signal, cutoff_frequency, sample_rate):
    T_s = 1 / sample_rate
    B_coef = np.exp(-2 * np.pi * cutoff_frequency * T_s)
    A_coef = 1 - B_coef

    y = np.zeros_like(signal)
    y[0] = A_coef * signal[0]
    for n in range(1, len(signal)):
        y[n] = A_coef * signal[n] + B_coef * y[n-1]
    return y, (A_coef, B_coef, 0)  # 0 for C since it is first order

# --- Differentiator ---
# y[n] = (x[n] – x[n-1]) / T_s
def differentiator(signal, sample_rate):
    T_s = 1 / sample_rate
    y = np.zeros_like(signal)
    # For first sample, approximate as x[0]/T_s
    y[0] = signal[0] / T_s
    for n in range(1, len(signal)):
        y[n] = 1/T_s*signal[n] - 1/T_s*signal[n-1]
    # Transfer function: H(z)= (1 - z^(-1))/T_s, so numerator = [1, -1] and denominator = [T_s]
    return y, (1, -1, 0)

# ==============================
# Frequency Response Plotter
# ==============================
def plot_formant_response(A, B, C, sample_rate, label):
    T_s = 1 / sample_rate
    f = np.linspace(0, sample_rate/2, 1024)
    theta = 2 * np.pi * f * T_s  # note: T_s cancels, but we keep for clarity
    # For the formant resonator, the transfer function is:
    # H(f) = (A * exp(-j*theta)) / (1 - B exp(-j*theta) - C exp(-j*2*theta))
    H = (A * np.exp(-1j * theta)) / (1 - B * np.exp(-1j * theta) - C * np.exp(-1j * 2 * theta))
    amplitude = 20 * np.log10(np.abs(H))
    phase = np.unwrap(np.angle(H))

    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(f, amplitude, 'b')
    plt.title(f"{label} - Amplitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(f, phase, 'r')
    plt.title(f"{label} - Phase Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_lowpass_response(A, B, sample_rate, label):
    T_s = 1 / sample_rate
    f = np.linspace(0, sample_rate/2, 1024)
    theta = 2 * np.pi * f * T_s
    # Low-pass: H(f) = A / (1 - B exp(-j*theta))
    H = A / (1 - B * np.exp(-1j * theta))
    amplitude = 20 * np.log10(np.abs(H))
    phase = np.unwrap(np.angle(H))

    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(f, amplitude, 'b')
    plt.title(f"{label} - Amplitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(f, phase, 'r')
    plt.title(f"{label} - Phase Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_diff_response(T_s, sample_rate, label):
    f = np.linspace(0, sample_rate/2, 1024)
    theta = 2 * np.pi * f * T_s
    # Differentiator: H(f) = (1 - exp(-j*theta)) / T_s
    H = (1 - np.exp(-1j * theta)) / T_s
    amplitude = 20 * np.log10(np.abs(H))
    phase = np.unwrap(np.angle(H))

    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(f, amplitude, 'b')
    plt.title(f"{label} - Amplitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(f, phase, 'r')
    plt.title(f"{label} - Phase Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================
# Main Script
# ==============================
if __name__ == '__main__':
    sample_rate = 16000   # Hz
    duration = 0.05       # seconds
    F0 = 100              # Fundamental frequency for the waveform
    waveform = "sawtooth"  # Options: "impulse" or "sawtooth"

    # Generate input signal
    if waveform == "impulse":
        signal = generate_impulse_train(F0, sample_rate, duration)
    else:
        signal = generate_sawtooth(F0, sample_rate, duration)

    # Apply Filters (each function returns the time–domain output and its coefficients)
    formant_out, formant_coeffs = formant_resonator(signal, formant_frequency=500, sample_rate=sample_rate, bandwidth=100)
    low_pass_out, lp_coeffs = low_pass_filter(signal, cutoff_frequency=500, sample_rate=sample_rate)
    diff_out, diff_coeffs = differentiator(signal, sample_rate=sample_rate)

    # layer f0s in parallel or cascade (arrays?) 

    # generate wav files !!!

    # Plot Frequency Responses
    A_f, B_f, C_f = formant_coeffs
    plot_formant_response(A_f, B_f, C_f, sample_rate, "Formant Resonator")

    A_lp, B_lp, _ = lp_coeffs
    plot_lowpass_response(A_lp, B_lp, sample_rate, "Low-Pass Filter")

    T_s = 1 / sample_rate
    plot_diff_response(T_s, sample_rate, "Differentiator")

    # Plot Time-Domain Outputs
    t = np.arange(len(signal)) / sample_rate
    outputs = [
        (signal, "Original Signal"),
        (formant_out, "Formant Resonator Output"),
        (low_pass_out, "Low-Pass Filter Output"),
        (diff_out, "Differentiator Output")
    ]
    plt.figure(figsize=(12, 14))
    for i, (out, title_str) in enumerate(outputs):
        plt.subplot(len(outputs), 1, i+1)
        plt.plot(t, out, 'k')
        plt.title(title_str, fontsize=12)
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.xlabel("Time (s)", loc='right')
    plt.tight_layout()
    plt.show()