# andys-project
this is the final assignment for the SLP course "speech processing 2: acoustic modelling": cascade and parallel formant synthesis. the task was to:
- implement a standard cascade formant synthesizer using 5 discrete-time formant resonators connected in series
  and
- design and implement a formant synthesizer using 3 discrete-time formant resonators connected in parallel
the end goal being to generate vowel waveforms in the style of klatt (1979).

the files within are: 
1. andys_template.py; written by dr. andrew murphy, contains methods for signal generators, filters (formant resonator, first-order low pass filter, differentiator),
a frequency response plotter, and a main script. parts of these methods are used in the scripts "cascade synth.py" and "parallel synth.py".
2. bandpass_bandstop.py
3. cascade synth.py
4. parallel synth.py
5. proj notes (markdown)
  
