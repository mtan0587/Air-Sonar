# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ELEC3305 (University of Sydney) group lab project implementing a **real-time acoustic sonar system** using a laptop's built-in speaker and microphone. The system transmits chirp pulses, records echoes, and estimates object distances via time-of-flight. The codebase is split between a reusable Python module (`rtsonar.py`) and two Jupyter notebooks for lab exercises and real-time operation.

## Environment Setup

Dependencies live in `.venv/`. Activate before running anything:

```powershell
.\.venv\Scripts\Activate.ps1
```

Key libraries: `numpy`, `scipy`, `matplotlib`, `bokeh`, `pyaudio`.

To launch notebooks:

```powershell
jupyter notebook
```

## Project Structure

| File | Role |
|------|------|
| `rtsonar.py` | Core real-time sonar module — threading, audio I/O, signal processing, Bokeh visualization |
| `TimeDomain-Sonar-New.ipynb` | Main lab notebook — chirp theory, Part 1 characterization, Part 2 system design with 5 student-implemented functions |
| `TimeDomain-RealTime-Sonar.ipynb` | Real-time runner — imports `rtsonar`, wires in the 5 student functions, exposes parameter tuning |

## Signal Processing Pipeline

```
genChirpPulse()  →  Hanning window  →  genPulseTrain()
                                              ↓
                              play_audio() [Thread]  →  speaker
                                                            ↓
                                              record_audio() [Thread]  ←  mic
                                                            ↓
                              signal_process() [Thread]:
                                  crossCorr() → matched filter (FFT convolution)
                                  findDelay() → synchronize to feedthrough pulse
                                  overlap-and-add rolling window
                                  interpolate to Nplot samples
                                  dist2time() → cm to time index
                                              ↓
                              image_update() [Thread]  →  Bokeh waterfall plot
```

## The 5 Student Functions (Part 2 / Real-Time)

Students must implement these functions in `TimeDomain-Sonar-New.ipynb` (Part 2) and copy them into `TimeDomain-RealTime-Sonar.ipynb`:

1. **`genChirpPulse(Npulse, f0, f1, fs)`** — Analytic chirp: `exp(j*2π*φ(t))` where `φ(t) = f₀t + (f₁−f₀)/(2T) · t²`
2. **`genPulseTrain(pulse, Nrep, Nseg)`** — Repeat pulse `Nrep` times with `Nseg` spacing (zero-pad between repetitions)
3. **`crossCorr(rcv, pulse_a)`** — Matched filter: `signal.fftconvolve(rcv, conj(pulse_a[::-1]))`
4. **`findDelay(Xrcv, Nseg)`** — Locate feedthrough (direct path) peak to synchronize receiver timing
5. **`dist2time(dist, temperature)`** — Convert distance (cm) → time: `v = 331.5*sqrt(1 + T/273.15)` m/s; `t = 2*dist/(100*v)`

## Key Signal Processing Concepts

- **Pulse compression**: Wider bandwidth chirp → narrower matched-filter peak → better range resolution
- **Sidelobes**: Hanning window on chirp reduces autocorrelation sidelobes at cost of slightly wider main lobe
- **Overlap-and-add**: `signal_process()` maintains a 3×Nseg rolling buffer so matched filtering spans chunk boundaries
- **Normalization**: 97th-percentile normalization (not max) prevents saturation from the large feedthrough pulse; gamma correction (power 1/1.8) enhances weak echoes
- **Sound speed**: `v_sound` depends on ambient temperature — `temperature` parameter in °C must be set correctly for accurate range

## Real-Time Parameters (TimeDomain-RealTime-Sonar.ipynb)

```python
fs = 48000        # sampling frequency (Hz)
f0, f1 = 6000, 12000  # chirp sweep range (Hz)
Npulse = 360      # pulse length (samples) — longer = more energy, worse range res
Nseg   = 4800     # inter-pulse spacing (samples) — sets max unambiguous range
Nrep   = 100      # number of rows in waterfall display
Nplot  = 200      # horizontal pixels (range bins) in display
maxdist = 200     # maximum displayed range (cm)
temperature = 20  # ambient temperature (°C)
```

Stop the real-time system cleanly with:

```python
stop_flag.set()
```

## Hardware Setup Notes

- Disable all audio enhancements (ambient noise reduction, auto-gain) in OS audio settings
- Set speaker volume to 70–80% to avoid distortion
- Position speaker and microphone as close together as possible
- Test in a quiet environment — room echoes add clutter

## Threading Architecture (rtsonar.py)

Five daemon threads communicate via `queue.Queue` FIFOs:

| Thread | Function | Queue In | Queue Out |
|--------|----------|----------|-----------|
| Data queuer | `put_data` | — | `Qout` |
| Playback | `play_audio` | `Qout` | — |
| Recording | `record_audio` | — | `Qin` |
| Signal processing | `signal_process` | `Qin` | `Qdata` |
| Visualization | `image_update` | `Qdata` | — |

All threads check `stop_flag` (a `threading.Event`) for graceful shutdown.
