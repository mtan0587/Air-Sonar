# Air Sonar

**Student:** Matthew Kimi Tanoyo  
**Course:** ELEC3305 — Digital Signal Processing, University of Sydney  
**Assessment:** Group Lab Project

---

## What is Air Sonar?

Air Sonar is an active ranging system that estimates the distance to objects by transmitting an acoustic pulse through air, recording the returning echo at a microphone, and computing the time-of-flight. It is the acoustic analogue of radar: the time delay $\tau$ between the transmitted pulse and its echo, combined with the speed of sound $v_s$, gives the range

$$d = \frac{v_s \cdot \tau}{2}$$

where the factor of 2 accounts for the two-way travel path. The speed of sound in air is temperature-dependent and is modelled here as

$$v_s = 331.5\sqrt{1 + \frac{T}{273.15}} \quad \text{(m/s)}$$

where $T$ is the ambient temperature in °C.

Rather than a simple tone burst, this system uses a **Linear Frequency Modulated (LFM) chirp** pulse and a **matched filter** receiver. This combination — known as **pulse compression** — yields a range resolution proportional to the signal bandwidth $B = f_1 - f_0$ rather than the pulse duration, allowing a long (high-energy) pulse to achieve the resolution of a much shorter one. The theoretical range resolution is approximately

$$\Delta d = \frac{v_s}{2B}$$

---

## Repository Files

| File | Role |
|------|------|
| `rtsonar.py` | Core real-time sonar engine — audio I/O threads, signal processing, live waterfall display |
| `TimeDomain-RealTime-Sonar.ipynb` | Notebook that wires the five student-implemented DSP functions into the real-time engine and exposes parameter tuning |
| `TimeDomain-Sonar-New.ipynb` | Lab notebook — theory, chirp characterisation (Part 1), offline sonar design and implementation (Part 2) |

---

## Signal Processing Pipeline

```
genChirpPulse(Npulse, f0, f1, fs)
      │  Analytic LFM chirp: x[n] = exp(j·2π·φ[n])
      │  φ[n] = f₀·n/fs + (k/2)·(n/fs)²,  k = (f₁−f₀)/T
      │  Windowed by Hann: w[n] = 0.5(1 − cos(2πn/Npulse))
      ↓
genPulseTrain(pulse, Nrep, Nseg)
      │  Repeat Nrep times with Nseg-sample zero-padded gaps
      ↓
play_audio() ─────────────────────────────→ speaker (sd.OutputStream)
                                                   │  (propagates through air)
record_audio() ←──────────────────────────── microphone (sd.InputStream)
      │  Streaming chunks via queue.Queue
      ↓
signal_process()
      │  crossCorr(rcv, pulse_a)   ← matched filter via FFT convolution
      │  Overlap-and-add across chunk boundaries
      │  findDelay(Xrcv, Nseg)     ← synchronise to feedthrough peak
      │  Interpolate segment to Nplot bins
      │  dist2time(maxdist, T)     ← cm → sample index cutoff
      ↓
image_update()
      │  97th-percentile normalisation (suppress feedthrough saturation)
      │  Gamma correction (power 1/1.8, enhances dim echoes)
      │  cm.jet colormap → RGBA → PIL PNG → ipywidgets.Image widget
      ↓
Live waterfall display (900×500 px, updates at fs/Nseg Hz)
```

---

## The Five Student-Implemented Functions

These functions are developed in `TimeDomain-Sonar-New.ipynb` (Part 2) and copied into `TimeDomain-RealTime-Sonar.ipynb`:

### 1. `genChirpPulse(Npulse, f0, f1, fs)`
Generates the **analytic (complex) LFM chirp**:

$$x[n] = e^{j2\pi\varphi[n]}, \quad \varphi[n] = f_0 \frac{n}{f_s} + \frac{k}{2}\left(\frac{n}{f_s}\right)^2, \quad k = \frac{f_1 - f_0}{T}$$

The analytic representation (positive-frequency only) is used throughout the matched filter chain.  
**Reference:** Oppenheim & Schafer, Ch. 2 — *Discrete-Time Signals and Systems*; analytic signal via the Hilbert transform, Ch. 12.

### 2. `genPulseTrain(pulse, Nrep, Nseg)`
Constructs a periodic pulse train by tiling the windowed chirp with `Nseg`-sample zero-padded intervals. The inter-pulse period $T_{seg} = N_{seg}/f_s$ determines the maximum unambiguous range.

### 3. `crossCorr(rcv, pulse_a)`
Computes the **matched filter** output via cross-correlation:

$$X_{rcv}[n] = (r \star x_a)[n] = \sum_k r[k]\, x_a^*[k - n]$$

Implemented as FFT-domain convolution using `scipy.signal.fftconvolve`:

$$X_{rcv} = \mathcal{F}^{-1}\!\lbrace\{\mathcal{F}\{r\} \cdot \overline{\mathcal{F}\{x_a\}}\rbrace\}$$

The matched filter maximises the output SNR when the noise is white and Gaussian.  
**Reference:** Oppenheim & Schafer, §8.7 — *Linear Filtering Methods Based on the DFT*; efficient convolution via the DFT. Ch. 9 — *Computation of the Discrete Fourier Transform* (FFT algorithms).

### 4. `findDelay(Xrcv, Nseg)`
Locates the index of the dominant peak in the matched filter output, corresponding to the **feedthrough** (direct speaker-to-microphone path). The receiver is then time-aligned to this reference, compensating for hardware latency and buffering delays.

### 5. `dist2time(dist, temperature)`
Converts a physical distance in cm to a sample index using the temperature-corrected speed of sound:

$$n = \text{round}\!\left(\frac{2 \cdot d}{100 \cdot v_s} \cdot f_s\right)$$

---

## DSP Techniques and References

### Hann Window — Sidelobe Suppression
The chirp pulse is multiplied by a **Hann (Hanning) window** before transmission and matched filtering:

$$w[n] = 0.5\!\left(1 - \cos\!\frac{2\pi n}{N-1}\right)$$

This reduces the autocorrelation sidelobes of the chirp at the cost of a slightly wider main lobe. Without windowing, high sidelobes would create ghost echoes at ranges where no object exists.  
**Reference:** Oppenheim & Schafer, §7.4 — *Design of FIR Filters by Windowing*, Table 7.2 (window characteristics).

### Overlap-and-Add — Block Streaming Convolution
Because audio arrives in real-time chunks shorter than the chirp correlation length, `signal_process()` maintains a 3×Nseg rolling buffer and accumulates matched filter outputs using the **overlap-and-add** method. This ensures that convolution results spanning chunk boundaries are correctly summed.  
**Reference:** Oppenheim & Schafer, §8.7.1 — *Overlap-Add Method*.

### FFT Convolution — Computational Efficiency
Direct cross-correlation of an $N$-sample recording with an $M$-sample pulse requires $O(NM)$ multiplications. The FFT-based approach reduces this to $O(N \log N)$, making real-time processing feasible at 48 kHz.  
**Reference:** Oppenheim & Schafer, Ch. 9 — *Computation of the Discrete Fourier Transform*; specifically the reduction of convolution complexity via the FFT.

### Pulse Compression
By sweeping from $f_0$ to $f_1$ over the pulse duration, the chirp spreads its energy across bandwidth $B = f_1 - f_0$. The matched filter then **compresses** this energy into a narrow peak of width $\approx 1/B$ seconds — equivalent to transmitting a short pulse of duration $1/B$ while enjoying the SNR benefit of a long pulse.  
**Reference:** Oppenheim & Schafer, §2.3.3 — *Correlation of Sequences*; the matched filter as the correlation peak maximiser.

### Normalisation and Gamma Correction
The matched filter output is normalised to its 97th percentile (rather than the global maximum) to prevent the large feedthrough spike at $d \approx 0$ from saturating the colorscale and washing out distant echoes. A gamma correction of $\gamma = 1/1.8$ is then applied to further enhance weak reflections.

### Time-of-Flight Ranging
The fundamental measurement is the round-trip propagation delay $\tau$, obtained from the peak location of the matched filter output. The temperature-corrected speed of sound accounts for the variation in $v_s$ that would otherwise introduce systematic range errors on the order of 0.17% per °C.  
**Reference:** Oppenheim & Schafer, §2.2 — *Discrete-Time Systems*; also the broader context of system identification and impulse response estimation.

---

## Threading Architecture (`rtsonar.py`)

The real-time engine uses five concurrent threads communicating via `queue.Queue` FIFOs:

| Thread | Function | Queue In | Queue Out |
|--------|----------|----------|-----------|
| Data queuer | `put_data` | — | `Qout` |
| Playback | `play_audio` | `Qout` | — |
| Recording | `record_audio` | — | `Qin` |
| Signal processing | `signal_process` | `Qin` | `Qdata` |
| Visualisation | `image_update` | `Qdata` | — |

All threads poll `stop_flag` (a `threading.Event`) for graceful shutdown. The display thread writes PNG bytes directly to an `ipywidgets.Image` widget — an operation that is thread-safe without requiring any GUI event loop synchronisation.

---

## Default Parameters

```python
fs          = 48000        # Sampling frequency (Hz)
f0, f1      = 6000, 12000  # Chirp sweep range (Hz) → bandwidth B = 6 kHz
Npulse      = 500          # Pulse length (samples) — longer → more energy, worse range res
Nseg        = 4096         # Inter-pulse spacing (samples) → max range ≈ 142 cm at 20°C
Nrep        = 100          # Waterfall history rows
Nplot       = 200          # Range bins in display
maxdist     = 100          # Maximum displayed range (cm)
temperature = 20           # Ambient temperature (°C)
```

**Update rate:** $f_s / N_{seg} \approx 11.7$ Hz at default settings. Halving `Nseg` doubles the frame rate but halves the maximum unambiguous range.

---

## Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Array operations, FFT |
| `scipy` | any | `fftconvolve`, `interpolate.interp1d` |
| `sounddevice` | any | Audio I/O (replaces PyAudio) |
| `matplotlib` | any | Jet colormap (`cm.jet`) |
| `Pillow` | any | Fast PNG encoding for live display |
| `ipywidgets` | any | Thread-safe in-notebook image widget |
| `jupyter` / `notebook` | any | Notebook runtime |

Install all dependencies into the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
pip install numpy scipy sounddevice matplotlib Pillow ipywidgets notebook
```

---

## Hardware Setup

- **Disable all audio enhancements** on both the microphone and speaker in Windows Sound Settings (Recording → Properties → Enhancements → Disable all). Noise suppression and automatic gain control will gate out the echo signal.
- Set speaker volume to **70–80%** to avoid clipping distortion.
- Place the speaker and microphone as close together as possible.
- Run in a **quiet environment** — room reverberation adds clutter to the waterfall.

---

## References

1. Oppenheim, A. V., & Schafer, R. W. (2014). *Discrete-Time Signal Processing* (3rd ed., Pearson New International Edition). Pearson Education.
2. Richards, M. A., Scheer, J. A., & Holm, W. A. (2010). *Principles of Modern Radar: Basic Principles*. SciTech Publishing. *(Pulse compression and matched filter theory.)*
3. Proakis, J. G., & Manolakis, D. G. (2007). *Digital Signal Processing: Principles, Algorithms, and Applications* (4th ed.). Pearson. *(FFT convolution, overlap-add.)*
4. SciPy documentation — `scipy.signal.fftconvolve`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html
5. SoundDevice documentation: https://python-sounddevice.readthedocs.io/
