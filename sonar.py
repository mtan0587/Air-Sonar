"""
sonar.py — Standalone acoustic sonar using laptop speaker + microphone.

Chirp pulse compression, matched filtering, 97th-percentile normalization,
gamma correction, and hot-colormap waterfall display.

Inspired by:
  - EE123 Laptop Sonar Lab, UC Berkeley (Prof. Miki Lustig)
    https://github.com/alexwal/EE123-Labs/blob/master/Lab1A_Laptop_Sonar.ipynb
  - ELEC3305 Time-Domain Sonar Lab, University of Sydney

Usage:
    python sonar.py              # runs both pure-tone and chirp experiments
    python sonar.py --list       # list available audio devices
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import queue
from threading import Lock
from scipy import signal
import pyaudio

# ── Default parameters ────────────────────────────────────────────────────────
FS      = 48000   # sampling rate (Hz)
IN_DEV  = 9       # mic device index   (Intel Smart Sound — run --list to find yours)
OUT_DEV = 8       # speaker device idx (Realtek          — run --list to find yours)

F0      = 6000    # chirp start frequency (Hz)
F1      = 12000   # chirp end frequency   (Hz)  → BW = 6 kHz
NPULSE  = 360     # pulse length (samples) — 7.5 ms at 48 kHz
NSEG    = 4800    # segment length (samples) = 100 ms → max range ≈ 17 m
NREP    = 100     # waterfall rows (number of pings)
MAXDIST = 250     # display range (cm)
TEMP    = 21      # ambient temperature (°C)

# ── Student functions ─────────────────────────────────────────────────────────

def genChirpPulse(Npulse, f0, f1, fs):
    """Analytic chirp pulse: exp(j*2π*φ(t)), shape (Npulse, 1)."""
    t   = np.arange(Npulse) / fs
    k   = (f1 - f0) / (Npulse / fs)
    phi = f0 * t + (k / 2) * t**2
    return np.exp(1j * 2 * np.pi * phi).reshape(-1, 1)


def genPulseTrain(pulse, Nrep, Nseg):
    """Repeat pulse Nrep times with Nseg spacing (zero-padded between pulses)."""
    pulse_1d = np.ravel(pulse)
    segment  = np.zeros(Nseg, dtype=pulse_1d.dtype)
    segment[:len(pulse_1d)] = pulse_1d
    return np.tile(segment, Nrep)


def crossCorr(rcv, pulse_a):
    """Matched filter: FFT cross-correlation of rcv with pulse_a."""
    p = np.ravel(pulse_a)
    return signal.fftconvolve(np.ravel(rcv), np.conj(p[::-1]), mode='full')


def findDelay(Xrcv, Nseg):
    """
    Find the index of the strongest peak in Xrcv.
    When called from the tracking loop (len == Nseg), searches only the
    centre 50 % of the window to prevent echo-induced drift.
    """
    arr = np.abs(np.ravel(Xrcv))
    if len(arr) == 0:
        return 0
    if 0 < len(arr) <= Nseg:          # loop context: constrain search
        lo = Nseg // 4
        hi = min(3 * Nseg // 4, len(arr))
        if lo < hi:
            return lo + int(np.argmax(arr[lo:hi]))
    return int(np.argmax(arr))


def dist2time(dist_cm, temperature=21):
    """Convert distance in cm → two-way travel time in seconds."""
    v_s = 331.5 * np.sqrt(1 + temperature / 273.15)
    return 2 * dist_cm / (100 * v_s)


def time2dist(t, temperature=21):
    """Convert two-way travel time in seconds → distance in cm."""
    v_s = 331.5 * np.sqrt(1 + temperature / 273.15)
    return t * v_s * 100 / 2


# ── Audio I/O ─────────────────────────────────────────────────────────────────

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *_): pass


def _play_audio(Q, p, fs, dev=None):
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),
                    output=True, output_device_index=dev)
    while True:
        data = Q.get()
        if isinstance(data, str) and data == 'EOT':
            break
        try:
            stream.write(data.astype(np.float32).tobytes())
        except Exception:
            break


def _record_audio(q, p, fs, dev=None, chunk=2048, lock=None):
    ctx = lock if lock is not None else _NullCtx()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),
                    input=True, input_device_index=dev, frames_per_buffer=chunk)
    while True:
        try:
            with ctx:
                raw = stream.read(chunk, exception_on_overflow=False)
        except Exception:
            break
        q.put(np.frombuffer(raw, dtype=np.float32))


def xciever(sig, fs, in_dev=None, out_dev=None):
    """Play sig and simultaneously record; returns recorded numpy array."""
    rcv  = []
    Qin  = queue.Queue()
    Qout = queue.Queue()
    lock = Lock()
    p    = pyaudio.PyAudio()

    t_rec  = threading.Thread(target=_record_audio, args=(Qin, p, fs),
                               kwargs={'dev': in_dev, 'lock': lock}, daemon=True)
    t_play = threading.Thread(target=_play_audio,   args=(Qout, p, fs),
                               kwargs={'dev': out_dev}, daemon=True)
    t_rec.start()
    t_play.start()

    Qout.put(sig)
    Qout.put('EOT')
    time.sleep(len(sig) / fs + 2.0)

    with lock:
        p.terminate()
    while not Qin.empty():
        rcv = np.append(rcv, Qin.get())
    return rcv


# ── Core sonar engine ─────────────────────────────────────────────────────────

def run_sonar(npulse=NPULSE, f0=F0, f1=F1, fs=FS, nrep=NREP, nseg=NSEG,
              in_dev=IN_DEV, out_dev=OUT_DEV, temperature=TEMP):
    """
    Transmit a Hanning-windowed chirp pulse train, record echoes,
    apply matched filtering, and build a (nrep × nseg) waterfall image.
    Returns: img, fs, nrep, nseg, temperature
    """
    bw = abs(f1 - f0) if f1 != f0 else fs / npulse
    range_res = time2dist(1 / bw, temperature) if bw else None

    print(f"\n{'─'*50}")
    print(f"  Chirp: {f0}–{f1} Hz  |  BW = {bw} Hz  |  Npulse = {npulse}")
    print(f"  Range resolution ≈ {range_res:.1f} cm" if range_res else "  Pure tone (no pulse compression)")
    print(f"  Pings: {nrep}  |  PRI = {nseg/fs*1000:.0f} ms  |  Max range ≈ {time2dist(nseg/fs, temperature):.0f} cm")
    print(f"  Recording for {nrep*nseg/fs + 2:.1f} s …")

    # Build pulse
    pulse_a = genChirpPulse(npulse, f0, f1, fs)
    pulse_a = pulse_a * np.hanning(npulse).reshape(-1, 1)
    pulse   = np.real(pulse_a)
    ptrain  = genPulseTrain(pulse, nrep, nseg)

    # Record
    rcv = xciever(ptrain / 2.0, fs, in_dev=in_dev, out_dev=out_dev)

    # Matched filter
    Xrcv_a = np.abs(crossCorr(rcv, pulse_a)).reshape(1, -1)

    # Build waterfall
    idx = findDelay(Xrcv_a, nseg)
    img = np.zeros((nrep, nseg))

    end = min(idx + nseg, Xrcv_a.shape[1])
    img[0, :end - idx] = Xrcv_a[0, idx:end]

    for n in range(1, nrep):
        win_start = idx + nseg // 2
        win_end   = win_start + nseg
        if win_end > Xrcv_a.shape[1]:
            print(f"  Signal ended early at row {n}/{nrep} — remaining rows are zero.")
            break
        idxx = findDelay(Xrcv_a[0, win_start:win_end], nseg)
        idx  = win_start + idxx
        end  = min(idx + nseg, Xrcv_a.shape[1])
        img[n, :end - idx] = Xrcv_a[0, idx:end]

    print("  Done.")
    return img, fs, nrep, nseg, temperature


# ── Visualization ─────────────────────────────────────────────────────────────

def display_sonar(img, fs, nrep, nseg, temperature, maxdist=MAXDIST,
                  title='Sonar waterfall', cmap='hot'):
    """
    Display the waterfall image with:
      - 97th-percentile normalization  (feedthrough doesn't saturate the display)
      - gamma = 1/1.8 correction       (enhances weak echoes)
      - distance axis in cm
    """
    # Normalize: use 97th percentile so the huge feedthrough spike doesn't
    # set the scale and make all room echoes invisible.
    norm_val = np.percentile(img, 97)
    if norm_val == 0:
        norm_val = img.max() or 1.0
    img_norm = np.clip(img / norm_val, 0, 1)

    # Gamma correction: brings up faint echoes from the noise floor
    img_disp = img_norm ** (1 / 1.8)

    max_sample = min(int(dist2time(maxdist, temperature) * fs), nseg)
    extent = (
        0,
        time2dist(max_sample / fs, temperature),
        nrep * nseg / fs,
        0,
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(img_disp[:, :max_sample],
                   aspect='auto', cmap=cmap,
                   vmin=0, vmax=1,
                   interpolation='bilinear',
                   extent=extent)
    ax.set_xlabel('Distance (cm)', fontsize=12)
    ax.set_ylabel('Time (s)',       fontsize=12)
    ax.set_title(title,             fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalized echo intensity (γ-corrected)')
    plt.tight_layout()
    plt.show()
    return fig


def compare_sonar(img_tone, img_chirp, fs, nrep, nseg, temperature, maxdist=MAXDIST):
    """Side-by-side comparison of pure-tone vs chirp sonar images."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    datasets = [
        (img_tone, '8 kHz pure tone  (Npulse=72)  — wide main lobe'),
        (img_chirp, '6–12 kHz chirp  (Npulse=360, BW=6 kHz)  — 6× better resolution'),
    ]
    max_sample = min(int(dist2time(maxdist, temperature) * fs), nseg)
    extent = (0, time2dist(max_sample / fs, temperature), nrep * nseg / fs, 0)

    for ax, (img, label) in zip(axes, datasets):
        norm_val = np.percentile(img, 97) or img.max() or 1.0
        img_disp = np.clip(img / norm_val, 0, 1) ** (1 / 1.8)
        im = ax.imshow(img_disp[:, :max_sample], aspect='auto', cmap='hot',
                       vmin=0, vmax=1, interpolation='bilinear', extent=extent)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Distance (cm)')
    axes[0].set_ylabel('Time (s)')
    plt.colorbar(im, ax=axes, label='Echo intensity')
    plt.suptitle('Pulse compression comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig


# ── Utilities ─────────────────────────────────────────────────────────────────

def list_devices():
    """Print all available PyAudio audio devices."""
    p = pyaudio.PyAudio()
    print(f"\n{'─'*60}")
    print(f"  {'Idx':>4}  {'Name':<40}  {'In':>3}  {'Out':>3}  {'Rate':>7}")
    print(f"{'─'*60}")
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        print(f"  {i:>4}  {d['name']:<40}  {int(d['maxInputChannels']):>3}"
              f"  {int(d['maxOutputChannels']):>3}  {int(d['defaultSampleRate']):>7}")
    print(f"{'─'*60}\n")
    p.terminate()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Laptop acoustic sonar')
    parser.add_argument('--list',    action='store_true', help='List audio devices and exit')
    parser.add_argument('--in-dev',  type=int, default=IN_DEV,  help='Input  device index')
    parser.add_argument('--out-dev', type=int, default=OUT_DEV, help='Output device index')
    parser.add_argument('--nrep',    type=int, default=NREP,    help='Number of pings')
    parser.add_argument('--maxdist', type=int, default=MAXDIST, help='Display range (cm)')
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    kw = dict(fs=FS, nrep=args.nrep, nseg=NSEG,
              in_dev=args.in_dev, out_dev=args.out_dev, temperature=TEMP)

    print("=== ELEC3305 / EE123 Laptop Acoustic Sonar ===")
    print(f"Devices: in={args.in_dev}  out={args.out_dev}  |  Temp={TEMP}°C")

    # ── Experiment 1: pure 8 kHz tone ─────────────────────────────────────────
    img_tone, *a1 = run_sonar(npulse=72, f0=8000, f1=8000, **kw)
    display_sonar(img_tone, *a1, maxdist=args.maxdist,
                  title='Sonar — 8 kHz pure tone (Npulse=72, no pulse compression)')

    # ── Experiment 2: 6–12 kHz chirp ──────────────────────────────────────────
    img_chirp, *a2 = run_sonar(npulse=360, f0=6000, f1=12000, **kw)
    display_sonar(img_chirp, *a2, maxdist=args.maxdist,
                  title='Sonar — 6–12 kHz chirp (Npulse=360, BW=6 kHz, PCR≈45×)')

    # ── Side-by-side comparison ────────────────────────────────────────────────
    compare_sonar(img_tone, img_chirp, FS, args.nrep, NSEG, TEMP, args.maxdist)


if __name__ == '__main__':
    main()
