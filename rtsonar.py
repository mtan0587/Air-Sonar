# Import functions and libraries
import numpy as np
import matplotlib.cm as cm
from scipy import signal
from scipy import interpolate
from numpy import *
import threading, time, queue, io
import sounddevice as sd
import ipywidgets as widgets
from PIL import Image as PILImage
from IPython.display import display, clear_output
import sys

# Display resolution (pixels rendered to screen, independent of Nrep/Nplot)
_DISP_W = 900
_DISP_H = 500


def put_data(Qout, ptrain, Twait, stop_flag):
    while not stop_flag.is_set():
        if Qout.qsize() < 2:
            Qout.put(ptrain)
        time.sleep(Twait)
    Qout.put("EOT")


def play_audio(Qout, fs, stop_flag, dev=None):
    with sd.OutputStream(samplerate=int(fs), channels=1, dtype='float32', device=dev) as ostream:
        while not stop_flag.is_set():
            data = Qout.get()
            if str(data) == "EOT":
                break
            try:
                ostream.write(data.astype(np.float32).reshape(-1, 1))
            except Exception:
                break


def record_audio(Qin, fs, stop_flag, dev=None, chunk=2048):
    with sd.InputStream(samplerate=int(fs), channels=1, dtype='float32',
                        device=dev, blocksize=chunk) as istream:
        while not stop_flag.is_set():
            try:
                data, _ = istream.read(chunk)
                Qin.put(data[:, 0])
            except Exception as e:
                print("Unexpected error:", e)
                break
    Qin.put("EOT")


def signal_process(Qin, Qdata, pulse_a, Nseg, Nplot, fs, maxdist, temperature, functions, stop_flag):
    crossCorr = functions[2]
    findDelay = functions[3]
    dist2time = functions[4]

    Xrcv = zeros(3 * Nseg, dtype='complex')
    cur_idx = 0
    found_delay = False
    maxsamp = int(np.minimum(dist2time(maxdist, temperature) * fs, Nseg))

    while not stop_flag.is_set():
        chunk = Qin.get()
        if str(chunk) == "EOT":
            break

        Xchunk = crossCorr(chunk, pulse_a)
        Xchunk = np.reshape(Xchunk, (1, len(Xchunk)))

        try:
            Xrcv[cur_idx:(cur_idx + len(chunk) + len(pulse_a) - 1)] += Xchunk[0, :]
        except Exception:
            pass

        cur_idx += len(chunk)

        if found_delay and cur_idx >= Nseg:
            idx = findDelay(abs(Xrcv), Nseg)
            Xrcv = np.roll(Xrcv, -idx)
            Xrcv[-idx:] = 0
            cur_idx = cur_idx - idx

            Xrcv_seg = (abs(Xrcv[:maxsamp].copy()) / np.maximum(abs(Xrcv[0]), 1e-5)) ** 0.5
            interp = interpolate.interp1d(r_[:maxsamp], Xrcv_seg)
            Xrcv_seg = interp(r_[:maxsamp - 1:(Nplot * 1j)])

            Xrcv = np.roll(Xrcv, -Nseg)
            Xrcv[-Nseg:] = 0
            cur_idx = cur_idx - Nseg

            Qdata.put(Xrcv_seg)

        elif cur_idx > 2 * Nseg:
            idx = findDelay(abs(Xrcv), Nseg)
            Xrcv = np.roll(Xrcv, -idx)
            Xrcv[-idx:] = 0
            cur_idx = cur_idx - idx - 1
            found_delay = True

    Qdata.put("EOT")


def image_update(Qdata, img_widget, img, Nrep, Nplot, stop_flag):
    while not stop_flag.is_set():
        new_line = Qdata.get()
        if str(new_line) == "EOT":
            break

        # 97th-percentile normalise + gamma correction
        new_line = np.minimum(new_line / np.maximum(np.percentile(new_line, 97), 1e-5), 1) ** (1 / 1.8)

        # Roll waterfall and insert new row
        img = np.roll(img, 1, 0)
        img[0, :] = new_line

        # Colormap → RGBA → upscale → PNG bytes (PIL, no matplotlib overhead)
        rgba = (cm.jet(img) * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(rgba, 'RGBA').resize(
            (_DISP_W, _DISP_H), PILImage.NEAREST
        )
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        img_widget.value = buf.getvalue()

        Qdata.queue.clear()


def rtsonar(f0, f1, fs, Npulse, Nseg, Nrep, Nplot, maxdist, temperature, functions,
            in_dev=None, out_dev=None):

    clear_output()
    genChirpPulse = functions[0]
    genPulseTrain = functions[1]

    pulse_a = genChirpPulse(Npulse, f0, f1, fs)
    hanWin = np.hanning(Npulse)
    hanWin = np.reshape(hanWin, (Npulse, 1))
    pulse_a = np.multiply(pulse_a, hanWin)
    pulse = np.real(pulse_a)
    ptrain = genPulseTrain(pulse, Nrep, Nseg)

    Qin = queue.Queue()
    Qout = queue.Queue()
    Qdata = queue.Queue()

    img_arr = np.zeros((Nrep, Nplot))

    fps = fs / Nseg
    label_top = widgets.HTML(
        f'<b>Sonar — 0 → {maxdist} cm &nbsp;|&nbsp; '
        f'Row spacing: {Nseg/fs*1000:.0f} ms &nbsp;|&nbsp; '
        f'Max rate: {fps:.1f} Hz &nbsp;|&nbsp; '
        f'</b>'
    )
    img_widget = widgets.Image(format='png', width=_DISP_W, height=_DISP_H)
    label_bot = widgets.HTML(
        '<span style="font-size:11px;color:#555">'
        '← near (0 cm)'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        f'far ({maxdist} cm) →'
        '&nbsp;&nbsp;|&nbsp;&nbsp;'
        '<span style="color:#00f">&#9632;</span> no echo &nbsp;'
        '<span style="color:#0a0">&#9632;</span> weak &nbsp;'
        '<span style="color:#ff0">&#9632;</span> medium &nbsp;'
        '<span style="color:red">&#9632;</span> strong echo'
        '</span>'
    )
    display(widgets.VBox([label_top, img_widget, label_bot]))

    # Initial blank frame
    rgba0 = (cm.jet(img_arr) * 255).astype(np.uint8)
    buf0 = io.BytesIO()
    PILImage.fromarray(rgba0, 'RGBA').resize((_DISP_W, _DISP_H), PILImage.NEAREST).save(buf0, format='PNG')
    img_widget.value = buf0.getvalue()

    stop_flag = threading.Event()

    t_put_data       = threading.Thread(target=put_data,        args=(Qout, ptrain, Nseg / fs * 3, stop_flag))
    t_rec            = threading.Thread(target=record_audio,    args=(Qin, fs, stop_flag, in_dev))
    t_play_audio     = threading.Thread(target=play_audio,      args=(Qout, fs, stop_flag, out_dev))
    t_signal_process = threading.Thread(target=signal_process,  args=(Qin, Qdata, pulse_a, Nseg, Nplot, fs, maxdist, temperature, functions, stop_flag))
    t_image_update   = threading.Thread(target=image_update,    args=(Qdata, img_widget, img_arr, Nrep, Nplot, stop_flag))

    t_put_data.start()
    t_rec.start()
    t_play_audio.start()
    t_signal_process.start()
    t_image_update.start()

    return stop_flag
