"""Synthesize the showreel soundtrack.

Reads the sound cues exported by render.mjs (cues.json: every pop, stamp and
whoosh the page schedules, with its time in seconds) and writes a 48 kHz
stereo WAV: a 120 bpm track in A minor plus sound design synced to picture.

    python3 audio.py cues.json audio.wav
"""
import json
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48000
DUR = 15.0
N = int(SR * DUR)
BEAT = 0.5  # 120 bpm
rng = np.random.default_rng(2026)


# ---------------------------------------------------------------- helpers
def tt(d):
    return np.arange(int(d * SR)) / SR


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def lp(x, f, order=2):
    return sosfilt(butter(order, min(f, SR * 0.45), 'low', fs=SR, output='sos'), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, 'high', fs=SR, output='sos'), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, min(hi, SR * 0.45)], 'band', fs=SR, output='sos'), x)


def sweep(x, f0, f1, kind='low', block=256, curve=2.0):
    """Block-wise filter sweep (exponential in frequency)."""
    out = np.zeros_like(x)
    nb = int(np.ceil(len(x) / block))
    zi = None
    for b in range(nb):
        p = (b / max(1, nb - 1)) ** curve
        f = f0 * (f1 / f0) ** p
        sos = butter(2, min(f, SR * 0.45), kind, fs=SR, output='sos')
        seg = x[b * block:(b + 1) * block]
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        y, zi = sosfilt(sos, seg, zi=zi)
        out[b * block:(b + 1) * block] = y
    return out


def saw(freq, d, phase=0.0):
    """PolyBLEP band-limited sawtooth. freq may be an array (glides)."""
    n = int(d * SR)
    f = np.broadcast_to(np.asarray(freq, float), (n,)) if np.ndim(freq) else np.full(n, float(freq))
    dt = f / SR
    ph = (phase + np.cumsum(dt)) % 1.0
    y = 2 * ph - 1
    m1 = ph < dt
    x = ph[m1] / dt[m1]
    y[m1] -= x + x - x * x - 1
    m2 = ph > 1 - dt
    x = (ph[m2] - 1) / dt[m2]
    y[m2] -= x * x + x + x + 1
    return y


def sine(freq, d, phase=0.0):
    n = int(d * SR)
    f = np.broadcast_to(np.asarray(freq, float), (n,)) if np.ndim(freq) else np.full(n, float(freq))
    return np.sin(2 * np.pi * (phase + np.cumsum(f) / SR))


def noise(d):
    return rng.standard_normal(int(d * SR))


def env(d, a=0.002, decay=0.2, hold=0.0):
    t = tt(d)
    e = np.minimum(1, t / max(a, 1e-5))
    return e * np.where(t < a + hold, 1, np.exp(-(t - a - hold) / decay))


class Bus:
    def __init__(self):
        self.x = np.zeros((N, 2))

    def add(self, sig, t, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i >= N:
            return
        if sig.ndim == 1:
            l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            sig = np.stack([sig * l * 1.414, sig * r * 1.414], 1)
        j0 = max(0, -i)
        i = max(0, i)
        k = min(N - i, len(sig) - j0)
        if k > 0:
            self.x[i:i + k] += sig[j0:j0 + k] * gain


# ------------------------------------------------------------ instruments
def kick(big=False):
    d = 0.6 if big else 0.42
    t = tt(d)
    f = 44 + 120 * np.exp(-t / 0.032)
    body = sine(f, d) * np.exp(-t / (0.34 if big else 0.24))
    click = hp(noise(0.004), 3000) * 0.4
    body[:len(click)] += click
    return np.tanh(body * 1.6)


def clap():
    d = 0.3
    x = np.zeros(int(d * SR))
    for k, off in enumerate([0, 0.011, 0.022]):
        b = noise(0.03) * np.exp(-tt(0.03) / 0.006)
        i = int(off * SR)
        x[i:i + len(b)] += b * (0.8 if k < 2 else 1)
    tail = noise(d) * np.exp(-tt(d) / 0.09)
    x += tail * 0.7
    return bp(x, 900, 2600) * 1.6


def hat(open_=False):
    d = 0.22 if open_ else 0.05
    x = hp(noise(d), 7500, 3) * np.exp(-tt(d) / (0.07 if open_ else 0.014))
    return x


def snare():
    d = 0.18
    t = tt(d)
    tone = sine(190 * (1 + 0.4 * np.exp(-t / 0.01)), d) * np.exp(-t / 0.05)
    nz = bp(noise(d), 1500, 9000) * np.exp(-t / 0.06)
    return tone * 0.5 + nz


def supersaw(freq, d, voices=5, detune=0.14, cutoff=2400):
    L = np.zeros(int(d * SR))
    R = np.zeros_like(L)
    for v in range(voices):
        c = (v / (voices - 1) - 0.5) * 2 * detune  # semitone offset
        s = saw(freq * 2 ** (c / 12), d, phase=rng.random())
        w = v / (voices - 1)
        L += s * (1 - w * 0.7)
        R += s * (0.3 + w * 0.7)
    return lp(L, cutoff) / voices, lp(R, cutoff) / voices


def chord(notes, d, cutoff=2400, a=0.012, rel=0.08, level=1.0):
    L = np.zeros(int(d * SR))
    R = np.zeros_like(L)
    for m in notes:
        l, r = supersaw(midi(m), d, cutoff=cutoff)
        L += l
        R += r
    t = tt(d)
    e = np.minimum(1, t / a) * np.clip((d - t) / rel, 0, 1)
    return np.stack([L * e, R * e], 1) * level / np.sqrt(len(notes))


def stab(notes, d=0.6, cutoff=5200):
    s = chord(notes, d, cutoff=cutoff, a=0.003, rel=0.05)
    return s * np.exp(-tt(d) / 0.16)[:, None]


def bass_note(m, d):
    t = tt(d)
    s = saw(midi(m), d) * 0.7 + sine(midi(m - 12), d) * 0.8
    s = sweep(s, 1400, 260, 'low', curve=0.5)
    return s * np.minimum(1, t / 0.004) * np.clip((d - t) / 0.02, 0, 1) * np.exp(-t / 0.35)


def pluck(m, d=0.22):
    t = tt(d)
    s = saw(midi(m), d) * 0.6 + np.sign(sine(midi(m), d)) * 0.25
    s = sweep(s, 6000, 900, 'low', curve=0.6)
    return s * np.exp(-t / 0.07)


def bell(f, d=0.7):
    t = tt(d)
    return (sine(f, d) * np.exp(-t / 0.35) + 0.45 * sine(f * 2.76, d) * np.exp(-t / 0.12)
            + 0.2 * sine(f * 5.4, d) * np.exp(-t / 0.05))


def pop(p=1.0):
    d = 0.14
    t = tt(d)
    f = 420 * p * (1 + 1.4 * np.exp(-t / 0.018))
    return sine(f, d) * np.exp(-t / 0.045) + hp(noise(d), 2000) * np.exp(-t / 0.004) * 0.25


def riser(d):
    t = tt(d)
    nz = sweep(noise(d), 300, 9000, 'low', curve=1.6) * (t / d) ** 2.2
    tone = sine(180 * (6 ** (t / d) ** 1.5), d) * (t / d) ** 2 * 0.25
    return nz + tone


def whoosh(d):
    x = sweep(noise(d), 500, 7000, 'low', curve=1.2)
    t = tt(d)
    shape = np.sin(np.pi * np.clip(t / d, 0, 1) ** 1.7) ** 1.5
    return hp(x, 200) * shape


def swish(d):
    t = tt(d)
    return sweep(hp(noise(d), 1500), 2500, 9000, 'low', curve=1) * np.sin(np.pi * t / d) ** 2


def impact(big=False):
    d = 1.6 if big else 1.1
    t = tt(d)
    boom = sine(62 * np.exp(-t / 0.5) + 30, d) * np.exp(-t / (0.55 if big else 0.4))
    crash = hp(noise(d), 3500) * np.exp(-t / (0.9 if big else 0.55)) * 0.35
    snap = bp(noise(0.08), 800, 5000) * np.exp(-tt(0.08) / 0.02)
    out = boom * 1.2 + crash
    out[:len(snap)] += snap * 0.9
    return out


def stamp_sfx(big=False):
    d = 0.5
    t = tt(d)
    thud = sine(95 * np.exp(-t / 0.05) + 42, d) * np.exp(-t / (0.2 if big else 0.13))
    slap = lp(noise(0.06), 2200) * np.exp(-tt(0.06) / 0.012)
    thud[:len(slap)] += slap * 0.9
    return np.tanh(thud * 1.5)


def scribble(d):
    t = tt(d)
    am = np.abs(np.sin(2 * np.pi * 11 * t + 3 * np.sin(2 * np.pi * 3 * t))) ** 0.6
    return bp(noise(d), 1800, 6000) * am * np.sin(np.pi * t / d) * 0.7


def tick(f=1800):
    d = 0.03
    t = tt(d)
    return sine(f, d) * np.exp(-t / 0.006) * 0.6 + hp(noise(d), 4000) * np.exp(-t / 0.002) * 0.5


def tock(k):
    d = 0.12
    t = tt(d)
    f = [196, 220, 247, 262, 294, 330][k % 6] * 1.5
    return sine(f * (1 + 0.5 * np.exp(-t / 0.01)), d) * np.exp(-t / 0.035) + lp(noise(d), 1500) * np.exp(-t / 0.01) * 0.4


def confetti():
    d = 1.2
    t = tt(d)
    burst = bp(noise(d), 1200, 9000) * np.exp(-t / 0.05)
    crackle = np.zeros_like(t)
    for _ in range(90):
        i = int(rng.uniform(0.02, 1.0) ** 1.6 * len(t))
        c = hp(noise(0.006), 3000) * np.exp(-tt(0.006) / 0.0015) * rng.uniform(0.2, 0.7)
        crackle[i:i + len(c)] += c[:len(crackle) - i]
    return burst + crackle * np.exp(-t / 0.5)


# ------------------------------------------------------------ arrangement
def main(cue_path, out_path):
    cues = json.load(open(cue_path))
    drums, music, sfx, send = Bus(), Bus(), Bus(), Bus()

    CH = {
        'Am': [57, 60, 64, 67], 'F': [53, 57, 60, 64], 'C': [55, 60, 64, 67], 'G': [55, 59, 62, 67],
        'Cfin': [48, 55, 60, 64, 67, 74],
    }
    BASS = {'Am': 45, 'F': 41, 'C': 48, 'G': 43, 'Cfin': 36}
    # (start, end, chord)
    prog = [(2, 4, 'Am'), (4, 6, 'F'), (6, 8, 'C'), (8, 10, 'G'), (10, 12, 'Am'), (12, 12.5, 'F'),
            (12.5, 13.25, 'F'), (13.25, 14.0, 'G')]

    # intro stabs on the four hits
    for i, name in enumerate(['Am', 'F', 'C', 'G']):
        s = stab([n + 12 for n in CH[name]], 0.6)
        music.add(s, i * 0.5, 0.55)
        send.add(s, i * 0.5, 0.35)
        drums.add(kick(), i * 0.5, 0.9)
    # snare roll into the drop
    for k in range(12):
        tr = 1.5 + 0.5 * (1 - (1 - k / 12) ** 1.3)
        drums.add(snare(), tr, 0.12 + 0.35 * k / 12)

    # groove
    kick_times = []
    for b in range(int(DUR / BEAT)):
        t = b * BEAT
        in_groove = (2.0 <= t < 12.0) or (12.5 <= t < 14.0)
        if not in_groove:
            continue
        kick_times.append(t)
        drums.add(kick(), t, 1.0)
        if b % 2 == 1:
            c = clap()
            drums.add(c, t, 0.55)
            send.add(c, t, 0.18)
        drums.add(hat(), t + 0.25, 0.28, pan=0.25)
        if t >= 4.5:
            drums.add(hat(True), t + 0.25, 0.14, pan=0.3)
            for s16 in (0.125, 0.375):
                drums.add(hat(), t + s16, 0.1, pan=-0.2)
    # build 12.0 → 12.5
    for k in range(8):
        drums.add(snare(), 12.0 + k * 0.0625, 0.15 + 0.5 * k / 8)

    # sidechain envelope from the kicks
    t_all = np.arange(N) / SR
    duck = np.ones(N)
    for kt in kick_times:
        i = int(kt * SR)
        seg = t_all[i:i + int(0.35 * SR)] - kt
        duck[i:i + len(seg)] = np.minimum(duck[i:i + len(seg)], 1 - 0.65 * np.exp(-seg / 0.09))

    # pads + bass
    for (t0, t1, name) in prog:
        cut = 1800 if t0 < 10 else 2600
        if t0 == 12:
            cut = 1500
        p = chord(CH[name], t1 - t0 + 0.05, cutoff=cut, level=0.9)
        music.add(p, t0, 0.42)
        send.add(p, t0, 0.3)
        for k in range(int((t1 - t0) / 0.25)):
            tb = t0 + k * 0.25
            if tb >= 12.0 and tb < 12.5:
                continue
            m = BASS[name] + (12 if k % 4 == 3 else 0)
            music.add(bass_note(m, 0.24), tb, 0.5)

    # arp (7.5 → 14)
    arp_pat = [0, 1, 2, 3, 2, 1, 3, 2]
    for (t0, t1, name) in prog:
        for k in range(int((t1 - t0) / 0.125)):
            ta = t0 + k * 0.125
            if ta < 7.5 or (12.0 <= ta < 12.5):
                continue
            notes = CH[name]
            m = notes[arp_pat[k % 8] % len(notes)] + (24 if ta >= 10 else 12)
            pl = pluck(m)
            g = 0.16 if ta < 10 else 0.2
            music.add(pl, ta, g, pan=-0.35 if k % 2 else 0.35)
            music.add(pl, ta + 0.375, g * 0.35, pan=0.6 if k % 2 else -0.6)  # dotted-8th echo
            send.add(pl, ta, 0.08)

    # final chord on the stamp
    fin = chord(CH['Cfin'], 1.0, cutoff=3200, a=0.004, rel=0.3, level=1.0)
    fin *= np.exp(-tt(1.0) / 0.45)[:, None]
    music.add(fin, 14.0, 0.8)
    send.add(fin, 14.0, 0.5)
    music.add(sine(midi(36), 1.0) * np.exp(-tt(1.0) / 0.4), 14.0, 0.5)

    # ---------------- picture-synced sound design
    for c in cues:
        t, ty = c['t'], c['type']
        if ty == 'hit':
            s = impact(False)
            sfx.add(s, t, 0.45)
        elif ty == 'riser':
            d = c.get('dur', 0.5)
            s = riser(d)
            sfx.add(s, t, 0.5)
            send.add(s, t, 0.2)
        elif ty == 'drop':
            big = bool(c.get('big'))
            s = impact(True)
            sfx.add(s, t, 0.9 if big else 0.75)
            send.add(s, t, 0.3)
            drums.add(kick(True), t, 1.0)
        elif ty == 'pop':
            s = pop(c.get('p', 1.0))
            sfx.add(s, t, 0.4, pan=rng.uniform(-0.3, 0.3))
            send.add(s, t, 0.12)
        elif ty == 'thud':
            sfx.add(stamp_sfx(), t, 0.35 * c.get('v', 1))
        elif ty == 'swish':
            sfx.add(swish(c.get('dur', 0.3)), t, 0.22, pan=rng.uniform(-0.4, 0.4))
        elif ty == 'whoosh':
            d = c.get('dur', 0.4)
            s = whoosh(d * 1.25)
            sfx.add(s, t - d, 0.5)
        elif ty == 'ping':
            f = midi([69, 72, 76][c.get('p', 0) % 3] + 12)
            s = bell(f)
            sfx.add(s, t, 0.2, pan=[-0.4, 0, 0.4][c.get('p', 0) % 3])
            send.add(s, t, 0.2)
        elif ty == 'stamp':
            big = bool(c.get('big'))
            sfx.add(stamp_sfx(big), t, 0.9 if big else 0.7)
            send.add(stamp_sfx(big), t, 0.25)
            if big:
                sfx.add(impact(False), t, 0.4)
        elif ty == 'scribble':
            sfx.add(scribble(c.get('dur', 0.45)), t, 0.28, pan=0.3)
        elif ty == 'type':
            sfx.add(tick(1500 + 300 * (c.get('k', 0) % 3)), t, 0.12, pan=-0.3)
        elif ty == 'land':
            k = c.get('k', 0)
            sfx.add(tock(k), t, 0.3, pan=0.35 + 0.1 * np.sin(k))
        elif ty == 'blip':
            k = c.get('k', 1)
            scale = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24, 26]
            f = midi(69 + scale[k - 1])
            d = 0.35 if k == 12 else 0.06
            s = (np.sign(sine(f, d)) * 0.3 + sine(f, d) * 0.5) * np.exp(-tt(d) / (0.12 if k == 12 else 0.02))
            s = lp(s, 5000)
            sfx.add(s, t, 0.18 if k < 12 else 0.3)
            if k == 12:
                send.add(s, t, 0.3)
        elif ty == 'confetti':
            sfx.add(confetti(), t, 0.35)

    # ---------------- mix
    music.x *= duck[:, None]
    ir_d = 2.2
    ir_t = tt(ir_d)
    ir = np.stack([lp(noise(ir_d), 6000), lp(noise(ir_d), 6000)], 1) * np.exp(-ir_t / 0.45)[:, None]
    ir[: int(0.012 * SR)] = 0  # predelay
    ir /= np.sqrt((ir ** 2).sum(0))
    wet = np.stack([fftconvolve(send.x[:, c], ir[:, c])[:N] for c in range(2)], 1)
    wet = hp(wet.T, 250).T

    mix = drums.x * 0.7 + music.x * 1.3 + sfx.x * 1.0 + wet * 0.6
    db = lambda b: 20 * np.log10(np.sqrt((b ** 2).mean()) + 1e-9)
    print(f'bus rms  drums {db(drums.x * 0.7):.1f}  music {db(music.x * 1.3):.1f}  sfx {db(sfx.x):.1f}  reverb {db(wet * 0.6):.1f} dB')
    mix = hp(mix.T, 28).T
    # gentle bus compression + soft clip
    rms = np.sqrt(lp((mix ** 2).mean(1), 8) + 1e-9)
    gain = np.minimum(1, (0.25 / rms) ** 0.35)
    mix *= gain[:, None]
    mix /= np.abs(mix).max()
    mix = np.tanh(mix * 1.6) / np.tanh(1.6)
    # 10 ms fade-in, 250 ms fade-out
    fi, fo = int(0.01 * SR), int(0.25 * SR)
    mix[:fi] *= np.linspace(0, 1, fi)[:, None]
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5
    mix *= 10 ** (-1 / 20) / np.abs(mix).max()
    wavfile.write(out_path, SR, (mix * 32767).astype(np.int16))
    print(f'wrote {out_path}: {DUR}s, peak -1 dBFS, rms {20 * np.log10(np.sqrt((mix ** 2).mean())):.1f} dBFS')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cues.json', sys.argv[2] if len(sys.argv) > 2 else 'audio.wav')
