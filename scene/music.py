"""Task 5: procedural dance track (royalty-free, exactly on the choreography's beat grid).

120 BPM electro-pop in A minor (Am - F - C - G): kick on every beat, clap on 2 and 4, off-beat
hi-hats, an eighth-note bass line, a pad and a 16th-note arpeggio. One bar of intro and
outro. Written as a 16-bit mono WAV. Replace it with a real song via `--song`, but then also
regenerate the choreography with the matching BPM and length.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

CHORDS = [  # (bass root Hz, triad Hz) per bar
    (110.00, (220.00, 261.63, 329.63)),   # Am
    (87.31, (174.61, 220.00, 261.63)),    # F
    (130.81, (261.63, 329.63, 392.00)),   # C
    (98.00, (196.00, 246.94, 293.66)),    # G
]


def _env(n: int, sr: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / sr
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def _place(track: np.ndarray, sound: np.ndarray, start: int) -> None:
    end = min(len(track), start + len(sound))
    if start < end:
        track[start:end] += sound[: end - start]


def generate_song(path: str | Path, duration: float = 64.0, bpm: float = 120.0, sr: int = 22050,
                  seed: int = 38) -> Path:
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    beat = 60.0 / bpm
    bar = 4 * beat
    out = np.zeros(n, np.float32)
    drums, bass, pad, lead = (np.zeros(n, np.float32) for _ in range(4))

    # --- one-shot sounds
    k_n = int(0.35 * sr)
    tk = np.arange(k_n) / sr
    kick = np.sin(2 * np.pi * (45 * tk + (110 / 25) * (1 - np.exp(-25 * tk)))) * np.exp(-tk / 0.12)
    c_n = int(0.2 * sr)
    clap = rng.standard_normal(c_n) * _env(c_n, sr, 0.001, 0.05)
    clap = np.convolve(clap, [1, -0.9], "same")                          # crude high-pass
    h_n = int(0.06 * sr)
    hat = np.diff(rng.standard_normal(h_n + 1)) * _env(h_n, sr, 0.0005, 0.015)

    n_bars = int(duration / bar)
    for b in range(n_bars):
        root, triad = CHORDS[b % len(CHORDS)]
        bar0 = int(b * bar * sr)
        intro_outro = b == 0 or b >= n_bars - 1
        for k in range(4):
            s = bar0 + int(k * beat * sr)
            if not intro_outro:
                _place(drums, 0.9 * kick, s)
                if k in (1, 3):
                    _place(drums, 0.35 * clap, s)
            _place(drums, 0.12 * hat, s + int(beat / 2 * sr))
        # bass: eighth notes, root with an octave jump on the off-beats
        e_n = int(beat / 2 * sr)
        te = np.arange(e_n) / sr
        for e in range(8):
            f = root * (2 if e % 2 else 1)
            note = (np.sin(2 * np.pi * f * te) + 0.3 * np.sin(4 * np.pi * f * te)) * _env(e_n, sr, 0.005, 0.12)
            if not intro_outro:
                _place(bass, 0.35 * note, bar0 + e * e_n)
        # pad: soft saw chord for the whole bar
        b_n = int(bar * sr)
        tb = np.arange(b_n) / sr
        chord = sum(((f * tb) % 1.0 - 0.5) for f in triad) / 3
        chord = np.convolve(chord, np.ones(8) / 8, "same")                # soften the saw
        _place(pad, 0.18 * chord * _env(b_n, sr, 0.3, 3.0), bar0)
        # lead: 16th-note arpeggio over the triad, two octaves up
        s_n = int(beat / 4 * sr)
        ts = np.arange(s_n) / sr
        for s in range(16):
            f = 2 * triad[(s * 2 + (s // 4)) % 3]
            pluck = np.sign(np.sin(2 * np.pi * f * ts)) * 0.5 + np.sin(2 * np.pi * f * ts)
            if b >= 2 and not intro_outro:
                _place(lead, 0.06 * pluck * _env(s_n, sr, 0.002, 0.05), bar0 + s * s_n)

    out = drums + bass + pad + lead
    fade = int(min(2.0, duration / 4) * sr)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    out = np.tanh(1.2 * out) / np.tanh(1.2)                              # gentle limiter
    pcm = (np.clip(out, -1, 1) * 0.9 * 32767).astype(np.int16)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path
