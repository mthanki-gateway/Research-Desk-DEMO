"""A second opinion on the browser's Whisper log-mel.

WHY THIS SCRIPT EXISTS

`frontend/app/parley/mel.ts` reimplements `WhisperFeatureExtractor` in
TypeScript, because Smart Turn v3 takes mel features rather than a waveform and
there is no feature extractor in the browser. That reimplementation FAILS
SILENTLY when it is wrong: a filterbank off by a scale factor, a periodic
window where a symmetric one was meant, HTK's mel formula instead of Slaney's --
none of them throw. They produce a correctly shaped tensor of plausible
numbers, the model returns confident probabilities, and turn detection becomes
a coin flip that nobody can see is broken.

So the same spec is implemented here, independently, in numpy. Agreement does
not prove either is what PyTorch does -- both are mine -- but it catches the
coding mistakes, which is the failure mode that actually happens. The
behavioural check that the model separates finished speech from interrupted
speech lives in `frontend/app/parley/verifyTurn.mjs`.

Run:  docker compose exec api python -m app.scripts.mel_reference
"""

from __future__ import annotations

import json

import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 400
HOP = 160
N_MELS = 80
WINDOW_SAMPLES = 8 * SAMPLE_RATE
N_FRAMES = WINDOW_SAMPLES // HOP


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """Slaney: linear below 1kHz, logarithmic above."""
    hz = np.asarray(hz, dtype=np.float64)
    min_log_hz, min_log_mel = 1000.0, 15.0
    step = np.log(6.4) / 27.0
    return np.where(
        hz >= min_log_hz,
        min_log_mel + np.log(np.maximum(hz, 1e-9) / min_log_hz) / step,
        3.0 * hz / 200.0,
    )


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    min_log_hz, min_log_mel = 1000.0, 15.0
    step = np.log(6.4) / 27.0
    return np.where(
        mel >= min_log_mel,
        min_log_hz * np.exp(step * (mel - min_log_mel)),
        200.0 * mel / 3.0,
    )


def mel_filters() -> np.ndarray:
    n_bins = N_FFT // 2 + 1
    points = mel_to_hz(
        np.linspace(hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2)
    )
    bin_hz = np.arange(n_bins) * SAMPLE_RATE / N_FFT

    bank = np.zeros((N_MELS, n_bins), dtype=np.float64)
    for m in range(N_MELS):
        left, centre, right = points[m], points[m + 1], points[m + 2]
        rising = (bin_hz - left) / (centre - left)
        falling = (right - bin_hz) / (right - centre)
        bank[m] = np.maximum(0.0, np.minimum(rising, falling)) * (2.0 / (right - left))
    return bank


def log_mel(audio: np.ndarray) -> np.ndarray:
    """The [80, 800] tensor, from the LAST 8 seconds, left-padded."""
    # Last 8 seconds, then RIGHT-padded -- `truncate_audio_to_last_n_seconds`
    # followed by `padding="max_length"`, which pads on the right.
    samples = np.zeros(WINDOW_SAMPLES, dtype=np.float64)
    take = min(len(audio), WINDOW_SAMPLES)
    samples[:take] = audio[len(audio) - take :]

    # do_normalize=True, over the REAL SAMPLES ONLY, with the padding written
    # back to zero afterwards -- see `zero_mean_unit_var_norm`, which
    # normalises `vector[:length]` and then restores `padding_value`.
    real = samples[:take]
    samples[:take] = (real - real.mean()) / np.sqrt(real.var() + 1e-7)
    samples[take:] = 0.0

    # Symmetric Hann, reflect-padded, centred -- the same three choices the
    # TypeScript makes, spelled out rather than borrowed from a library so the
    # two are genuinely independent.
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT))
    padded = np.pad(samples, N_FFT // 2, mode="reflect")

    frames = np.stack(
        [padded[i * HOP : i * HOP + N_FFT] * window for i in range(N_FRAMES)]
    )
    spectrum = np.fft.rfft(frames, n=512, axis=1)[:, : N_FFT // 2 + 1]
    power = (spectrum.real**2 + spectrum.imag**2).T  # [bins, frames]

    mel = mel_filters() @ power
    logged = np.log10(np.maximum(mel, 1e-10))
    logged = np.maximum(logged, logged.max() - 8.0)
    return ((logged + 4.0) / 4.0).astype(np.float32)


def test_audio() -> np.ndarray:
    """Deterministic and broadband, so every filter has something in it.

    A chirp rather than a tone: a single frequency lights up two mel bins and
    would agree between two implementations that disagree everywhere else.

    The filler is TRIGONOMETRIC, not a random generator. numpy's PCG64 cannot
    be reproduced in JavaScript, and a different noise realisation is not a
    cosmetic difference here: the tensor is clamped 8 decades below its loudest
    bin, so a quieter filler pushes more cells onto the floor and shifts the
    mean of the whole thing. That read as a mel bug for a while, and was not
    one. sin/cos agree between the two languages to within double precision.
    """
    i = np.arange(3 * SAMPLE_RATE, dtype=np.float64)
    t = i / SAMPLE_RATE
    sweep = np.sin(2 * np.pi * (80 + (3800 - 80) * t / t[-1]) * t)
    envelope = np.sin(np.pi * t / t[-1]) ** 2
    filler = 0.02 * np.sin(i * 0.9137) * np.cos(i * 1.7231)
    return (sweep * envelope + filler).astype(np.float32)


def main() -> None:
    audio = test_audio()
    mel = log_mel(audio)

    out = {
        "shape": list(mel.shape),
        "mean": float(mel.mean()),
        "std": float(mel.std()),
        "min": float(mel.min()),
        "max": float(mel.max()),
        # POSITIONAL aggregates, not single cells. Individual cells mostly sit
        # on the clamp floor -- the tensor is cut 8 decades below its loudest
        # bin -- so probing six of them compared six identical floors and
        # would have agreed however wrong the code was.
        #
        # A per-row mean catches a wrong mel scale or filterbank, a per-frame
        # mean catches a wrong window or hop, and the argmax catches a
        # transpose. None of them can be satisfied by a floor.
        "row_means": {str(m): float(mel[m].mean()) for m in (0, 20, 40, 60, 79)},
        "col_means": {str(f): float(mel[:, f].mean()) for f in (0, 520, 600, 700, 799)},
        "argmax": [int(x) for x in np.unravel_index(int(mel.argmax()), mel.shape)],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
