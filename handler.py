#!/usr/bin/env python3
"""
Runpod Serverless handler — Foukenstein XTTS.

Port de /workspace/IA/bin/xtts_http_server.py vers la signature Runpod :
    def handler(event) -> dict

Le modèle est chargé UNE SEULE FOIS au scope module (hors de la fonction
handler) : Runpod garde le worker vivant entre invocations, et FlashBoot
snapshote cet état post-load → cold start ~10-20 s au lieu de ~30-60 s.

Contrat I/O — aligné sur deployment_v1/app/runpod_tts_client.py :

Input :
    {
      "input": {
        "chunks":     ["chunk 1", "chunk 2", ...],
        "language":   "fr",                # optionnel, défaut "fr"
        "out_format": "wav"                # "wav" | "mp3"
      }
    }

Output (ce que Runpod serialize comme `output`) :
    {
      "audio_base64": "UklGR...",
      "format":       "wav",
      "chunks":       <int>,
      "duration_ms":  <int>
    }
"""
from __future__ import annotations

import base64
import io
import os
import re
import time
import traceback

import torch
import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import runpod


# ── Config depuis l'env ──────────────────────────────────────────────────────
FTCKPT       = os.environ["FTCKPT"]          # /runpod-volume/xtts/best_model_19875.pth
ORIG         = os.environ["ORIG"]            # /runpod-volume/xtts/XTTS_v2.0_original_model_files
SPEAKERS_PTH = os.environ.get(
    "SPEAKERS_PTH",
    os.path.join(ORIG, "speakers_xtts.pth"),
)
SPEAKER_WAV  = os.environ["SPEAKER_WAV"]     # /runpod-volume/xtts/test_speaker.wav

CONFIG_PATH    = os.path.join(ORIG, "config.json")
TOKENIZER_PATH = os.path.join(ORIG, "vocab.json")
CKPT_DIR       = os.path.dirname(FTCKPT)

DEFAULT_LANG = os.environ.get("LANG", "fr").strip() or "fr"

SR = int(os.environ.get("SR", "24000"))
XTTS_HARD_LIMIT = 250  # limite dure XTTS pour le français (273), marge de sécurité


# ── Stitching params ─────────────────────────────────────────────────────────
# Logique actuelle :
#   - silence initial pour protéger le tout premier mot côté player/navigateur
#   - pré-silence avant chaque chunk APRÈS le premier
#   - fade-in appliqué après ajout du pré-silence, donc le fade-in tombe dans le silence
#   - fade-out court en fin de chunk pour lisser les raccords
#   - pas de crossfade long entre chunks, pour ne pas remanger les attaques
#
# Avant :
#   [chunk A] + [pause 220 ms] + [chunk B]
#
# Maintenant par défaut :
#   [chunk A] + [pause 150 ms] + [pré-silence 70 ms intégré au chunk B] + [chunk B]
#
# Durée totale entre deux paroles ≈ 220 ms, mais le début du chunk B est mieux protégé.

LEADING_SILENCE_MS = int(os.environ.get("LEADING_SILENCE_MS", "120"))

# Nouveau paramètre : silence ajouté au début de chaque chunk sauf le premier.
PRE_CHUNK_SILENCE_MS = int(os.environ.get("PRE_CHUNK_SILENCE_MS", "70"))

# Comme le fade-in tombe maintenant dans le pré-silence, on peut se permettre
# un fade-in un peu plus long sans manger l'attaque du mot.
CHUNK_FADE_IN_MS   = int(os.environ.get("CHUNK_FADE_IN_MS", "20"))
CHUNK_FADE_OUT_MS  = int(os.environ.get("CHUNK_FADE_OUT_MS", "18"))

XFADE_MS        = int(os.environ.get("XFADE_MS", "0"))
PAUSE_MS        = int(os.environ.get("PAUSE_MS", "150"))
MICRO_PAUSE_MS  = int(os.environ.get("MICRO_PAUSE_MS", "0"))
TAIL_SILENCE_MS = int(os.environ.get("TAIL_SILENCE_MS", "700"))
FADE_OUT_MS     = int(os.environ.get("FADE_OUT_MS", "40"))


# ── Split helpers ────────────────────────────────────────────────────────────
def _split_on_word_repetition(text: str, min_word_len: int = 6) -> list:
    cuts = [m.end() for m in re.finditer(r'[.!?…]\s+', text)]
    if not cuts:
        return [text]

    for cut in cuts:
        left  = text[:cut].strip()
        right = text[cut:].strip()

        if not left or not right:
            continue

        words_left = {
            w.lower()
            for w in re.findall(r'\b\w{%d,}\b' % min_word_len, left)
        }
        words_right = {
            w.lower()
            for w in re.findall(r'\b\w{%d,}\b' % min_word_len, right)
        }

        if words_left & words_right:
            return _split_on_word_repetition(left) + _split_on_word_repetition(right)

    return [text]


def _subsplit(text: str, limit: int = XTTS_HARD_LIMIT) -> list:
    if len(text) <= limit:
        return [text]

    parts = re.split(r'(?<=[.!?…])\s+', text)
    buf, result = "", []

    for p in parts:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= limit:
            buf += " " + p
        else:
            result.append(buf)
            buf = p

    if buf:
        result.append(buf)

    final = []
    for seg in result:
        while len(seg) > limit:
            cut = seg.rfind(" ", 0, limit)
            if cut == -1:
                cut = limit
            final.append(seg[:cut].strip())
            seg = seg[cut:].strip()

        if seg:
            final.append(seg)

    return final or [text[:limit]]


# ── Audio helpers ────────────────────────────────────────────────────────────
def add_silence(ms: int) -> torch.Tensor:
    n = max(0, int(SR * ms / 1000))
    return torch.zeros((1, n), dtype=torch.float32)


def edge_fade(wav: torch.Tensor, ms: int) -> torch.Tensor:
    """
    Ancienne fonction conservée pour compatibilité éventuelle/debug.
    Elle n'est plus utilisée dans le flux principal, car elle applique
    un fade-in et un fade-out symétriques.
    """
    if ms <= 0:
        return wav

    n = int(SR * ms / 1000)
    n = min(max(n, 0), wav.shape[1] // 2)

    if n <= 0:
        return wav

    wav = wav.clone()

    win_in = torch.linspace(
        0.0,
        1.0,
        steps=n,
        device=wav.device,
        dtype=wav.dtype,
    ).unsqueeze(0)

    win_out = torch.linspace(
        1.0,
        0.0,
        steps=n,
        device=wav.device,
        dtype=wav.dtype,
    ).unsqueeze(0)

    mask = torch.ones_like(wav)
    mask[:, :n] *= win_in
    mask[:, -n:] *= win_out

    return wav * mask


def chunk_fade(wav: torch.Tensor, fade_in_ms: int, fade_out_ms: int) -> torch.Tensor:
    """
    Fade séparé :
    - fade_in_ms lisse le début du chunk
    - fade_out_ms lisse la fin du chunk

    Important :
    dans cette version, pour les chunks après le premier, on ajoute d'abord
    PRE_CHUNK_SILENCE_MS au début du wav, puis on applique ce fade.
    Donc le fade-in tombe normalement dans le silence ajouté, pas sur la parole.

    Réglage actuel par défaut :
        PRE_CHUNK_SILENCE_MS=70
        CHUNK_FADE_IN_MS=20
        CHUNK_FADE_OUT_MS=18
    """
    if fade_in_ms <= 0 and fade_out_ms <= 0:
        return wav

    wav = wav.clone()

    if fade_in_ms > 0:
        n_in = int(SR * fade_in_ms / 1000)
        n_in = min(max(n_in, 0), wav.shape[1])

        if n_in > 0:
            win_in = torch.linspace(
                0.0,
                1.0,
                steps=n_in,
                device=wav.device,
                dtype=wav.dtype,
            ).unsqueeze(0)

            wav[:, :n_in] *= win_in

    if fade_out_ms > 0:
        n_out = int(SR * fade_out_ms / 1000)
        n_out = min(max(n_out, 0), wav.shape[1])

        if n_out > 0:
            win_out = torch.linspace(
                1.0,
                0.0,
                steps=n_out,
                device=wav.device,
                dtype=wav.dtype,
            ).unsqueeze(0)

            wav[:, -n_out:] *= win_out

    return wav


def prepend_silence(wav: torch.Tensor, ms: int) -> torch.Tensor:
    """
    Ajoute un silence au début d'un chunk audio.
    Utilisé pour les chunks après le premier, avant chunk_fade().
    """
    if ms <= 0:
        return wav

    silence = add_silence(ms).to(wav.device, wav.dtype)
    return torch.cat([silence, wav], dim=1)


def equal_power_crossfade(a: torch.Tensor, b: torch.Tensor, fade_samples: int) -> torch.Tensor:
    if fade_samples <= 0:
        return torch.cat([a, b], dim=1)

    fade_samples = min(fade_samples, a.shape[1], b.shape[1])

    if fade_samples <= 0:
        return torch.cat([a, b], dim=1)

    a1 = a[:, :-fade_samples]
    a2 = a[:, -fade_samples:]
    b1 = b[:, :fade_samples]
    b2 = b[:, fade_samples:]

    t = torch.linspace(
        0.0,
        1.0,
        steps=fade_samples,
        device=a.device,
        dtype=a.dtype,
    ).unsqueeze(0)

    wa = torch.cos(t * 0.5 * torch.pi)
    wb = torch.sin(t * 0.5 * torch.pi)

    mixed = a2 * wa + b1 * wb

    return torch.cat([a1, mixed, b2], dim=1)


def add_leading_silence(wav: torch.Tensor, ms: int) -> torch.Tensor:
    if ms <= 0:
        return wav

    return torch.cat([add_silence(ms).to(wav.device, wav.dtype), wav], dim=1)


def add_tail_silence(wav: torch.Tensor, ms: int) -> torch.Tensor:
    if ms <= 0:
        return wav

    return torch.cat([wav, add_silence(ms).to(wav.device, wav.dtype)], dim=1)


def fade_out(wav: torch.Tensor, ms: int) -> torch.Tensor:
    if ms <= 0:
        return wav

    n = int(SR * ms / 1000)
    n = min(max(n, 0), wav.shape[1])

    if n <= 0:
        return wav

    wav = wav.clone()

    win = torch.linspace(
        1.0,
        0.0,
        steps=n,
        device=wav.device,
        dtype=wav.dtype,
    ).unsqueeze(0)

    mask = torch.ones_like(wav)
    mask[:, -n:] *= win

    return wav * mask


# ── Chargement modèle, une seule fois au scope module ────────────────────────
def _must_exist(p: str) -> None:
    if not p or not os.path.exists(p):
        raise FileNotFoundError(p)


print("[handler] validating asset paths...", flush=True)

for _p in (CONFIG_PATH, TOKENIZER_PATH, FTCKPT, SPEAKERS_PTH, SPEAKER_WAV):
    _must_exist(_p)

print("[handler] loading XTTS model once (module scope)...", flush=True)

_t0 = time.time()

_config = XttsConfig()
_config.load_json(CONFIG_PATH)

MODEL = Xtts.init_from_config(_config)

MODEL.load_checkpoint(
    _config,
    checkpoint_dir=CKPT_DIR,
    checkpoint_path=FTCKPT,
    vocab_path=TOKENIZER_PATH,
    speaker_file_path=SPEAKERS_PTH,
    use_deepspeed=False,
)

if torch.cuda.is_available():
    MODEL.cuda()

MODEL.eval()

GPT_COND_LATENT, SPEAKER_EMBEDDING = MODEL.get_conditioning_latents(
    audio_path=[SPEAKER_WAV]
)

FADE_SAMPLES = int(SR * XFADE_MS / 1000)

print(
    f"[handler] model ready in {time.time() - _t0:.1f}s "
    f"(device={'cuda' if torch.cuda.is_available() else 'cpu'}, sr={SR}, "
    f"leading_silence_ms={LEADING_SILENCE_MS}, "
    f"pre_chunk_silence_ms={PRE_CHUNK_SILENCE_MS}, "
    f"chunk_fade_in_ms={CHUNK_FADE_IN_MS}, "
    f"chunk_fade_out_ms={CHUNK_FADE_OUT_MS}, "
    f"xfade_ms={XFADE_MS}, pause_ms={PAUSE_MS}, "
    f"micro_pause_ms={MICRO_PAUSE_MS}, tail_silence_ms={TAIL_SILENCE_MS}, "
    f"fade_out_ms={FADE_OUT_MS})",
    flush=True,
)


# ── Synthèse ─────────────────────────────────────────────────────────────────
def _synthesize(chunks: list[str], language: str) -> tuple[torch.Tensor, int]:
    original_chunks_count = len(chunks)

    chunks = [str(c) for c in chunks if str(c).strip()]

    if not chunks:
        raise ValueError("empty chunks")

    expanded = []

    for c in chunks:
        for part in _split_on_word_repetition(c):
            expanded.extend(_subsplit(part))

    chunks = expanded

    pieces = []

    for idx, txt in enumerate(chunks, start=1):
        if len(txt) > XTTS_HARD_LIMIT:
            txt = txt[:XTTS_HARD_LIMIT]

        with torch.inference_mode():
            out = MODEL.inference(
                text=txt,
                language=language,
                gpt_cond_latent=GPT_COND_LATENT,
                speaker_embedding=SPEAKER_EMBEDDING,
                temperature=0.65,
                length_penalty=1.0,
                repetition_penalty=2.0,
                top_k=50,
                top_p=0.85,
            )

            wav_raw = out["wav"]

            if isinstance(wav_raw, torch.Tensor):
                wav = wav_raw.float()
            else:
                wav = torch.from_numpy(wav_raw).float()

            if wav.ndim == 1:
                wav = wav.unsqueeze(0)

        # Nouveau :
        # Pour tous les chunks après le premier, on ajoute un silence AVANT la parole.
        # Ce silence est ajouté AVANT le fade-in, donc le fade-in ne mange pas l'attaque.
        if idx > 1 and PRE_CHUNK_SILENCE_MS > 0:
            wav = prepend_silence(wav, PRE_CHUNK_SILENCE_MS)

        # Fade-in/fade-out du chunk.
        # Sur les chunks après le premier, le fade-in tombe dans le pré-silence.
        wav = chunk_fade(wav, CHUNK_FADE_IN_MS, CHUNK_FADE_OUT_MS)

        pieces.append(("audio", wav))

        if MICRO_PAUSE_MS > 0:
            pieces.append(("pause", add_silence(MICRO_PAUSE_MS)))

        if idx < len(chunks):
            pieces.append(("pause", add_silence(PAUSE_MS)))

    final = None

    for kind, w in pieces:
        if final is None:
            final = w
            continue

        w = w.to(final.device, final.dtype)

        if kind == "pause":
            final = torch.cat([final, w], dim=1)
        else:
            final = equal_power_crossfade(final, w, FADE_SAMPLES)

    final = add_leading_silence(final, LEADING_SILENCE_MS)
    final = add_tail_silence(final, TAIL_SILENCE_MS)
    final = fade_out(final, FADE_OUT_MS)

    return final.cpu(), len(chunks)


def _to_wav_base64(wav: torch.Tensor) -> tuple[str, int]:
    # wav shape: (1, N) float32
    samples = wav.squeeze(0).numpy()

    # Sécurité anti-clipping légère avant export PCM16.
    peak = abs(samples).max() if len(samples) else 0.0
    if peak > 0.98:
        samples = samples / peak * 0.98

    buf = io.BytesIO()

    sf.write(
        buf,
        samples,
        SR,
        format="WAV",
        subtype="PCM_16",
    )

    data = buf.getvalue()
    duration_ms = int(1000 * len(samples) / SR)

    return base64.b64encode(data).decode("ascii"), duration_ms


# ── Runpod entrypoint ────────────────────────────────────────────────────────
def handler(event):
    try:
        payload = (event or {}).get("input") or {}

        chunks = payload.get("chunks")

        if not isinstance(chunks, list) or not chunks:
            return {"error": "missing or empty 'chunks'"}

        language = str(payload.get("language") or DEFAULT_LANG).strip() or "fr"

        out_format = str(payload.get("out_format") or "wav").lower()

        if out_format != "wav":
            # mp3 pas supporté pour l'instant — le client fallback sur wav
            out_format = "wav"

        t0 = time.time()

        wav, n_chunks = _synthesize(chunks, language=language)

        b64, duration_ms = _to_wav_base64(wav)

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "audio_base64": b64,
            "format": "wav",
            "chunks": n_chunks,
            "duration_ms": duration_ms,
            "elapsed_ms": elapsed_ms,
            "stitching": {
                "sr": SR,
                "leading_silence_ms": LEADING_SILENCE_MS,
                "pre_chunk_silence_ms": PRE_CHUNK_SILENCE_MS,
                "chunk_fade_in_ms": CHUNK_FADE_IN_MS,
                "chunk_fade_out_ms": CHUNK_FADE_OUT_MS,
                "xfade_ms": XFADE_MS,
                "pause_ms": PAUSE_MS,
                "micro_pause_ms": MICRO_PAUSE_MS,
                "tail_silence_ms": TAIL_SILENCE_MS,
                "fade_out_ms": FADE_OUT_MS,
            },
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
