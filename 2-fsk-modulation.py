#!/usr/bin/env python3
"""
audio_modem.py — Encode text into digital audio and decode it back.

Modulation: 2-FSK (two-tone). Each bit is a sine at f0 or f1 for a fixed bit duration.
Sync: search for best sample offset that maximizes preamble match via Goertzel energy.
Framing: [PREAMBLE (64 bits 0xAA)] [LEN (32 bits)] [PAYLOAD (LEN bytes)] [CRC32 (32 bits)]

WAV: 44.1 kHz, 16-bit PCM, mono.
"""

import argparse
import math
import struct
import wave
import zlib
from typing import List, Tuple

import numpy as np

# -------- Parameters --------
SAMPLE_RATE = 44100
BIT_DURATION_S = 0.05        # 50 ms per bit => 20 bps (slow but robust). You can reduce to ~0.01 for faster.
FREQ0 = 1200.0               # Hz for bit 0
FREQ1 = 2200.0               # Hz for bit 1
AMPLITUDE = 0.9              # Peak amplitude (float), will be scaled to int16
PREAMBLE_BYTE = 0xAA         # 10101010
PREAMBLE_BITS = 64           # length of preamble in bits
LEADING_SILENCE_S = 0.2      # silence before preamble, helps with playback capture chains
TRAILING_SILENCE_S = 0.2

# -------- Bit/byte helpers --------
def bytes_to_bits(data: bytes) -> List[int]:
    bits = []
    for b in data:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    return bits

def bits_to_bytes(bits: List[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("bit length not multiple of 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        v = 0
        for j in range(8):
            v = (v << 1) | (bits[i + j] & 1)
        out.append(v)
    return bytes(out)

def u32_to_bits(u: int) -> List[int]:
    return [(u >> (31 - i)) & 1 for i in range(32)]

def bits_to_u32(bits: List[int]) -> int:
    if len(bits) != 32:
        raise ValueError("need exactly 32 bits")
    v = 0
    for b in bits:
        v = (v << 1) | (b & 1)
    return v

# -------- Signal generation --------
def tone_for_bit(bit: int, num_samples: int, sr: int) -> np.ndarray:
    freq = FREQ1 if bit else FREQ0
    t = np.arange(num_samples) / sr
    return AMPLITUDE * np.sin(2 * math.pi * freq * t)

def synth_from_bits(bits: List[int], sr: int = SAMPLE_RATE) -> np.ndarray:
    bit_samples = int(round(BIT_DURATION_S * sr))
    chunks = []
    # leading silence
    if LEADING_SILENCE_S > 0:
        chunks.append(np.zeros(int(LEADING_SILENCE_S * sr), dtype=np.float32))
    # bits
    for b in bits:
        chunks.append(tone_for_bit(b, bit_samples, sr).astype(np.float32))
    # trailing silence
    if TRAILING_SILENCE_S > 0:
        chunks.append(np.zeros(int(TRAILING_SILENCE_S * sr), dtype=np.float32))
    signal = np.concatenate(chunks)
    # light fade-in/out to avoid clicks
    fade = int(0.005 * sr)
    if fade > 0 and len(signal) > 2 * fade:
        window = np.linspace(0, 1, fade, dtype=np.float32)
        signal[:fade] *= window
        signal[-fade:] *= window[::-1]
    return signal

def write_wav(path: str, signal: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    # clip and convert to int16
    sig = np.clip(signal, -1.0, 1.0)
    pcm = (sig * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

def read_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        nchan = wf.getnchannels()
        assert nchan == 1, "Only mono WAV supported"
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        assert sampwidth == 2, "Only 16-bit PCM supported"
        nframes = wf.getnframes()
        data = wf.readframes(nframes)
    pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sr

# -------- Goertzel: tone energy in a block --------
def goertzel_power(block: np.ndarray, sr: int, freq: float) -> float:
    # classic Goertzel
    k = int(0.5 + (len(block) * freq) / sr)
    omega = (2.0 * math.pi * k) / len(block)
    coeff = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for x in block:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    return power

# -------- Framing (preamble + len + payload + crc) --------
def build_frame(payload: bytes) -> List[int]:
    # preamble: 0xAA repeated (10101010)
    pre_bits = bytes_to_bits(bytes([PREAMBLE_BYTE] * (PREAMBLE_BITS // 8)))
    length_bits = u32_to_bits(len(payload))
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    crc_bits = u32_to_bits(crc)
    bits = pre_bits + length_bits + bytes_to_bits(payload) + crc_bits
    return bits

def parse_frame(bits: List[int]) -> bytes:
    # expects: after preamble
    # first 32 bits = length, then payload bytes, then 32 bits CRC
    if len(bits) < 32 + 32:
        raise ValueError("bitstream too short")
    length = bits_to_u32(bits[:32])
    payload_bits = bits[32:32 + 8 * length]
    if len(payload_bits) != 8 * length:
        raise ValueError("incomplete payload")
    payload = bits_to_bytes(payload_bits)
    crc_bits = bits[32 + 8 * length: 32 + 8 * length + 32]
    if len(crc_bits) != 32:
        raise ValueError("missing CRC")
    rx_crc = bits_to_u32(crc_bits)
    calc_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if rx_crc != calc_crc:
        raise ValueError(f"CRC mismatch: expected {calc_crc:08X}, got {rx_crc:08X}")
    return payload

# -------- Decoder --------
def detect_best_offset(samples: np.ndarray, sr: int, bit_samples: int, preamble_bits: List[int]) -> int:
    """
    Try all sample offsets in [0, bit_samples) and score how well the preamble matches.
    We compute Goertzel energies per bit window and compare to expected bits.
    """
    best_score = -1e18
    best_offset = 0
    need = len(preamble_bits) * bit_samples
    if len(samples) < need + bit_samples:
        # pad with zeros for scoring
        samples = np.pad(samples, (0, need + bit_samples - len(samples)))
    for offset in range(0, bit_samples):
        score = 0.0
        idx = offset
        for b in preamble_bits:
            win = samples[idx: idx + bit_samples]
            e0 = goertzel_power(win, sr, FREQ0)
            e1 = goertzel_power(win, sr, FREQ1)
            # add positive score if expected tone has higher energy
            if b == 0:
                score += (e0 - e1)
            else:
                score += (e1 - e0)
            idx += bit_samples
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_offset

def bits_from_signal(samples: np.ndarray, sr: int) -> List[int]:
    # normalize
    if np.max(np.abs(samples)) > 0:
        samples = samples / np.max(np.abs(samples))
    bit_samples = int(round(BIT_DURATION_S * sr))

    # preamble bits pattern
    pre_bits = bytes_to_bits(bytes([PREAMBLE_BYTE] * (PREAMBLE_BITS // 8)))

    # find best offset for bit windows
    offset = detect_best_offset(samples, sr, bit_samples, pre_bits)

    # now slice windows and decide bits by comparing energy at f0 vs f1
    bits = []
    idx = offset
    total_bits_est = (len(samples) - offset) // bit_samples
    for _ in range(total_bits_est):
        win = samples[idx: idx + bit_samples]
        if len(win) < bit_samples:
            break
        e0 = goertzel_power(win, sr, FREQ0)
        e1 = goertzel_power(win, sr, FREQ1)
        bits.append(1 if e1 > e0 else 0)
        idx += bit_samples
    return bits, offset

def find_preamble_index(bits: List[int]) -> int:
    # we search for the start index where the preamble pattern begins
    pre = bytes_to_bits(bytes([PREAMBLE_BYTE] * (PREAMBLE_BITS // 8)))
    L = len(pre)
    # simple search allowing exact match – with real audio you might allow Hamming distance tolerance
    for i in range(0, len(bits) - L + 1):
        if bits[i:i + L] == pre:
            return i + L  # return index just after preamble
    # fallback: approximate search with small tolerance
    best_i = -1
    best_match = -1
    for i in range(0, len(bits) - L + 1):
        match = sum(1 for a, b in zip(bits[i:i+L], pre) if a == b)
        if match > best_match:
            best_match = match
            best_i = i
    # require at least 90% match
    if best_match >= 0.9 * L:
        return best_i + L
    raise ValueError("preamble not found")

# -------- API --------
def encode_text_to_wav(text: str, out_path: str) -> None:
    payload = text.encode("utf-8")
    bits = build_frame(payload)
    signal = synth_from_bits(bits, SAMPLE_RATE)
    write_wav(out_path, signal, SAMPLE_RATE)

def decode_wav_to_text(wav_path: str) -> str:
    samples, sr = read_wav(wav_path)
    bits, _offset = bits_from_signal(samples, sr)
    start = find_preamble_index(bits)
    # attempt to parse frame from start onward
    tail = bits[start:]
    payload = parse_frame(tail)
    return payload.decode("utf-8")

# -------- CLI --------
def main():
    global BIT_DURATION_S
    p = argparse.ArgumentParser(description="Encode/decode text via 2-FSK audio")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="encode text to WAV")
    pe.add_argument("--text", required=True, help="text to encode")
    pe.add_argument("--out", required=True, help="output WAV path")
    pe.add_argument("--bitrate", type=float, default=1.0 / BIT_DURATION_S, help="bits per second (default ~20)")

    pd = sub.add_parser("decode", help="decode WAV to text")
    pd.add_argument("--wav", required=True, help="input WAV path")

    args = p.parse_args()

    if args.cmd == "encode":
        if args.bitrate <= 0:
            raise SystemExit("bitrate must be > 0")
        BIT_DURATION_S = 1.0 / args.bitrate
        encode_text_to_wav(args.text, args.out)
        print(f"Wrote {args.out}")
    elif args.cmd == "decode":
        text = decode_wav_to_text(args.wav)
        print(text)

if __name__ == "__main__":
    main()
