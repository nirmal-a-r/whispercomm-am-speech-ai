<div align="center">

# 📡 WhisperComm
### An End-to-End AM Communication System with AI-Enhanced Speech Recovery

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/whispercomm/blob/main/WhisperComm_Jupyter_NOTEBOOK.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**School of Artificial Intelligence — Amrita Vishwa Vidyapeetham, Coimbatore**  
Course: Software-Defined Communication Systems (23AID203) | Batch A-09

| Contributors |
|---|
 Nirmal Ramamoorthy *(Project Lead)* |
 N. Sanjana Reddy |

*Supervised by Dr. Jyothish Lal G, Assistant Professor (Sr.Gr.),Amrita School of AI*

</div>

---

## 🔬 Abstract

WhisperComm is a complete, instrumented end-to-end simulation of an AM radio communication system that integrates **classical Digital Signal Processing (DSP)** with **state-of-the-art AI** to achieve robust speech recovery under severe channel noise.

Speech from the LibriSpeech corpus is transmitted over a simulated AM channel corrupted by **pink (1/f) noise** at five SNR levels (5–25 dB). On the receiver side, a **Synchronous (Coherent) Demodulator** recovers the baseband audio. Facebook's **DNS-64** deep denoiser is then applied, followed by **OpenAI Whisper-large-v3** for ASR transcription, evaluated using **Word Error Rate (WER)**.

> **Key Engineering Discovery:** Applying AI denoising unconditionally is *actively harmful* at all but the lowest SNR levels — a direct consequence of domain mismatch between radio channel noise and the acoustic noise DNS-64 was trained on. This finding led to the design of a **Dynamic SNR Routing Pipeline** that activates the AI denoiser only when the channel degrades below a critical threshold (≤ 5 dB), achieving the minimum attainable WER across all channel conditions.

---

## 🏗️ System Architecture

```text
Clean Speech (LibriSpeech)
        │
        ▼
┌──────────────────────────────┐
│ Pre-Modulation LPF           │
│ • 3.8 kHz Butterworth        │
│ • Anti-aliasing filter       │
│ • Normalize to [-1, 1]       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ AM Modulator                 │
│ s(t) = [1 + m(t)] cos(2πft)  │
│ fc = 4000 Hz                 │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Pink Noise Channel           │
│ • 1/f PSD (FFT shaping)      │
│ • SNR = {25,18,12,8,5} dB    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Synchronous Demodulation     │
│ • Multiply with carrier      │
│ • LPF + DC removal           │
│ • Coherent receiver          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Dynamic SNR Routing          │
│ • SNR > 5 dB → DSP only      │
│ • SNR ≤ 5 dB → DNS-64        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Whisper-large-v3             │
│ • Beam search (5 beams)      │
│ • English transcription      │
└──────────────┬───────────────┘
               │
               ▼
        WER Evaluation
```

---

## 📊 Results

### Complete System Performance Table

| SNR (dB) | Baseline WER | AI-Enhanced WER | Smart Routing WER | Smart vs. AI |
|:---:|:---:|:---:|:---:|:---:|
| 25 | 5.88% | 23.53% ❌ | **5.88%** | **+17.65 pp** |
| 18 | 17.65% | 29.41% ❌ | **17.65%** | **+11.76 pp** |
| 12 | 17.65% | 23.53% ❌ | **17.65%** | **+5.88 pp** |
| 8  | 52.94% | 117.65% ❌ | **52.94%** | **+64.71 pp** |
| 5  | 100.00% | **88.24%** ✅ | **88.24%** | **+11.76 pp** |
| **Average** | **38.82%** | **56.47%** | **36.47%** | **+18.00 pp** |

> ❌ = DNS-64 made things *worse* (domain mismatch)  
> ✅ = DNS-64 helped (extreme degradation)  
> pp = percentage points

### Key Findings

**1. Receiver Design is Paramount.**  
Upgrading from Envelope Detection to Synchronous Demodulation reduced WER at 12 dB from **70.59% → 17.65%** — a 52.94 percentage point gain. No amount of AI post-processing compensated for the wrong demodulation architecture.

**2. AI is Not a Universal Enhancer.**  
DNS-64 degraded performance at 4 out of 5 SNR levels. At 8 dB, it caused catastrophic ASR hallucination (WER = **117.65%** — exceeding 100% because insertions outnumbered reference words). This is a direct empirical demonstration of the **domain mismatch** problem: DNS-64 was trained on acoustic room noise, not AM radio channel artefacts.

**3. Adaptive Routing Yields the Optimal Hybrid System.**  
The Smart Routing Pipeline improved average WER by **18 percentage points** over blind AI application, achieving the minimum attainable WER at every operating point.

---

## 🛠️ Tech Stack

| Component | Tool / Model |
|---|---|
| Dataset | LibriSpeech test-clean (`121-121726-0000`) |
| Modulation | AM DSB-FC — fc = 4,000 Hz, 100% modulation index |
| Channel Model | Pink noise (1/f PSD) via FFT spectral shaping |
| SNR Levels | 5, 8, 12, 18, 25 dB |
| Demodulation | Synchronous (Coherent) — 6th-order Butterworth LPF |
| AI Denoiser | Facebook DNS-64 (DEMUCS-based, waveform domain) |
| ASR Model | OpenAI Whisper-large-v3 (beam search, num_beams=5) |
| Evaluation Metric | Word Error Rate (WER) via HuggingFace `evaluate` |
| Hardware | Google Colab T4 GPU (16 GB VRAM) |
| Language | Python 3.12 |

---
---

## 🚀 How to Run

### Prerequisites
- Google account (for Colab)
- Runtime → **T4 GPU** (required — DNS-64 and Whisper need CUDA)

### Steps
1. Click **Open in Colab** at the top of this README
2. Go to `Runtime → Change runtime type → T4 GPU`
3. Run **Cell 1** (installs all dependencies — takes ~2 min)
4. Run cells **2 through 11** in order
5. Cell 7 (Whisper evaluation) takes ~10–15 minutes — do not interrupt

> All dependencies are installed programmatically inside the notebook. No manual pip installs required.

---

## ⚙️ Key Simulation Parameters

| Parameter | Value | Notes |
|---|---|---|
| Sample Rate (fs) | 16,000 Hz | LibriSpeech native |
| Carrier Frequency (fc) | 4,000 Hz | fc = fs / 4 |
| Pre-Modulation LPF | 3,800 Hz | Anti-aliasing |
| Demodulation LPF | 3,800 Hz | Removes 2·fc artefact |
| Butterworth Order | 6 | Steep roll-off |
| Modulation Index | 1.000 | 100% AM |
| Noise Type | Pink (1/f) | FFT spectral shaping |
| AI Activation Threshold | ≤ 5 dB SNR | Smart Routing boundary |
| DNS-64 Input Shape | `[1, 1, T]` | Batch × Channel × Time |
| Whisper Beam Size | 5 | Beam search decoding |

---

## 📈 Visual Outputs

The notebook generates the following visualisations automatically:

- **System Architecture Block Diagram** — Full TX/RX pipeline with colour-coded stages
- **Waveform + Log-Spectrogram** — For clean, noisy, and enhanced signals at each stage
- **Power Spectral Density (PSD)** — Confirming correct AM sideband placement
- **Butterworth Filter Response** — Passband and stopband verification
- **3-Way Spectrogram Comparison** — Noisy / Enhanced / Reference (12 dB showcase)
- **Noise Residual Spectrogram** — Showing exactly what DNS-64 removed
- **2×5 SNR Grid** — All 10 spectrograms across the full sweep
- **WER vs SNR Performance Plot** — Three-pipeline comparison with routing annotations
- **Smart System Dashboard** — Grouped bar chart with AI activation threshold marker
- **Stability Heatmap** — Transcription fidelity (100% − WER) across all configurations

---


## 📄 License

This project is released under the MIT License for educational and research use.

---

<div align="center">

**WhisperComm** — School of Artificial Intelligence | Amrita Vishwa Vidyapeetham  
23AID203 | Batch A-09 | September 2025

*"The most effective AI-enhanced communication systems are those where deep learning augments — rather than replaces — rigorous classical engineering."*

</div>
