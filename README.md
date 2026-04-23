# 🌿 Edge-AI Crop Disease Classifier (KTT-T2.1)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/itsazza/KTT-DAY2-Model-mobilenet_v3_small)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/itsazza/KTT-DAY-2)

This repository contains the complete submission for the **AIMS KTT Hackathon Tier 2.1 Challenge**. The project delivers a highly compressed, high-performance crop disease classifier designed for rural farmers in Rwanda and beyond.

---

## 📊 Performance Summary

- **Backbone:** MobileNetV3-Small (Fine-tuned)
- **Clean Test Macro-F1:** **0.9933** (Target: $\ge 80\%$)
- **Robustness Bonus (Field):** **0.8810** (Drop < 12 points)
- **Final Model Size:** **1.96 MB** (Target: < 10 MB)
- **Format:** ONNX (INT8 Quantized)
- **Inference Latency:** ~15ms (CPU-only)

---

## 🛠️ Reproduction Steps (≤ 2 Commands)

To reproduce the prediction service on a free Colab CPU or local machine:

**1. Install Dependencies**

```bash
pip install -r requirements.txt
```
