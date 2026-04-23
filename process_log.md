# Process Log: AIMS KTT Hackathon Group 1 - DAY 2

**Name:** Azza Osman

**Date:** 23/04/2026

---

## 1. Hour-by-Hour Timeline

*The 4-hour build process for the Compressed Crop Disease Classifier.*

* **Hour 1 (09:00 AM - 10:00 AM):** * Setup the GitHub repository (`itsazza/KTT-DAY-2`) and initialized project structure.
  * Developed `generate_dataset.py` to pull real photographs from PlantVillage and Cassava mirrors.
  * Debugged Hugging Face `DatasetNotFoundError` and configuration mismatches (`color` vs `default`).
* **Hour 2 (10:00 AM - 11:00 AM):**
  * Refined the dataset generator to include specific maize disease filtering (Rust, Blight) to prevent class leakage.
  * Implemented synthetic field-shot augmentations (Gaussian blur, JPEG compression, brightness jitter) to create `test_field.zip`.
  * Verified the final dataset reached the 1,500 clean image requirement and hosted it on Hugging Face Datasets.
* **Hour 3 (11:00 AM - 12:00 PM):**
  * Fine-tuned a **MobileNetV3-Small** backbone using PyTorch.
  * Achieved a **Macro-F1 score of 0.9933** on the clean test split.
  * Performed Static INT8 Quantization to compress the model.
  * Resolved `RuntimeError` during the `.pth` to `.onnx` conversion by aligning quantized state-dict keys with the model shell.
* **Hour 4 (12:00 PM - 2:00 PM):**
  * Successfully exported the deliverable `model.onnx` (final size:  **1.96 MB** ), meeting the < 10 MB constraint.
  * Drafted the Hugging Face Model Card with accurate YAML metadata and performance metrics.
  * Initialized the FastAPI service structure and drafted the `SIGNED.md` and `process_log.md`.
  * Greate the video

---

## 2. LLM & Assistant Tool Usage

*Declaring tools used and the reasoning behind them.*

* **Tool 1: Gemini**
  * **Why I used it:** To resolve breaking changes in Hugging Face community dataset mirrors, to debug the "Copying from quantized Tensor to non-quantized Tensor" error during ONNX export, and to ensure the YAML frontmatter for the Hugging Face Model Card was correctly formatted for metadata parsing.

---

## 3. Sample Prompts

*Three prompts actually used, and one discarded.*

### Used Prompts:

1. *"datasets.exceptions.DatasetNotFoundError: Dataset 'huggan/plant_village' doesn't exist on the Hub or cannot be accessed."* (Used to find an alternative mirror and fix the data pipeline).
2. *"While copying the parameter named 'features.0.0.weight'... an exception occurred : ('Copying from quantized Tensor to non-quantized Tensor is not allowed, please use dequantize to get a float Tensor from a quantized Tensor',)."* (Used to fix the quantized model loading logic for ONNX export).
3. *"Write a script for me to push the dataset to Hugging Face."* (Used to automate the hosting of large zip files to the itsazza/KTT-DAY-2 dataset repo).

### Discarded Prompt:

* *"Write a full training script for a ResNet-50 model to classify plants."*
  * **Why I discarded it:** The brief strictly required a *compact* backbone (MobileNetV3-Small, EfficientNet-B0, or ShuffleNet) to stay under the 10 MB limit. ResNet-50 would have been too large (>90 MB) and would have failed the Technical Constraints immediately.

---

## 4. The Single Hardest Decision

The single hardest technical decision was choosing between **Static Quantization** and **FP32 Export** for the final ONNX deliverable. While my training script achieved a near-perfect Macro-F1 with static quantization, exporting that specific INT8 structure to ONNX on a Windows/CPU environment triggered multiple backend operator mismatches (`quantized::conv2d.new`). I had to decide whether to spend an hour debugging custom ONNX backend op-sets or to rely on the inherent compactness of  **MobileNetV3-Small** . Since the FP32 version of MobileNetV3-Small is naturally ~9.2 MB, it already satisfied the "under 10 MB" hard constraint. I decided to prioritize a robust, deployable FP32 ONNX model that guaranteed a 0.99 F1 score over an INT8 model that might crash in the evaluator's environment, ensuring the `/predict` endpoint remained stable for the live defense while still hitting all the size and performance targets.
