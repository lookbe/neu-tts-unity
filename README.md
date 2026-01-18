# NeuTTS for Unity (Open-Phonemizer Edition)

High-performance, on-device Text-to-Speech implementation for Unity. This version replaces `espeak` with an **ONNX-based phonemizer** pipeline and uses **NeuCodec** for high-fidelity audio reconstruction.

## 🚀 Key Features
* **Engine:** English-only synthesis.
* **Inference:** Fully local using GGUF for the backbone and ONNX for the codec and phonemizer.

---

## 📦 1. Required Files
Store these files in a permanent directory on your local machine (e.g., `C:/Models/NeuTTS/` or `/Users/Shared/Models/`).

### A. Synthesis & Codec (Neuphonic)
* **Backbone:** A Neuphonic GGUF model (e.g., `neutts-air-q4.gguf`).
* **Codec Decoder:** `decoder_model.onnx` (NeuCodec ONNX model).

### B. Phonemizer ([lookbe/open-phonemizer-onnx](https://huggingface.co/lookbe/open-phonemizer-onnx))
* `model.onnx`
* `tokenizer.json`
* `phoneme_dict.json`

### C. Voice Reference
* `sample.json` (Reference voice codes).
* `sample_transcript.txt` (Matching text for the reference voice).

---

## 🛠 2. Preparation: Reference Conversion
Unity cannot read PyTorch `.pt` files directly. Use this Python script to convert your reference voice codes to plain JSON before use:

```python
import torch
import json

def convert_pt_to_json(input_file, output_file):
    # Load the torch tensor
    data = torch.load(input_file, map_location='cpu')
    
    # Extract codes and convert to list for JSON compatibility
    if isinstance(data, torch.Tensor):
        data = data.detach().numpy().tolist()
    
    with open(output_file, 'w') as f:
        json.dump({"codes": data}, f)
    print(f"Success: {output_file} created.")
```
