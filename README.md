# NeuTTS for Unity (Open-Phonemizer Edition)

High-performance, on-device Text-to-Speech implementation for Unity. This version replaces `espeak` with an **ONNX-based phonemizer** pipeline and uses **NeuCodec** for high-fidelity audio reconstruction.

## 🚀 Key Features
* **Engine:** English-only synthesis.
* **Inference:** Fully local using GGUF for the backbone and ONNX for the codec and phonemizer.

---

## Installation

Follow these steps exactly to ensure all native dependencies are resolved.

### Configure manifest.json
Open your project's `Packages/manifest.json` and update it to include the scoped registry and the Git dependencies.


```json
{
  "scopedRegistries": [
    {
      "name": "npm",
      "url": "https://registry.npmjs.com",
      "scopes": [
        "com.github.asus4"
      ]
    }
  ],
  "dependencies": {
    "com.github.asus4.onnxruntime": "0.4.2",
    "com.github.asus4.onnxruntime.unity": "0.4.2",
    "ai.lookbe.llamacpp": "https://github.com/lookbe/llama-cpp-unity.git",
    "ai.lookbe.neutts": "https://github.com/lookbe/neu-tts-unity.git",

    ... other dependencies
  }
}
```

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

## 🖥️ How to Run

1.  **Organize Files:** Place all required models and configuration files in a dedicated folder on your hard drive (outside of the Unity project).
2.  **Open Scene:** Open the `BasicNeuTTS` scene in the Unity Editor.
3.  **Find NeuTTS Object:** Select the **NeuTTS** GameObject in the Hierarchy window.
4.  **Input Absolute Paths:** In the Inspector, enter the full system path for each required file:
    * **Windows Example:** `C:\Users\Name\Models\NeuTTS\neutts-air-q4.gguf`
    * **macOS/Linux Example:** `/Users/Name/Models/NeuTTS/decoder_model.onnx`
    * *Note: Do not use relative paths or StreamingAssets syntax. Ensure the path points exactly to where the file is stored on your disk.*
5.  **Press Play:** Click the **Play** button in Unity. The system will load the models directly from the specified absolute locations and initialize the synthesis pipeline.

---
