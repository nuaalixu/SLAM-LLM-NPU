
## 🔥 FireRedASR-LLM Example (English)

### 🧩 Introduction
This is an example of integrating the FireRedASR-LLM model into the SLAM framework. It supports **multi-node and multi-GPU training**, demonstrating how to plug a **pretrained and well-integrated model** into SLAM to enable efficient inference and training workflows.

---

### 📁 Project Structure
```
examples/asr_fireredasr/
├── conf/                  # Configuration files (JSON/YAML): prompt, deepspeed settings
│   ├── prompt.yaml
│   └── ds_config.json
│
├── script/                # Shell scripts for training and inference
│   ├── decode_fireredasr.sh
│   └── finetune_fireredasr_deepspeed.sh
│
├── model/                 # Contains model factory and dataset logic
│   └── slam_fireredasr.py
│
├── fireredasr_config.py   # Main configuration file
└── inference_fireredasr.py# Supplementary config for inference or you can also add config for finetuning
```

---

### 📦 Data Loading Support
To support custom pipelines like those used in Xiaohongshu-style models, this example utilizes SLAM's `get_dataset` parameter:
```bash
get_dataset=examples/asr_fireredasr/model/slam_fireredasr.py:get_speech_testwavdataset
```
This parameter will be passed into the SLAM engine, which will automatically locate and execute the specified dataset function—allowing you to reuse original or custom data handling logic.

---

### 🧠 Model Construction Logic
The model instantiation is defined through a script argument:
```bash
file=examples/asr_fireredasr/model/slam_fireredasr.py:model_factory
```
This links SLAM to the correct function for model construction. You can define modular components (like `aispeech_asr`) or use a pre-integrated loading function (like in this example).

---

### 🚀 Interface Compatibility
To ensure smooth integration, your model returned by `model_factory` **must implement**:
- `forward` (for training)
- `generate` (for inference)

As long as these two methods are present, your model is fully SLAM-compatible.