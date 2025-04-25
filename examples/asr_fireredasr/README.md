## 🔥 FireRedASR-LLM 示例文档（中文）

### 🧩 简介
这是一个将 FireRedASR-LLM 模型集成进 SLAM 框架的完整示例，支持多机多卡训练。该示例展示了如何将一个**预训练且高集成化的模型**纳入 SLAM 流程，从而实现便捷的推理与训练工作流。

---

### 📁 项目结构说明
```
examples/asr_fireredasr/
├── conf/                  # 参数配置文件（JSON/YAML），如 prompt 设置、Deepspeed 参数等
│   ├── prompt.yaml
│   └── ds_config.json
│
├── script/                # shell 脚本：训练/推理入口
│   ├── decode_fireredasr.sh
│   └── finetune_fireredasr_deepspeed.sh
│
├── model/                 # 数据处理和模型加工厂定义
│   └── slam_fireredasr.py
│
├── fireredasr_config.py   # 主参数配置脚本
└── inference_fireredasr.py# 推理所需的额外参数脚本(同理你也可以设置训练所需的额外脚本)
```

---

### 📦 数据加载机制
针对小红书等模型使用自定义数据管道的问题，本项目支持通过 `get_dataset` 参数调用自定义的数据集函数。例如：
```bash
get_dataset=examples/asr_fireredasr/model/slam_fireredasr.py:get_speech_testwavdataset
```
该参数将传递至 SLAM 引擎，自动调用对应的数据处理逻辑，从而兼容原模型处理方式或自定义管道。

---

### 🧠 模型构建机制
模型的构建方式同样通过参数传递：
```bash
file=examples/asr_fireredasr/model/slam_fireredasr.py:model_factory
```
此项将指明模型构造函数在何处定义。你可以如 `aispeech_asr` 示例中细化不同组件，也可以如本例一样，调用一个集成好的整体模型加载函数。

---

### 🚀 接口统一要求
为兼容 SLAM 框架，`model_factory` 返回的模型实例应具备以下两个接口：
- `forward`：用于训练流程
- `generate`：用于推理流程

确保模型具备以上能力，即可无缝集成至 SLAM 的统一训练与推理逻辑中。

---
