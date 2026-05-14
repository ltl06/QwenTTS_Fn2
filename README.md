# QwenTTS_Fn2

基于阿里通义千问 **Qwen3-TTS** 的文本转语音集成工具，提供命令行和 Python API 两种调用方式，支持音色选择、语音克隆和声音设计三种合成模式。

## 功能特性

- **9 种预置音色**：涵盖中文、英文、日文、韩文等多语种
- **语音克隆**：上传 5~30 秒参考音频，克隆任意音色
- **声音设计**：用自然语言描述想要的音色特征
- **多语言支持**：支持中文、英文、日语、韩语等 11 种语言
- **多模型支持**：0.6B 轻量模型 / 1.7B 高质量模型
- **自动分片**：长文本自动拆分为多个片段并拼接
- **多后端适配**：GPU 自动推理，支持 CPU 回退

## 目录结构

```
QwenTTS_Fn2/
├── generate_audio.py           # TTS 音频生成脚本（命令行/API）
├── integrated_app.py          # Gradio WebUI 集成界面
├── clean_launch.py            # WebUI 启动器（自动打开浏览器）
├── make_cert.py              # SSL 证书生成工具
├── bin/                      # 音频处理工具
│   ├── ffmpeg.exe           # FFmpeg（需从系统 PATH 或自行下载）
│   ├── ffplay.exe            # FFplay
│   ├── ffprobe.exe           # FFprobe
│   └── sox.exe              # SoX 音频处理
├── models/                   # Qwen3-TTS 模型文件（需单独下载）
├── Qwen3-TTS/              # Qwen3-TTS 官方 SDK（含训练/推理代码）
├── WPy64-312101/           # 嵌入式 Python 运行环境（可选，也可使用标准 Python）
├── pkgs/                    # 预装 Python 包（重新 pip install 即可）
└── README.md
```

## 环境要求

- **Python**: 3.10+（标准 Python 环境即可，不需要 WPy64-312101）
- **GPU**: NVIDIA GPU（推荐 RTX 3060 以上）
- **系统**: Windows / Linux
- **磁盘空间**: 10GB+（用于模型文件）

### 环境准备

#### 方式一：使用现有 Python（推荐）

如果你的系统已经有 Python 3.10+，可以直接使用，不需要 WPy64-312101：

```bash
# 安装依赖
pip install torch soundfile numpy transformers accelerate huggingface_hub

# 如果需要 WebUI 界面
pip install gradio
```

#### 方式二：使用嵌入式 Python（WPy64-312101）

如果系统没有 Python，可以使用附带的嵌入式 Python（位于 WPy64-312101/ 目录）。运行前需将其加入 PATH。

#### FFmpeg 准备

本项目的 `generate_audio.py` 本身不依赖 ffmpeg（使用 Python 的 soundfile 库处理音频），但如果需要视频处理或格式转换，可从以下地址获取 ffmpeg：

- 官网：https://ffmpeg.org/download.html
- Windows 一键包：https://www.ghesc.xyz/ffmpeg.html
- 放入 `bin/` 目录或加入系统 PATH

## 模型下载

模型文件未包含在代码仓库中，需手动下载：

| 模型 | 说明 | 大小 | 下载地址 |
|------|------|------|---------|
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 自定义音色模型（默认） | ~2GB | [HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) |
| Qwen3-TTS-12Hz-0.6B-Base | 基础模型 | ~2GB | [HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 大尺寸自定义音色 | ~6GB | [ModelScope](https://www.modelscope.cn/models) |

下载后将模型文件夹放入 `models/` 目录：

```
QwenTTS_Fn2/models/Qwen3-TTS-12Hz-0.6B-CustomVoice/
QwenTTS_Fn2/models/Qwen3-TTS-12Hz-0.6B-Base/
```

## 快速开始

### 命令行使用

```bash
# 音色模式（使用预置音色）
python generate_audio.py --text "你好，欢迎使用Qwen3-TTS" --output output.wav --speaker Vivian --language Auto

# 克隆模式（使用参考音频）
python generate_audio.py --text "你好" --output output.wav --speaker Vivian --language Auto --clone ref_voice.wav

# 声音设计模式（用文字描述音色）
python generate_audio.py --text "你好" --output output.wav --mode voice_design --style "温柔的女声，语速稍慢" --language Auto
```

### Python API 使用

```python
from generate_audio import generate_audio

# 基础用法
result = generate_audio(
    text="你好，欢迎使用Qwen3-TTS",
    output_path="output.wav",
    speaker="Vivian",
    language="Auto"
)

# 语音克隆
result = generate_audio(
    text="你好",
    output_path="output.wav",
    speaker="Vivian",
    language="Auto",
    clone_audio="reference.wav",
    mode="voice_clone"
)

# 声音设计
result = generate_audio(
    text="你好",
    output_path="output.wav",
    mode="voice_design",
    instruct="活泼开朗的年轻女声",
    language="Auto"
)
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--text` | string | 必填 | 待合成的文本内容，最大 10000 字 |
| `--output` | string | 必填 | 输出 WAV 文件路径 |
| `--speaker` | string | Vivian | 预置音色名称 |
| `--language` | string | Auto | 语言：`Auto` / `Chinese` / `English` / `Japanese` 等 |
| `--style` | string | "" | 风格指令（声音设计模式使用） |
| `--clone` | string | "" | 语音克隆参考音频路径 |
| `--mode` | string | custom_voice | 合成模式：`custom_voice` / `voice_clone` / `voice_design` |

### 预置音色列表

| ID | 名称 | 语言 | 推荐场景 |
|----|------|------|---------|
| Vivian | Vivian | 中文 | 甜美女声，客服讲解 |
| Serena | Serena | 中文 | 知性女声，商务演示 |
| Uncle_Fu | Uncle_Fu | 中文 | 福叔声线，方言讲解 |
| Dylan | Dylan | 中文 | 京片子男生，趣味内容 |
| Eric | Eric | 中文 | 四川方言 |
| Ryan | Ryan | 中英混合 | 磁性男声 |
| Aiden | Aiden | 英文 | 英文男声 |
| Ono_Anna | Ono_Anna | 日文 | 日文女声 |
| Sohee | Sohee | 韩文 | 韩文女声 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TTS_MODEL_SIZE` | 1.7B | 模型大小：`1.7B` 或 `0.6B` |
| `TTS_FORCE_CPU` | 0 | 强制使用 CPU：`0`（GPU）或 `1`（CPU） |
| `TTS_ENABLE_GPU_OPT` | 1 | 启用 GPU 优化：`0` 或 `1` |
| `TTS_DBG_LOG` | "" | 调试日志文件路径 |

示例：

```bash
# 使用 0.6B 轻量模型
set TTS_MODEL_SIZE=0.6B
python generate_audio.py ...

# 强制使用 CPU
set TTS_FORCE_CPU=1
python generate_audio.py ...
```

## 模型加载策略

脚本自动尝试以下加载策略（按优先级）：

1. **GPU device_map=auto** — 尝试自动分片加载到 GPU
2. **GPU cuda:0 单卡** — 限制显存使用量（≤ 3.5GB 或显存 45%）
3. **CPU fp32** — 保守模式，自动回退到 CPU

## 长文本处理

文本超过 150 字时自动拆分为多个片段，分别合成后再拼接。拆分以句子边界（`。！？\n`）为准。

## 与 DigitalHumanMVP 集成

本项目作为 DigitalHumanMVP 的 TTS 引擎使用。在 DigitalHumanMVP 的 `config.yaml` 中配置：

```yaml
QWEN_TTS_ROOT: "D:/hecheng/QwenTTS_Fn2"  # 指向本项目目录
```

## 常见问题

### Q: 启动报错 "模型路径不存在"

确保 `models/` 目录下有对应的模型文件夹，且文件夹名称与代码中的 `MODEL_PATHS` 完全一致。

### Q: 显存不足（OOM）

- 设置 `TTS_MODEL_SIZE=0.6B` 使用轻量模型
- 设置 `TTS_FORCE_CPU=1` 强制使用 CPU

### Q: 生成的音频有杂音或截断

长文本分段合成时，每段独立生成后拼接。如遇截断，可能是文本分段不合理，可手动缩短每段文本长度。

### Q: GPU 检测失败但实际可用

检查 NVIDIA 驱动和 CUDA 是否正确安装：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 技术栈

- TTS 引擎：Qwen3-TTS (Apache-2.0)
- 深度学习框架：PyTorch
- 音频处理：soundfile
- 模型加载：transformers + accelerate

## License

本项目代码遵循 MIT License。Qwen3-TTS 模型遵循 Apache-2.0 License。
