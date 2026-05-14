"""
TTS 音频生成脚本（供 subprocess 调用）

用法（命令行）：
    python generate_audio.py --text "你好" --output out.wav --speaker Vivian --language Auto

或通过 Python API:
    from generate_audio import generate_audio
    result = generate_audio("你好", "out.wav", "Vivian", "Auto")

此脚本独立运行，不依赖主应用的 FastAPI 环境。
"""
from __future__ import annotations
from typing import Optional

import argparse
import os
import sys
import traceback

# ── Debug logging ──
_DBG_LOG = os.environ.get("TTS_DBG_LOG", "")
def _dbg(msg, *args):
    if _DBG_LOG:
        import time as _t
        with open(_DBG_LOG, "a", encoding="utf-8") as _f:
            _f.write(f"[{_t.strftime('%H:%M:%S')}] {msg % args}\n")

_dbg("generate_audio.py START | args=%s", sys.argv)
_dbg("ENV | TTS_FORCE_CPU=%s | TTS_MODEL_SIZE=%s | TTS_DBG_LOG=%s",
     os.environ.get("TTS_FORCE_CPU",""), os.environ.get("TTS_MODEL_SIZE",""), _DBG_LOG)

# Force offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MODELSCOPE_OFFLINE"] = "1"

# ── GPU 性能优化：默认禁用，需要显式启用 ──
_TTS_ENABLE_GPU_OPT = os.environ.get("TTS_ENABLE_GPU_OPT", "1") == "1"
if _TTS_ENABLE_GPU_OPT:
    import torch
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True

import gc

# 强制 CPU 模式（由 engine.py 通过环境变量传入）
_FORCE_CPU = os.environ.get("TTS_FORCE_CPU", "0") == "1"
if _FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── Log after torch import ──
try:
    import torch
    _dbg("generate_audio.py: torch imported OK | cuda=%s | force_cpu=%s | version=%s",
         torch.cuda.is_available(), _FORCE_CPU, torch.__version__)
except Exception as _e:
    _dbg("generate_audio.py: torch import FAILED: %s", _e)
    raise

# 禁用 SSL（仅在内网环境使用）
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

QWEN_TTS_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(QWEN_TTS_ROOT, "models")
PYTORCH_WINDOWS_PATH = os.path.join(QWEN_TTS_ROOT, "WPy64-312101", "python.exe")
SITE_PACKAGES = os.path.join(QWEN_TTS_ROOT, "WPy64-312101", "python", "Lib", "site-packages")

if SITE_PACKAGES not in sys.path:
    sys.path.insert(0, SITE_PACKAGES)

MODEL_PATHS_17B = {
    "custom_voice": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "base": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-Base"),
    "voice_design": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
}

MODEL_PATHS_06B = {
    "custom_voice": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
    "base": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-0.6B-Base"),
}

HUGGINGFACE_REPO_06B = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

CURRENT_MODEL_SIZE = os.environ.get("TTS_MODEL_SIZE", "1.7B")

MODEL_PATHS = MODEL_PATHS_17B if CURRENT_MODEL_SIZE == "1.7B" else MODEL_PATHS_06B

current_model = None
current_model_type = None


# ══════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════

def _is_gpu_usable(timeout_sec: int = 15) -> tuple[bool, str]:
    try:
        import threading
        result = {"ok": False, "reason": "unknown"}

        def _check():
            try:
                t = torch.tensor([1.0], device="cuda")
                _ = t + 1
                result["ok"] = True
            except Exception as e:
                result["reason"] = str(e)

        thread = threading.Thread(target=_check)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_sec)
        if thread.is_alive():
            return False, f"GPU 操作在 {timeout_sec}s 后仍无响应"
        if not result["ok"]:
            return False, f"GPU 操作失败: {result['reason']}"
        return True, "ok"
    except Exception as e:
        return False, f"GPU 检查异常: {e}"


def get_model(model_type: str = "custom_voice"):
    global current_model, current_model_type

    if current_model is not None and current_model_type == model_type:
        return current_model

    force_cpu = os.environ.get("TTS_FORCE_CPU", "").strip() in ("1", "true", "True")

    _dbg("get_model: START | model_type=%s | force_cpu=%s | CURRENT_MODEL_SIZE=%s", model_type, force_cpu, CURRENT_MODEL_SIZE)

    from qwen_tts import Qwen3TTSModel

    model_path = MODEL_PATHS.get(model_type, MODEL_PATHS["custom_voice"])
    if not os.path.exists(model_path):
        if model_type == "voice_design" and CURRENT_MODEL_SIZE == "0.6B":
            model_path = MODEL_PATHS_17B["voice_design"]
            print(f"[INFO] 0.6B VoiceDesign 不存在，降级使用 1.7B VoiceDesign")
        else:
            repo_id = HUGGINGFACE_REPO_06B.get(model_type)
            if repo_id:
                print(f"[INFO] 模型不存在，尝试从 HuggingFace 下载: {repo_id}")
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id=repo_id, local_dir=model_path)
                    print(f"[INFO] 下载完成: {model_path}")
                except Exception as e:
                    print(f"[WARN] 下载失败: {e}")
                    raise FileNotFoundError(f"模型路径不存在且无法下载: {model_path}")

    attn_mode = "sdpa"
    compute_dtype = torch.float32

    if torch.cuda.is_available() and not force_cpu:
        print(f"[INFO] GPU 可用，使用 fp32 推理（避免 bf16/fp16 NaN 问题）")
        compute_dtype = torch.float32

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    print(f"[INFO] 加载模型: {model_path}")

    if current_model is not None:
        print(f"[INFO] 切换模型类型: {current_model_type} -> {model_type}")
        del current_model
        current_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _dbg("get_model: GPU check | torch.cuda.is_available()=%s | force_cpu=%s", torch.cuda.is_available(), force_cpu)

    if force_cpu:
        gpu_available = False
        print(f"[INFO] 强制 CPU 模式 (TTS_FORCE_CPU=1)")
        _dbg("get_model: CPU forced by TTS_FORCE_CPU=1")
    elif torch.cuda.is_available():
        _dbg("get_model: GPU health check START")
        gpu_ok, gpu_reason = _is_gpu_usable(timeout_sec=15)
        _dbg("get_model: GPU health check DONE | gpu_ok=%s | reason=%s", gpu_ok, gpu_reason)
        print(f"[INFO] GPU 健康检查: {gpu_reason}")
        if not gpu_ok:
            print(f"[WARN] GPU 不可用，将使用 CPU 模式")
            gpu_available = False
        else:
            gpu_available = True
    else:
        gpu_available = False
        print(f"[INFO] CUDA 不可用，使用 CPU 模式")
        _dbg("get_model: CUDA not available, using CPU")

    _attempts = []
    _load_strategy = 0

    if gpu_available:
        _load_strategy = 1
        _dbg("get_model: Strategy 1 START (GPU device_map=auto)")
        try:
            print(f"[INFO] 策略1: 尝试 GPU 分片加载 (device_map=auto)")
            current_model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation=attn_mode,
                dtype=compute_dtype,
                local_files_only=True,
            )
            _dbg("get_model: Strategy 1 OK (GPU device_map=auto)")
            print(f"[INFO] GPU 加载成功 (device_map=auto)")
            current_model_type = model_type
            return current_model
        except Exception as _e:
            _attempts.append(("GPU auto", str(_e)))
            _dbg("get_model: Strategy 1 FAILED: %s", _e)
            print(f"[WARN] GPU device_map=auto 失败: {_e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if gpu_available:
        _load_strategy = 2
        _dbg("get_model: Strategy 2 START (GPU cuda:0, max_memory)")
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_mem_gb = min(3.5, total_mem * 0.45)
            print(f"[INFO] 策略2: 尝试 GPU 单卡映射 (max_memory={max_mem_gb:.1f}GB)")
            current_model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map="cuda:0",
                low_cpu_mem_usage=True,
                max_memory={0: f"{max_mem_gb:.1f}GB"},
                attn_implementation=attn_mode,
                dtype=compute_dtype,
                local_files_only=True,
            )
            _dbg("get_model: Strategy 2 OK (GPU cuda:0)")
            print(f"[INFO] GPU 单卡加载成功 (max_memory={max_mem_gb:.1f}GB)")
            current_model_type = model_type
            return current_model
        except Exception as _e:
            _attempts.append(("GPU cuda:0", str(_e)))
            _dbg("get_model: Strategy 2 FAILED: %s", _e)
            print(f"[WARN] GPU cuda:0 失败: {_e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _load_strategy = 3
    _dbg("get_model: Strategy 3 START (CPU fp32)")
    try:
        print(f"[INFO] 策略3: 尝试 CPU fp32 模式 (保守)")
        ncpu = os.cpu_count() or 4
        torch.set_num_threads(max(2, ncpu - 2))
        print(f"[INFO] 设置 torch 线程数: {torch.get_num_threads()}")
        import tempfile
        _tmpdir = tempfile.gettempdir()
        _old_env = os.environ.get("HF_HOME", "")
        os.environ["HF_HOME"] = _tmpdir
        current_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cpu",
            low_cpu_mem_usage=True,
            dtype=torch.float32,
            local_files_only=True,
        )
        os.environ["HF_HOME"] = _old_env
        _dbg("get_model: Strategy 3 OK (CPU fp32)")
        print(f"[INFO] CPU fp32 加载成功")
        current_model_type = model_type
        return current_model
    except Exception as _e:
        _attempts.append(("CPU-fp32", str(_e)))
        _dbg("get_model: Strategy 3 FAILED: %s", _e)
        print(f"[WARN] CPU fp32 失败: {_e}")
        gc.collect()

    _diag = "\n".join(f"  - {n}: {e}" for n, e in _attempts)
    raise RuntimeError(
        f"模型加载失败 (已尝试 {len(_attempts)} 种策略):\n{_diag}\n"
        f"建议: (1) 关闭其他占用内存的程序后再试; "
        f"(2) 增加 Windows 页面文件大小; "
        f"(3) 重启后先运行 TTS"
    )


def unload_model():
    global current_model
    if current_model is not None:
        print(f"[INFO] 释放模型...")
        del current_model
        current_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# 核心生成函数
# ══════════════════════════════════════════════════════════════

# 长文本分片阈值
# GPU 实测 25字 chunk 耗时 65 秒（固定开销），后续每增加 25 字额外增加约 47 秒
# 600s 超时：600 - 20(加载) - 10(写盘) = 570s 可用，570 / 47 ≈ 12 段 × 25 = 300 字上限
# 设 150 字：约 6 段 × 65 = 390s，安全余量充足
_MAX_TEXT_CHARS = 150


def _split_text(text: str) -> list[str]:
    """将长文本拆分为不超过 MAX_TEXT_CHARS 字符的片段（按句子边界拆分）"""
    import re
    # 按句子结束符拆分，捕获标点
    parts = re.split(r'([。！？\n]+)', text)
    chunks, current = [], ""

    for part in parts:
        if not part:
            continue
        # 标点附在前一句
        if re.match(r'^[。！？\n]+$', part):
            if current:
                current += part
        else:
            if current and len(current) + len(part) > _MAX_TEXT_CHARS:
                chunks.append(current)
                current = part
            elif not current:
                current = part
            else:
                current += part

    if current:
        chunks.append(current)
    return chunks or [text]


def _generate_one(model, text: str, output_path: str, speaker: str, language: str,
                  instruct: str, clone_audio: Optional[str], mode: str):
    """实际执行单段音频生成"""
    import soundfile as sf
    print(f"[INFO] 开始生成 | mode={mode} | text_len={len(text)} | language={language} | clone={clone_audio or '无'}")

    if mode == "voice_clone" and clone_audio:
        ref_audio, ref_sr = sf.read(clone_audio)
        if len(ref_audio.shape) > 1:
            ref_audio = ref_audio[:, 0]
        wavs, sr = model.generate_voice_clone(
            text=text, language=language, ref_audio=ref_audio,
            ref_text=None, instruct=instruct,
        )
    elif mode == "voice_design":
        wavs, sr = model.generate_voice_design(
            text=text, language=language, instruct=instruct,
        )
    else:
        wavs, sr = model.generate_custom_voice(
            text=text, language=language, speaker=speaker, instruct=instruct,
        )

    if hasattr(wavs[0], 'numpy'):
        audio_data = wavs[0].numpy()
    elif hasattr(wavs[0], 'cpu'):
        audio_data = wavs[0].cpu().numpy()
    else:
        audio_data = wavs[0]

    sf.write(output_path, audio_data, sr)
    duration = len(audio_data) / sr
    print(f"[INFO] 生成成功 | output={output_path} | sr={sr} | duration={duration:.2f}s")


def generate_audio(
    text: str,
    output_path: str,
    speaker: str = "Vivian",
    language: str = "Auto",
    instruct: str = "",
    clone_audio: Optional[str] = None,
    mode: str = "custom_voice",
) -> bool:
    chunks = _split_text(text)
    print(f"[INFO] 长文本分片: {len(chunks)} 段")
    _dbg("generate_audio: text split into %d chunks", len(chunks))

    import soundfile as sf
    import numpy as np
    all_audio, sr = [], None

    # 只加载一次模型，后续 chunk 复用（KV cache 自然累积不影响正确性）
    model = get_model(mode)
    _dbg("generate_audio: model loaded OK")

    for i, chunk_text in enumerate(chunks):
        print(f"[INFO] 生成片段 {i+1}/{len(chunks)}: {len(chunk_text)} 字符")
        _dbg("generate_audio: chunk=%d/%d text_len=%d", i+1, len(chunks), len(chunk_text))

        tmp_path = output_path + f".chunk{i}.wav"
        try:
            _generate_one(model, chunk_text, tmp_path, speaker, language, instruct, clone_audio, mode)
            seg_audio, seg_sr = sf.read(tmp_path)
            if seg_audio.ndim > 1:
                seg_audio = seg_audio[:, 0]
            all_audio.append(seg_audio.astype(np.float32))
            sr = seg_sr
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if all_audio:
        final = np.concatenate(all_audio)
        sf.write(output_path, final, sr)
        print(f"[INFO] 分片合并完成 | total_duration={len(final)/sr:.2f}s")
    return True


# ══════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS 音频生成")
    parser.add_argument("--text", type=str, required=True, help="待合成的文本")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--speaker", type=str, default="Vivian", help="音色名称")
    parser.add_argument("--language", type=str, default="Auto", help="语言")
    parser.add_argument("--style", type=str, default="", help="风格指令")
    parser.add_argument("--clone", type=str, default="", help="语音克隆参考音频路径")
    parser.add_argument("--mode", type=str, default="custom_voice",
                        choices=["custom_voice", "voice_clone", "voice_design"],
                        help="生成模式")

    args = parser.parse_args()

    try:
        mode = args.mode
        if args.clone:
            mode = "voice_clone"

        success = generate_audio(
            text=args.text,
            output_path=args.output,
            speaker=args.speaker,
            language=args.language,
            instruct=args.style,
            clone_audio=args.clone if args.clone else None,
            mode=mode,
        )
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"[ERROR] 生成失败: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    finally:
        unload_model()


if __name__ == "__main__":
    main()
