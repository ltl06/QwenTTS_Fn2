import os
import sys
# 【核心修复】强制离线
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['MODELSCOPE_OFFLINE'] = '1'

# 【核心修复】禁用 SSL 验证
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import gc
import torch
import gradio as gr
from qwen_tts import Qwen3TTSModel

# --- 显存与模型调度逻辑 ---
current_model = None
current_type = None

# 获取项目真正的根目录 (假设脚本在根目录或者 bin 目录下)
def get_project_root():
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 如果脚本在 bin 文件夹里，向上走一级
    if os.path.basename(current_path).lower() == 'bin':
        return os.path.dirname(current_path)
    return current_path

ROOT_DIR = get_project_root()

# 统一处理路径：转绝对路径 + 反斜杠转正斜杠
def get_safe_path(rel_path):
    abs_path = os.path.join(ROOT_DIR, rel_path)
    return abs_path.replace("\\", "/")

MODEL_PATHS = {
    "声音设计": get_safe_path("models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    "语音克隆": get_safe_path("models/Qwen3-TTS-12Hz-1.7B-Base"),
    "自定义音色": get_safe_path("models/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
}

def unload_model():
    global current_model, current_type
    if current_model is not None:
        print(f"[显存管理] 正在释放: {current_type} 模型...")
        del current_model
        current_model = None
        current_type = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def load_model(m_type):
    global current_model, current_type
    if current_type == m_type:
        return current_model
    
    unload_model()
    path = MODEL_PATHS[m_type]
    
    # --- [核心：Flash-Attention 自动识别逻辑] ---
    # 默认设置
    attn_mode = "sdpa"
    compute_dtype = torch.float16
    
    # 1. 检查 BF16 支持 (RTX 30/40/50系列)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        
    # 2. 检查 Flash-Attention 插件
    try:
        import flash_attn
        attn_mode = "flash_attention_2"
        print(f"[硬件加速] 检测到 Flash-Attention，已开启高性能模式。")
    except ImportError:
        print(f"[硬件加速] 未检测到加速插件，使用标准 SDPA 模式。")
    # ------------------------------------------

    print(f"[显存管理] 正在加载: {m_type} 模型...")
    
    current_model = Qwen3TTSModel.from_pretrained(
        path,
        device_map="cuda",
        attn_implementation=attn_mode, # 动态识别结果
        dtype=compute_dtype,           # 动态识别结果
        local_files_only=True
    )
    current_type = m_type
    return current_model

# --- 核心生成函数 ---
def fn_voice_design(text, lang, instruct):
    model = load_model("声音设计")
    if model is None: return None, "错误：找不到模型文件，请检查 models 目录"
    wavs, sr = model.generate_voice_design(text=text, language=lang, instruct=instruct)
    return (sr, wavs[0]), "生成成功！"

def fn_voice_clone(text, lang, ref_audio, ref_text):
    model = load_model("语音克隆")
    if model is None: return None, "错误：找不到模型文件"
    
    # --- 【关键逻辑：动态切换模式】 ---
    # 如果用户没有填参考文本，或者只填了空格
    if not ref_text or str(ref_text).strip() == "":
        use_x_vector = True
        status_msg = "克隆成功！(模式：零样本音色克隆)"
        print("[系统] 参考文本为空，已自动切换至 x_vector 模式。")
    else:
        use_x_vector = False
        status_msg = "克隆成功！(模式：ICL 高质量克隆)"

    try:
        wavs, sr = model.generate_voice_clone(
            text=text, 
            language=lang, 
            ref_audio=ref_audio, 
            ref_text=ref_text,
            x_vector_only_mode=use_x_vector  # 传入动态判断的结果
        )
        return (sr, wavs[0]), status_msg
    except Exception as e:
        return None, f"克隆失败: {str(e)}"

def fn_custom_voice(text, lang, speaker, instruct):
    model = load_model("自定义音色")
    if model is None: return None, "错误：找不到模型文件，请检查 models 目录"
    wavs, sr = model.generate_custom_voice(text=text, language=lang, speaker=speaker, instruct=instruct)
    return (sr, wavs[0]), "生成成功！"

# --- UI 构建 ---
def run_integrated(ip, port):
    langs = ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto"]
    speakers = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]

    with gr.Blocks(title="Qwen3-TTS 生活作弊码懒人包") as demo:
        gr.Markdown("# Qwen3-TTS 全能懒人包集成版")
        gr.Markdown("""
        ### 🚀 功能说明：
        - **🎨 声音设计 (Voice Design)**：通过自然语言描述（如“甜美的萝莉音”）直接创建定制化音色。
        - **👥 语音克隆 (Voice Clone)**：基于一段参考音频和文本，完美复刻目标人物的声音。
        - **🌟 精品音色 (CustomVoice)**：使用官方预设的高质量说话人，支持愤怒、开心等多种情感控制。
        
        💡 **提示**：强烈建议先访问https://qwen.ai/blog?id=qwen3tts-0115 ，这里有详细的演示和解释。
        
        - 本工具由阿里巴巴通义实验室 Qwen 团队研发，本项目已集成三个1.7B模型。切换标签页时会自动释放旧模型，建议显存8G及以上。
        """)
        
        with gr.Tabs() as tabs:
            with gr.Tab("声音设计 (Voice Design)", id="声音设计"):
                with gr.Row():
                    with gr.Column():
                        txt_in = gr.Textbox(label="文本", lines=3, value="哥哥，你回来啦，人家等了你好久好久了，要抱抱！")
                        lang_in = gr.Dropdown(langs, label="语言", value="Auto")
                        ins_in = gr.Textbox(label="声音描述", placeholder="例如：甜美的萝莉音", lines=2)
                        btn_gen = gr.Button("开始生成", variant="primary")
                    with gr.Column():
                        aud_out = gr.Audio(label="生成音频")
                        msg_out = gr.Textbox(label="状态")
                btn_gen.click(fn_voice_design, [txt_in, lang_in, ins_in], [aud_out, msg_out])

            with gr.Tab("语音克隆 (Voice Clone)", id="语音克隆"):
                with gr.Row():
                    with gr.Column():
                        txt_in_c = gr.Textbox(label="文本", lines=3, value="你好，很高兴见到你。")
                        lang_in_c = gr.Dropdown(langs, label="语言", value="Auto")
                        ref_aud = gr.Audio(label="参考音频", type="filepath")
                        ref_txt = gr.Textbox(label="参考文本", placeholder="参考音频里的台词")
                        btn_gen_c = gr.Button("开始克隆", variant="primary")
                    with gr.Column():
                        aud_out_c = gr.Audio(label="生成音频")
                        msg_out_c = gr.Textbox(label="状态")
                btn_gen_c.click(fn_voice_clone, [txt_in_c, lang_in_c, ref_aud, ref_txt], [aud_out_c, msg_out_c])

            with gr.Tab("自定义音色 (Custom Voice)", id="自定义音色"):
                with gr.Row():
                    with gr.Column():
                        txt_in_v = gr.Textbox(label="文本", lines=3, value="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
                        lang_in_v = gr.Dropdown(langs, label="语言", value="Auto")
                        spk_in = gr.Dropdown(speakers, label="音色", value="Vivian")
                        ins_in_v = gr.Textbox(label="风格", placeholder="例如：用愤怒的语气")
                        btn_gen_v = gr.Button("开始合成", variant="primary")
                    with gr.Column():
                        aud_out_v = gr.Audio(label="生成音频")
                        msg_out_v = gr.Textbox(label="状态")
                btn_gen_v.click(fn_custom_voice, [txt_in_v, lang_in_v, spk_in, ins_in_v], [aud_out_v, msg_out_v])

        tabs.select(fn=lambda evt: load_model(evt.value))

    cert_path = os.path.abspath("cert.pem")
    key_path = os.path.abspath("key.pem")

    demo.launch(
        server_name=ip,
        server_port=int(port),
        ssl_certfile=cert_path,
        ssl_keyfile=key_path,
        ssl_verify=False,      # 告诉 Gradio 内部不要验证 SSL
        show_error=True,
        quiet=True
    )

if __name__ == "__main__":
    # 默认启动第一个模型
    load_model("声音设计")
    run_integrated("127.0.0.1", "7860")