from modelscope.hub.snapshot_download import snapshot_download
import os

root_dir = os.getcwd()
models_to_download = [
    'Qwen/Qwen3-TTS-12Hz-1.7B-Base',
    'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
    'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
]

for model_id in models_to_download:
    folder_name = model_id.split('/')[-1]
    model_base_path = os.path.join(root_dir, "models", folder_name)
    
    if not os.path.exists(os.path.join(model_base_path, "config.json")):
        print(f"\n[正在下载] {model_id} ...")
        snapshot_download(model_id, local_dir=model_base_path)
    else:
        print(f"\n[跳过] {model_id} 已存在。")

print("\n[OK] 所有模型准备就绪！")