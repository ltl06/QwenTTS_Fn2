import os
import sys

# --- 【暴力补丁：必须在最前面】 ---
# 1. 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['MODELSCOPE_OFFLINE'] = '1'
os.environ['PYTHONHTTPSVERIFY'] = '0'

# 2. 强行修改底层 SSL 验证函数
import ssl
try:
    # 这一步直接把全局默认的 SSL 验证替换成“不验证”
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# 3. 针对 Gradio 和 Requests 等库的额外补丁
import warnings
warnings.filterwarnings("ignore")
# -------------------------------

import asyncio
import logging
import threading
import webbrowser
import time
import socket

def silent_exception_handler(loop, context):
    exception = context.get('exception')
    message = context.get('message', '')
    if isinstance(exception, ConnectionResetError) or "10054" in str(exception):
        return
    if "_call_connection_lost" in message or "ProactorBasePipeTransport" in message:
        return
    loop.default_exception_handler(context)

def auto_open_browser(ip, port):
    url = f"https://{ip}:{port}"
    print(f"[系统] 正在等待引擎加载，完成后将自动弹出网页...")
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((ip, int(port))) == 0: break
            if sock.connect_ex((ip, 7860)) == 0:
                url = f"https://{ip}:7860"; break
        time.sleep(1)
    time.sleep(2)
    print(f"[系统] 服务就绪，正在打开浏览器...")
    webbrowser.open(url)

def start_app():
    ip, port = "127.0.0.1", "7860"
    for i, arg in enumerate(sys.argv):
        if arg == "--ip": ip = sys.argv[i+1]
        if arg == "--port": port = sys.argv[i+1]

    threading.Thread(target=auto_open_browser, args=(ip, port), daemon=True).start()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(silent_exception_handler)

    try:
        import integrated_app
        integrated_app.run_integrated(ip, port)
    except Exception as e:
        #日志
        import traceback
        print("\n[ERROR] 致命错误详情:")
        traceback.print_exc()
        input("\n按任意键退出...")

if __name__ == "__main__":
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    os.system("") 
    start_app()