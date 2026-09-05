"""下载 Qwen3-8B-FP8 模型(ModelScope,国内网络可达)。

用法: python scripts/download_model.py
"""

from modelscope import snapshot_download

path = snapshot_download("Qwen/Qwen3-8B-FP8",
                         local_dir="models/Qwen3-8B-FP8")
print(f"模型已就绪: {path}")
