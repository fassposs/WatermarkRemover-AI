from huggingface_hub import snapshot_download
# 使用示例
if __name__ == "__main__":
    # 可以增加这一句加速下载
    # export HF_ENDPOINT=https://hf-mirror.com
    snapshot_download(repo_id="microsoft/Florence-2-large", repo_type="model",local_dir="./models/Florence-2-large")
    print(111)
