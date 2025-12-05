import easyocr
import torch
from PIL import Image
import numpy as np
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor

# imgPath = './testInput/test003.jpg'
# reader = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory
# result = reader.readtext('./testInput/test003.jpg')


# print(result)
def remove_watermark_with_qwen(img_path, output_path):
    """
    使用Qwen-VL模型处理图片水印

    Args:
        img_path: 输入图片路径
        output_path: 输出图片路径
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. 加载Qwen-VL模型和处理器
    model_name = "Qwen/Qwen-VL-Chat"
    model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name,trust_remote_code=True)

    # model_name = "microsoft/Florence-2-large"
    # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    # processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # 2. 加载图片
    image = Image.open(img_path).convert("RGB")

    # 3. 构造提示词
    prompt = "请识别这张图片中水印的位置，并以bounding box坐标形式返回"

    # 4. 处理输入
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 5. 模型推理
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        # pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )

    # 6. 后处理结果
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]

    # 6. 后处理结果
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return response
    # 7. 保存处理后的图片
    # 注意：Qwen-VL主要输出文本描述，实际图片编辑需结合其他方法
    print(f"模型响应: {response}")

    # 如果模型返回了处理后的图片，保存它
    # 这里只是一个示例，实际实现可能需要调整
    image.save(output_path)
    print(f"图片已保存到: {output_path}")


# 使用示例
if __name__ == "__main__":
    imgPath = './testInput/test003.jpg'
    output_path = './testOutput/test003_clean.jpg'
    remove_watermark_with_qwen(imgPath, output_path)
