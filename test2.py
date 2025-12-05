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

    # 1. 加载Qwen-VL模型和处理器
    model_name = "Qwen/Qwen-VL-Chat"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 2. 加载图片
    image = Image.open(img_path).convert("RGB")

    # 3. 构造提示词
    prompt = "请移除这张图片中的水印，保持图片内容完整清晰"

    # 4. 处理输入
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    # 5. 模型推理
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
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

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

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
