import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
from loguru import logger
from enum import Enum

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray


class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""


def identify(task_prompt: TaskType, image: MatLike, text_input: str, model: AutoModelForCausalLM, processor: AutoProcessor, device: str):
    """
    根据给定的任务类型、图像和文本输入，使用指定的模型和处理器进行推理并返回处理结果。

    参数:
        task_prompt (TaskType): 任务类型枚举值，用于确定要执行的具体任务
        image (MatLike): 输入的图像数据，用于视觉相关的任务处理
        text_input (str): 额外的文本输入，可与任务提示结合使用
        model (AutoModelForCausalLM): 预训练的因果语言模型，用于生成响应
        processor (AutoProcessor): 模型处理器，负责处理文本和图像输入
        device (str): 运行设备标识符（如'cpu'或'cuda'）

    返回:
        处理后的生成结果，具体格式取决于任务类型和后处理逻辑

    异常:
        ValueError: 当task_prompt不是TaskType实例时抛出
    """
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    # 构造完整的提示文本，如果提供了额外文本输入则将其附加到任务提示后面
    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input

    # 使用处理器将文本提示和图像转换为模型输入张量
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 执行模型推理生成响应
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    # 解码生成的token ID为文本，并进行后处理
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )


# 获取图片的水印区域
def get_watermark_mask(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float):
    # text_input = "watermark"
    text_input = "waterlogo"
    task_prompt = TaskType.OPEN_VOCAB_DETECTION
    # 识别图像中与“watermark”相关的边界框
    parsed_answer = identify(task_prompt, image, text_input, model, processor, device)

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    detection_key = "<OPEN_VOCABULARY_DETECTION>"
    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        image_area = image.width * image.height
        for bbox in parsed_answer[detection_key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            if (bbox_area / image_area) * 100 <= max_bbox_percent:
                draw.rectangle([x1, y1, x2, y2], fill=255)
            else:
                logger.warning(f"Skipping large bounding box: {bbox} covering {bbox_area / image_area:.2%} of the image")

    # 补充一个手动添加啊的区域
    x1,y1,x2,y2 = 19,606,105,700
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def process_image_with_lama(image: MatLike, mask: MatLike, model_manager: ModelManager):
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def main():
    # 输入
    input_path = "D:\\workspace\\vmshareroom\\python_project\\watermarkRemover\\testInput\\photo_2025-12-02_19-08-50.jpg"

    # 判断是用cpu还是gpu
    useDevice = "cpu"
    florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(useDevice).eval()
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    logger.info("Florence-2 Model loaded")

    model_manager = ModelManager(name="lama", device=torch.device(useDevice))
    logger.info("LaMa model loaded")

    # 处理图片
    pil_image = Image.open(input_path)

    # Get watermark mask
    mask_image = get_watermark_mask(pil_image, florence_model, florence_processor, useDevice, 100.0)

    # 增加手动选中的区域

    # 处理帧
    lama_result = process_image_with_lama(np.array(pil_image), np.array(mask_image), model_manager)
    result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))
    print("11111111111111")


if __name__ == "__main__":
    main()
