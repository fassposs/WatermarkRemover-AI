import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
from loguru import logger
from enum import Enum
import tempfile
from pathlib import Path
import tqdm
import subprocess
import shutil
import os

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

def main1():
    # 输入
    # input_path = "D:\\workspace\\vmshareroom\\python_project\\watermarkRemover\\testInput\\test001.mp4"
    # output_path = "D:\\workspace\\vmshareroom\\python_project\\watermarkRemover\\testOutput\\test001.mp4"
    input_path = "E:\\workspace\\vmshareroom\\python_project\\WatermarkRemover\\testInput\\test001.mp4"
    output_path = "E:\\workspace\\vmshareroom\\python_project\\WatermarkRemover\\testOutput\\test001.mp4"

    # 判断是用cpu还是gpu
    useDevice = "cuda" if torch.cuda.is_available() else "cpu"
    # useDevice = "cpu"
    florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(useDevice).eval()
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    logger.info("Florence-2 Model loaded")

    model_manager = ModelManager(name="lama", device=torch.device(useDevice))
    logger.info("LaMa model loaded")

    # 处理视频
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建一个临时文件，用于存放没有音频的视频
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / "temp_no_audio.mp4"

    # 根据输出格式设置编解码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

    # 处理每一帧
    with tqdm.tqdm(total=total_frames, desc="Processing video frames") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 将帧转换为 PIL 图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Get watermark mask
            mask_image = get_watermark_mask(pil_image, florence_model, florence_processor, useDevice, 100.0)

            # 处理帧
            lama_result = process_image_with_lama(np.array(pil_image), np.array(mask_image), model_manager)
            result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

            # 转换回 OpenCV 格式并写入输出视频
            frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            out.write(frame_result)

            # 更新进度
            frame_count += 1
            pbar.update(1)
            progress = int((frame_count / total_frames) * 100)
            print(f"Processing frame {frame_count}/{total_frames}, progress:{progress}%")

    # Release resources
    cap.release()
    out.release()

    # 使用 FFmpeg 将处理后的视频与原始音频合并
    try:
        logger.info("Fusion de la vidéo traitée avec l'audio original...")

        # 检查 FFmpeg 是否可用
        try:
            subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg n'est pas disponible. La vidéo sera produite sans audio.")
            shutil.copy(str(temp_video_path), str(output_path))
        else:
            # 使用 FFmpeg 将处理后的视频与原始音频合并
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_video_path),  # Vidéo traitée sans audio
                "-i", str(input_path),  # Vidéo originale avec audio
                "-c:v", "copy",  # Copier la vidéo sans réencodage
                "-c:a", "aac",  # Encoder l'audio en AAC pour meilleure compatibilité
                "-map", "0:v:0",  # Utiliser la piste vidéo du premier fichier (vidéo traitée)
                "-map", "1:a:0",  # Utiliser la piste audio du deuxième fichier (vidéo originale)
                "-shortest",  # Terminer quand la piste la plus courte se termine
                str(output_path)
            ]

            # 运行 FFmpeg
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Fusion audio/vidéo terminée avec succès!")
    except Exception as e:
        logger.error(f"Erreur lors de la fusion audio/vidéo: {str(e)}")
        # 如果出现错误，请使用无声视频
        shutil.copy(str(temp_video_path), str(output_path))
    finally:
        # 清理临时文件
        try:
            os.remove(str(temp_video_path))
            os.rmdir(temp_dir)
        except:
            pass

    logger.info(f"input_path:{input_path}, output_path:{output_path}, overall_progress:100")
    print("11111111111111")

if __name__ == "__main__":
    # main()
    main1()