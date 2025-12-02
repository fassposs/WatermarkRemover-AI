import sys
import click
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
from torch.nn import Module
import tqdm
from loguru import logger
from enum import Enum
import os
import tempfile
import shutil
import subprocess

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
    text_input = "watermark"
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


def make_region_transparent(image: Image.Image, mask: Image.Image):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image


def is_video_file(file_path):
    """Check if the file is a video based on its extension"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return Path(file_path).suffix.lower() in video_extensions


def process_video(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format):
    """通过提取帧、去除水印和重建视频来处理视频文件"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 确定输出格式
    if force_format:
        output_format = force_format.upper()
    else:
        output_format = "MP4"  # 默认mp4格式

    # 输出路径
    output_path = Path(output_path)
    if output_path.is_dir():
        output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")

    # 创建一个临时文件，用于存放没有音频的视频
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"

    # 根据输出格式设置编解码器
    if output_format.upper() == "MP4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format.upper() == "AVI":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #默认mp4

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
            mask_image = get_watermark_mask(pil_image, florence_model, florence_processor, device, max_bbox_percent)

            # 处理帧
            if transparent:
                # 视频无法使用透明度，所以我们会填充颜色或背景
                result_image = make_region_transparent(pil_image, mask_image)
                # 将 RGBA 转换为 RGB，方法是用白色填充透明区域
                background = Image.new("RGB", result_image.size, (255, 255, 255))  # type: ignore
                background.paste(result_image, mask=result_image.split()[3])
                result_image = background
            else:
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
            shutil.copy(str(temp_video_path), str(output_file))
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
                str(output_file)
            ]

            # 运行 FFmpeg
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Fusion audio/vidéo terminée avec succès!")
    except Exception as e:
        logger.error(f"Erreur lors de la fusion audio/vidéo: {str(e)}")
        # 如果出现错误，请使用无声视频
        shutil.copy(str(temp_video_path), str(output_file))
    finally:
        # 清理临时文件
        try:
            os.remove(str(temp_video_path))
            os.rmdir(temp_dir)
        except:
            pass

    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")
    return output_file


def handle_one(image_path: Path, output_path: Path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite):
    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return

    # 检查是否是视频文件
    if is_video_file(image_path):
        return process_video(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format)

    # Process image
    image = Image.open(image_path).convert("RGB")
    mask_image = get_watermark_mask(image, florence_model, florence_processor, device, max_bbox_percent)

    if transparent:
        result_image = make_region_transparent(image, mask_image)
    else:
        lama_result = process_image_with_lama(np.array(image), np.array(mask_image), model_manager)
        result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

    # Determine output format
    if force_format:
        output_format = force_format.upper()
    elif transparent:
        output_format = "PNG"
    else:
        output_format = image_path.suffix[1:].upper()
        if output_format not in ["PNG", "WEBP", "JPG"]:
            output_format = "PNG"

    # Map JPG to JPEG for PIL compatibility
    if output_format == "JPG":
        output_format = "JPEG"

    if transparent and output_format == "JPG":
        logger.warning("Transparency detected. Defaulting to PNG for transparency support.")
        output_format = "PNG"

    new_output_path = output_path.with_suffix(f".{output_format.lower()}")
    result_image.save(new_output_path, format=output_format)
    logger.info(f"input_path:{image_path}, output_path:{new_output_path}")
    return new_output_path


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--overwrite", is_flag=True, help="以批量模式覆盖现有文件.")
@click.option("--transparent", is_flag=True, help="将水印区域设为透明，而不是移除.")
@click.option("--max-bbox-percent", default=10.0, help="边界框可以覆盖图像的最大百分比.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG", "MP4", "AVI"], case_sensitive=False), default=None, help="强制输出格式。默认为输入格式.")
def main(input_path: str, output_path: str, overwrite: bool, transparent: bool, max_bbox_percent: float, force_format: str):
    # 输入
    in_path = Path(input_path)
    # 输出
    out_path = Path(output_path)
    # 判断是用cpu还是gpu
    useDevice = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {useDevice}")
    florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(useDevice).eval()
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    logger.info("Florence-2 Model loaded")

    if not transparent:
        model_manager = ModelManager(name="lama", device=torch.device(useDevice))
        logger.info("LaMa model loaded")
    else:
        model_manager = None

    if in_path.is_dir():
        if not out_path.exists():
            out_path.mkdir(parents=True)

        # Include video files in the search
        images = list(in_path.glob("*.[jp][pn]g")) + list(in_path.glob("*.webp"))
        videos = list(in_path.glob("*.mp4")) + list(in_path.glob("*.avi")) + list(in_path.glob("*.mov")) + list(in_path.glob("*.mkv"))
        files = images + videos
        total_files = len(files)

        for idx, file_path in enumerate(tqdm.tqdm(files, desc="Processing files")):
            output_file = out_path / file_path.name
            handle_one(file_path, output_file, florence_model, florence_processor, model_manager, useDevice, transparent, max_bbox_percent, force_format, overwrite)
            progress = int((idx + 1) / total_files * 100)
            print(f"in_path:{file_path}, out_path:{output_file}, overall_progress:{progress}")
    else:
        # 输出文件
        output_file = out_path
        if is_video_file(in_path) and out_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
            # 确保视频输出具有正确的扩展名
            if force_format and force_format.upper() in ["MP4", "AVI"]:
                output_file = out_path.with_suffix(f".{force_format.lower()}")
            else:
                output_file = out_path.with_suffix(".mp4")  # 默认用mp4

        handle_one(in_path, output_file, florence_model, florence_processor, model_manager, useDevice, transparent, max_bbox_percent, force_format, overwrite)
        print(f"in_path:{in_path}, out_path:{output_file}, overall_progress:100")


if __name__ == "__main__":
    main()
