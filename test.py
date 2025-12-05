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
import time
from cv2.typing import MatLike
#  from paddleocr import PaddleOCR
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from easyocr import Reader
import signal
import sys

# 全局变量控制程序运行状态
running = True


def signal_handler(sig, frame):
    global running
    print("\n接收到中断信号，正在停止程序...")
    running = False
    sys.exit(0)


# 在 main1() 函数开始处注册信号处理器
signal.signal(signal.SIGINT, signal_handler)


class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""


def identify(task_prompt: TaskType, image: Image, text_input: str, model: AutoModelForCausalLM, processor: AutoProcessor, device: str):
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

    time1 = time.time()
    # 使用处理器将文本提示和图像转换为模型输入张量
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    time2 = time.time()
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
    # generated_ids = model.generate(
    #     input_ids=inputs["input_ids"],
    #     pixel_values=inputs["pixel_values"],
    #     max_new_tokens=512,  # 减少最大生成令牌数
    #     early_stopping=True,  # 启用早停
    #     do_sample=False,
    #     num_beams=1,  # 减少束搜索数量
    # )
    time3 = time.time()
    # 解码生成的token ID为文本，并进行后处理
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    time4 = time.time()
    result = processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )
    time5 = time.time()
    print(f"processor={int((time2 - time1) * 1000)}ms, "
          f"generate={int((time3 - time2) * 1000)}ms, "
          f"batch_decode={int((time4 - time3) * 1000)}ms, "
          f"process_generation={int((time5 - time4) * 1000)}ms, ")
    return result


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
                draw.rectangle((x1, y1, x2, y2), fill=255)
            else:
                logger.warning(f"Skipping large bounding box: {bbox} covering {bbox_area / image_area:.2%} of the image")

    # 补充一个手动添加啊的区域
    x1, y1, x2, y2 = 19, 606, 105, 700
    draw.rectangle((x1, y1, x2, y2), fill=255)
    return mask


# 使用easyocr来识别区域
def get_watermark_maskEx2(reader: Reader, image: MatLike, index):
    mask = Image.new("L", (image.width, image.height), 0)
    draw = ImageDraw.Draw(mask)
    result = reader.readtext(np.array(image))
    for detection in result:
        # 解包三元组：坐标、文本、置信度
        coordinates = detection[0]  # 四个角点坐标
        text = detection[1]  # 识别的文本
        # confidence = detection[2]  # 置信度

        # 获取边界框坐标
        # coordinates 是一个4x2的数组，包含四个角点：左上、右上、右下、左下
        top_left = coordinates[0]  # [x, y]
        # top_right = coordinates[1]     # [x, y]
        bottom_right = coordinates[2]  # [x, y]
        # bottom_left = coordinates[3]  # [x, y]
        target_chars = ["@今", "今晚", "晚追", "追剧"]
        if any(char_pair in text for char_pair in target_chars):
            draw.rectangle((float(top_left[0]), float(top_left[1]), float(bottom_right[0]), float(bottom_right[1])), fill=255)
        # else:
        #     logger.info(f"888888888888888888888888888   ====={index}")

    # if len(result) == 0:
    #     logger.info(f"9999999999999999999999999999   ====={index}")

    # 补充一个手动添加啊的区域
    x1, y1, x2, y2 = 19, 606, 105, 700
    draw.rectangle((x1, y1, x2, y2), fill=255)
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
    # config = Config(
    #     ldm_steps=20,  # 减少步数提高速度（原50）
    #     ldm_sampler=LDMSampler.ddim,
    #     hd_strategy=HDStrategy.CROP,
    #     hd_strategy_crop_margin=32,  # 减小边缘留白（原64）
    #     hd_strategy_crop_trigger_size=600,  # 降低触发阈值（原800）
    #     hd_strategy_resize_limit=1200,  # 降低尺寸限制（原1600）
    # )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def main():
    # 输入
    input_path = "D:\\workspace\\vmshareroom\\python_project\\watermarkRemover\\testInput\\photo_2025-12-02_19-08-50.jpg"

    # 判断是用cpu还是gpu
    useDevice = "cpu"
    # florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(useDevice).eval()
    # florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(useDevice).eval()
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
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
    input_path = "./testInput/test001.mp4"
    output_path = "./testOutput/test001.mp4"
    # input_path = "E:\\workspace\\vmshareroom\\python_project\\WatermarkRemover\\testInput\\test001.mp4"
    # output_path = "E:\\workspace\\vmshareroom\\python_project\\WatermarkRemover\\testOutput\\test001.mp4"

    # 判断是用cpu还是gpu
    useDevice = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"useDevice: {useDevice}")

    # useDevice = "cpu"
    local_model_path = "./models/Florence-2-large"
    florence_model = None # AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True).to(useDevice).eval()
    florence_processor = None # AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True)
    # florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(useDevice).eval()
    # florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    logger.info("Florence-2 Model loaded")
    ocr = None
    # ocr = PaddleOCR(
    #     device="cpu",
    #     # lang="ch",
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     use_textline_orientation=False)
    reader = Reader(['ch_sim', 'en'], gpu=True if useDevice == "cuda" else False)
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

    # frame_img_list = list()

    max_workers = 50
    threadPool = ThreadPoolExecutor(max_workers=max_workers)
    # futures = list()
    # 创建线程安全的队列
    shared_queue = queue.Queue(maxsize=100)

    def process_frame(pil_image1,florence_model1,florence_processor1,useDevice1,model_manager1,reader1, index):
        if not running:
            return
        # 通过大模型获取水印
        # mask_image = get_watermark_mask(pil_image1, florence_model1, florence_processor1, useDevice1, 100.0)

        # 获取水印区域通过easyocr
        mask_image = get_watermark_maskEx2(reader1, pil_image1, index)

        # 处理帧
        lama_result = process_image_with_lama(np.array(pil_image1), np.array(mask_image), model_manager1)
        return {"index": index, "data": lama_result}

    # 处理数据结果
    def get_result():
        time.sleep(3)
        needIndex = 0
        tmpList = list()
        while needIndex < total_frames - 1:
            if not running:
                break
            # 2秒扫一轮
            # time.sleep(1)
            while len(tmpList) > 0 and needIndex == tmpList[0]["index"]:
                print(f"处理结果: {tmpList[0]["index"]}")
                needIndex += 1
                # 重新加载为图片
                result_image = Image.fromarray(cv2.cvtColor(tmpList[0]["data"], cv2.COLOR_BGR2RGB))
                # 转换回 OpenCV 格式并写入输出视频
                frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                # 写入帧
                out.write(frame_result)
                # 去掉第一个数据
                tmpList.pop(0)

            future = shared_queue.get()
            if not future.done():
                shared_queue.put(future)
            else:
                result = future.result()
                if needIndex == result["index"]:
                    print(f"收到结果: {result["index"]}")
                    needIndex += 1
                    # 重新加载为图片
                    result_image = Image.fromarray(cv2.cvtColor(result["data"], cv2.COLOR_BGR2RGB))
                    # 转换回 OpenCV 格式并写入输出视频
                    frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                    # 写入帧
                    out.write(frame_result)
                else:
                    tmpList.append(result)
                    # 按index从小到大排序一次
                    tmpList.sort(key=lambda x: x["index"])

    # 启动读取线程
    t = threading.Thread(target=get_result, name='get_result')
    t.start()

    # 处理每一帧
    with tqdm.tqdm(total=total_frames, desc="Processing video frames") as pbar:
        frame_count = 0
        while cap.isOpened():
            if not running:
                break
            start_time = time.time()  # 开始计时
            ret, frame = cap.read()
            if not ret:
                break

            read_frame_time = time.time()  # 读取帧完成时间

            # 将帧转换为 PIL 图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            convert_time = time.time()  # 转换完成时间

            shared_queue.put(threadPool.submit(process_frame, pil_image, florence_model,florence_processor,useDevice,model_manager,reader,frame_count))

            # 更新进度
            frame_count += 1
            pbar.update(1)

            # 打印各步骤耗时（单位：毫秒）
            print(f"Frame {frame_count}: "
                  f"Read={int((read_frame_time - start_time) * 1000)}ms, "
                  f"Convert={int((convert_time - read_frame_time) * 1000)}ms, ")

    # 等数据用完
    while shared_queue.qsize() > 0:
        time.sleep(1)
    # Release resources
    cap.release()
    out.release()

    # 使用 FFmpeg 将处理后的视频与原始音频合并
    try:
        logger.info("将处理后的视频与原始音频合并...")

        # 检查 FFmpeg 是否可用
        try:
            subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg不可用 视频将不带声音.")
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
            logger.info("音视频融合成功完成!")
    except Exception as e:
        logger.error(f"音频/视频合并过程中出错: {str(e)}")
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
