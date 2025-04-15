import argparse
import csv
import datetime
import os
import sys
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import psutil
import torch
from demo import load_checkpoints
from skimage import img_as_ubyte
from skimage.transform import resize
from src.services import AnimationService, LoggingService

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

os.makedirs("./data/output", exist_ok=True)
os.makedirs("./data/metrics", exist_ok=True)

log = LoggingService.setup_logger(__name__)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image", required=True, help="Path to image to animate")
ap.add_argument("-c", "--checkpoint", required=True, help="Path to checkpoint")
ap.add_argument("-v", "--input_video", required=False, help="Path to video input")
args = vars(ap.parse_args())


RELATIVE = True
ADAPT_MOVEMENT_SCALE = True
USE_CPU = True

log.info("Loading source image and checkpoint")
source_path = args["input_image"]
checkpoint_path = args["checkpoint"]
video_path = args["input_video"] if args["input_video"] else None
source_image = imageio.imread(source_path)
source_image = resize(source_image, (256, 256))[..., :3]

generator, kp_detector = load_checkpoints(
    config_path="./data/configs/vox-256.yaml", checkpoint_path=checkpoint_path, cpu=USE_CPU
)


# Инициализация метрик
start_time = time.time()
frame_count = 0
total_latency = 0
max_memory = 0
metrics_file = "./data/metrics/performance_metrics.csv"

# Создаем файл для записи метрик
with open(metrics_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "FPS", "Latency(ms)", "Memory(MB)", "Timestamp"])

if video_path:
    cap = cv2.VideoCapture(video_path)
    log.info("Loading video from the given path")
else:
    cap = cv2.VideoCapture(0)
    log.info("Initializing front camera")

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out1 = cv2.VideoWriter("./data/output/test.avi", fourcc, 12, (256 * 3, 256), True)

cv2_source = cv2.cvtColor(source_image.astype("float32"), cv2.COLOR_BGR2RGB)
with torch.no_grad():
    predictions = []
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if not USE_CPU:
        source = source.cuda()
    kp_source = kp_detector(source)
    count = 0

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if not video_path:
            x = 143
            y = 87
            w = 322
            h = 322
            frame = frame[y : y + h, x : x + w]

        frame1 = resize(frame, (256, 256))[..., :3]

        if count == 0:
            source_image1 = frame1
            source1 = torch.tensor(source_image1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            kp_driving_initial = kp_detector(source1)

        frame_test = torch.tensor(frame1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

        driving_frame = frame_test
        if not USE_CPU:
            driving_frame = driving_frame.cuda()

        kp_driving = kp_detector(driving_frame)
        kp_norm = AnimationService.normalize_kp(
            kp_source=kp_source,
            kp_driving=kp_driving,
            kp_driving_initial=kp_driving_initial,
            use_relative_movement=RELATIVE,
            use_relative_jacobian=RELATIVE,
            adapt_movement_scale=ADAPT_MOVEMENT_SCALE,
        )

        # Замер памяти перед обработкой
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # в MB
        max_memory = max(max_memory, current_memory)

        out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

        # Расчет метрик производительности
        frame_count += 1
        frame_latency = (time.time() - frame_start) * 1000  # в мс
        total_latency += frame_latency
        current_fps = 1 / (frame_latency / 1000) if frame_latency > 0 else 0

        # Запись метрик в файл
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    frame_count,
                    current_fps,
                    frame_latency,
                    current_memory,
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )

        # Вывод в лог каждые 10 кадров
        if frame_count % 10 == 0:
            avg_fps = frame_count / (time.time() - start_time)
            avg_latency = total_latency / frame_count
            log.info(
                f"Performance - Frame {frame_count}: {current_fps:.1f} FPS | Latency: {frame_latency:.1f}ms | Memory: {current_memory:.1f}MB"
            )
            log.info(f"Average - FPS: {avg_fps:.1f} | Latency: {avg_latency:.1f}ms | Peak Memory: {max_memory:.1f}MB")

        predictions.append(np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0])
        im = np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        joinedFrame = np.concatenate((cv2_source, im, frame1), axis=1)

        cv2.imshow("Test", joinedFrame)
        out1.write(img_as_ubyte(joinedFrame))
        count += 1

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

# Итоговый отчет
log.info("\n=== Final Performance Report ===")
log.info(f"Total frames processed: {frame_count}")
log.info(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")
log.info(f"Average Latency: {total_latency / frame_count:.1f}ms")
log.info(f"Peak memory usage: {max_memory:.1f} MB")
log.info(f"Metrics saved to: {metrics_file}")

cap.release()
out1.release()
cv2.destroyAllWindows()
