







































































































































import os
import glob
import cv2
from pathlib import Path


root = Path(__file__).resolve().parents[1]
video_dir = str(Path(os.environ.get('PSYMM_DATA', root / 'data')) / 'AvaMERG/video_v5_0')
image_dir = str(root / 'infer_outputs_esconv/avamerg_test_images')


video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')


os.makedirs(image_dir, exist_ok=True)


for ext in video_extensions:

    pattern = os.path.join(video_dir, ext)

    for video_path in glob.glob(pattern):


        video_raw_name = os.path.splitext(os.path.basename(video_path))[0]


        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            continue


        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"无法获取帧数: {video_path}")
            cap.release()
            continue


        mid_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)

        ret, frame = cap.read()

        if not ret:
            print(f"无法读取帧: {video_path}")
            cap.release()
            continue


        output_path = os.path.join(image_dir, f"{video_raw_name}.jpg")

        cv2.imwrite(output_path, frame)

        print(f"成功: {video_raw_name}.jpg")

        cap.release()

print("全部视频处理完成")
