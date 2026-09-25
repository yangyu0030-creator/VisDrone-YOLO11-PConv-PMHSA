import os
import random
import shutil
from PIL import Image

SOURCE_DIRS = {
    "VisDrone2019-DET-train"
}

OUTPUT_DIR = './datasets/VisDrone2019-DET'
SPLIT_RATIO = 0.8
RANDOM_SEED = 42

CLASS_NAMES = {
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
}

def convert_annotation(anno_path, img_w, img_h):
    yolo_lines = []
    with open(anno_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue

            bbox_left = float(parts[0])
            bbox_top = float(parts[1])
            bbox_width = float(parts[2])
            bbox_height = float(parts[3])
            score = int(parts[4])
            category = int(parts[5])

            if score == 0 or bbox_width <= 0 or bbox_height <= 0:
                continue

            x_center = (bbox_left + bbox_width / 2.0) / img_w
            y_center = (bbox_top + bbox_height / 2.0) / img_h
            norm_w = bbox_width / img_w
            norm_h = bbox_height / img_h

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            norm_w = min(max(norm_w, 0.0), 1.0)
            norm_h = min(max(norm_h, 0.0), 1.0)

            class_id = category - 1
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                continue

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )
        return yolo_lines

def main():
    all_samples = []
    for src_dir in SOURCE_DIRS:
        img_dir = os.path.join(src_dir, 'images')
        anno_dir = os.path.join(src_dir, 'annotations')

        if not os.path.isdir(img_dir) or not os.path.isdir(anno_dir):
            print(f"⚠️ 跳过 {src_dir}，找不到 images 或 annotations 文件夹")
            continue

        for anno_file in os.listdir(anno_dir):
            if not anno_file.endswith('.txt'):
                continue

            img_name = anno_file.replace('.txt', '.jpg')
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                # 尝试其他扩展名
                for ext in ['.png', '.jpeg', '.JPG']:
                    alt = os.path.join(img_dir, anno_file.replace('.txt', ext))
                    if os.path.exists(alt):
                        img_path = alt
                        break
                else:
                    print(f"❌ 找不到图片: {img_name}")
                    continue

            anno_path = os.path.join(anno_dir, anno_file)

            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"❌ 读取图片失败: {img_path}, {e}")
                continue

            yolo_lines = convert_annotation(anno_path, img_w, img_h)
            if yolo_lines:
                all_samples.append((img_path, yolo_lines))

    print(f"✅ 总共处理了 {len(all_samples)} 个有效样本")
    if not all_samples:
        print("没有有效样本，退出。")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * SPLIT_RATIO)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    def save_samples(samples, split):
        for img_path, yolo_lines in samples:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'

            # 复制图片
            dst_img = os.path.join(OUTPUT_DIR, 'images', split, img_name)
            shutil.copy2(img_path, dst_img)

            # 写入标签
            dst_label = os.path.join(OUTPUT_DIR, 'labels', split, label_name)
            with open(dst_label, 'w') as f:
                f.write('\n'.join(yolo_lines))

    save_samples(train_samples, 'train')
    save_samples(val_samples, 'val')

    print(f"🎉 划分完成！训练集: {len(train_samples)}，验证集: {len(val_samples)}")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    main()