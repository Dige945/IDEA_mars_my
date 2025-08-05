import os
import json
from collections import Counter, defaultdict
import glob

def debug_annotations_and_dataset(root):
    """
    1. 检查标注文件是否存在
    2. 加载 JSON，建立 caption_map
    3. 列出 RGB 目录下所有图片，检查有多少张没有对应标注
    4. 检查 PID 提取及映射是否合理
    """
    data_dir = os.path.join(root, 'marslite')
    train_rgb_dir = os.path.join(data_dir, 'train', 'RGB')
    ann_file = os.path.join(data_dir, 'text_update', 'train_annotations.json')
    
    print(f"DATA ROOT: {data_dir}")
    print(f"RGB TRAIN DIR: {train_rgb_dir}")
    print(f"ANNOTATION FILE: {ann_file}")
    print()

    # 1. 加载标注
    if not os.path.isfile(ann_file):
        print("❌ train_annotations.json 不存在！请检查路径。")
        return
    with open(ann_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    print(f"✅ 加载标注条目: {len(annotations)}")

    # 2. 建立 caption_map
    caption_map = {}
    for item in annotations:
        img_name = os.path.basename(item.get('img_path',''))
        caps = item.get('captions', [])
        if len(caps)>0:
            caption_map[img_name] = caps[0]
    print(f"✅ 有效 caption_map 项目: {len(caption_map)}")
    print()

    # 3. 列出所有 RGB 图片
    rgb_imgs = sorted(glob.glob(os.path.join(train_rgb_dir, '*.jpg')))
    print(f"🔍 训练集 RGB 图片总数: {len(rgb_imgs)}")
    
    # 检查无标注的图片
    no_ann = [os.path.basename(p) for p in rgb_imgs if os.path.basename(p) not in caption_map]
    print(f"⚠️  无标注图片数量: {len(no_ann)}")
    if len(no_ann)>0:
        print("  例子:", no_ann[:10])
    print()

    # 4. 检查 PID 提取和映射
    import re
    pattern = re.compile(r'([a-z\d]+)_c(\d)')
    pid_container = set()
    pid_counter = Counter()
    for p in rgb_imgs:
        name = os.path.basename(p)
        m = pattern.search(name)
        if m:
            pid_str, _ = m.groups()
            pid_container.add(pid_str)
            pid_counter[pid_str] += 1
    print(f"👥 提取到的唯一原始 PID 数量: {len(pid_container)}")
    print(f"📄 前 10 个 PID 示例: {list(pid_container)[:10]}")
    print(f"🖼️  平均每个 PID 图片数: {sum(pid_counter.values())/len(pid_counter):.1f}")
    print()

    # 5. 构建 pid2label
    pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_container))}
    print(f"✅ 构建 pid2label (0 ~ {len(pid2label)-1})")
    print("  前 5 个映射:", list(pid2label.items())[:5])
    print()

    # 6. 最终 dataset entry 数
    dataset = []
    for p in rgb_imgs:
        name = os.path.basename(p)
        if name not in caption_map: 
            continue
        m = pattern.search(name)
        if not m: 
            continue
        pid_str, cam_str = m.groups()
        pid = pid2label[pid_str]
        dataset.append((name, pid))
    print(f"🎯 最终可用训练条目数: {len(dataset)}")
    print(f"🎯 其中标签分布:", Counter([pid for _, pid in dataset]).most_common(5))
    print()

if __name__ == "__main__":
    # 请根据实际修改 root 路径
    debug_annotations_and_dataset('/data2/xieyudi/datasets')