"""
用于剪切YOLO格式的图片
"""

import os

from PIL import Image

classes = ['boat']
views = ['left', 'right', 'up', 'front', 'behind']
backgrounds = ['sea', 'other']

def image_cut(img_dir, annotation_dir, out_path):
    # print("%s is processing...."%(img_dir))
    img = Image.open(img_dir)
    img = img.convert("RGB")
    width, height = img.size
    annotation = open(annotation_dir)
    line = annotation.readline()
    lines = []
    while line:
        lines.append(line.split())
        line = annotation.readline()

    new_cut = [float(item) for item in lines[0]]
    if int(lines[1][0]) < int(lines[2][0]):
        view = int(lines[1][0])
        background = int(lines[2][0])
    else:
        background = int(lines[1][0])
        view = int(lines[2][0])
    # print(len(new_cut))
    x, y, w, h = new_cut[1:]
    new_cut.clear()
    # cut.clear()
    left = int(width * x - 0.5 * width * w)
    right = int(width * x + 0.5 * width * w)
    upper = int(height * y - 0.5 * height * h)
    lower = int(height * y + 0.5 * height * h)
    cropped = img.crop((left, upper, right, lower))  # (left, upper, right, lower)
    out_path = out_path[:-4] + "_" + str(view - 1) + "_" + str(background - 6) + ".jpg"
    cropped.save(out_path)


def main():
    resource_dir = r"/home/xyc/datasets/test_labelimg/new/"  # 第一级目录，对应姓名目录
    save_path = r"/home/xyc/datasets/test_labelimg/cut/"
    for vessel_id in os.listdir(resource_dir):
        sub_dir = os.path.join(resource_dir, vessel_id)  # 二级目录，对应船舶编号
        save_dir = os.path.join(save_path, vessel_id)
        for file in os.listdir(sub_dir):
            path = os .path.join(sub_dir, file)
            if not os.path.isdir(path):
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img_dir = path
                label_dir = os.path.join(sub_dir, "labels", file[:-4]+".txt")
                out_path = os.path.join(save_dir, file)
                image_cut(img_dir, label_dir, out_path)

if __name__=="__main__":
    main()