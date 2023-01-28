"""
用于剪切YOLO格式的图片
"""

import os

from PIL import Image

classes = ['ship']
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
    # print(len(new_cut))
    x, y, w, h, confidence = new_cut[1:]
    new_cut.clear()
    # cut.clear()
    if confidence >= 0.8:
        left = int(width * x - 0.5 * width * w)
        right = int(width * x + 0.5 * width * w)
        upper = int(height * y - 0.5 * height * h)
        lower = int(height * y + 0.5 * height * h)
        cropped = img.crop((left, upper, right, lower))  # (left, upper, right, lower)
        cropped.save(out_path)
    # print("%s is processed!!\n" % (img_dir))

def main():
    img_dir = "D:\Dataset\\new_vessel\images (2)"
    label_dir = "D:\Dataset\\new_vessel\images_labels"
    save_path = "D:\Dataset\\new_vessel\yolo_cut"
    for class_name in os.listdir(label_dir):
        path = os.path.join(label_dir, class_name, "labels")
        if os.path.isdir(path):
            for img_lable in os.listdir(path):
                img_lable_dir = os.path.join(path, img_lable)
                img_path = os.path.join(img_dir, class_name, img_lable[:-3]+"jpg")
                if not os.path.exists(os.path.join(save_path, class_name)):
                    os.makedirs(os.path.join(save_path, class_name))
                out_path_dir = os.path.join(save_path, class_name, img_lable[:-3]+"jpg")
                image_cut(img_path, img_lable_dir, out_path_dir)
                print(f"{img_lable} has processed!")

        print(f"{class_name} has processed!")

    print("All images are peocessed!")

if __name__=="__main__":
    main()