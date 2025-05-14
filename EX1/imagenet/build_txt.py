import os
from build_dict import build_dict

# 定义路径
train_path = '../data/flower_dataset/train'   # 训练集路径
val_path = '../data/flower_dataset/val'     # 验证集路径
classes_txt_path = '../data/flower_dataset/classes.txt'  # 类别文件路径
train_txt_path = '../data/flower_dataset/train.txt'  # 训练集文件路径
val_txt_path = '../data/flower_dataset/val.txt'    # 验证集文件路径

# 获取所有类别名称
classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
classes.extend([d for d in os.listdir(val_path) if d not in classes and os.path.isdir(os.path.join(val_path, d))])


# 将类别名称写入classes.txt
with open(classes_txt_path, 'w') as f:
    for index, class_name in enumerate(classes):
        f.write(f"({index},'{class_name}')\n")

# 构建训练集和验证集的图片编号与类别名称的映射字典
train_image_dict = build_dict(train_path)
val_image_dict = build_dict(val_path)

# 定义一个函数来写入train.txt和val.txt文件
def write_to_txt(file_path, image_dict, classes, dataset_path):
    with open(file_path, 'w') as f:
        for  index, class_name in enumerate(classes):
            class_path = os.path.join(dataset_path, class_name)
            for file in os.listdir(class_path):
                if file.endswith(('.jpg', '.jpeg')):
                    label_idx = file.split('_')[0]
                    if label_idx in image_dict[class_name]:
                        f.write(f"{class_name}/{file} {index}\n")

# 写入train.txt
write_to_txt(train_txt_path, train_image_dict, classes, train_path)
# 写入val.txt
write_to_txt(val_txt_path, val_image_dict, classes, val_path)

# 输出两个文件中写入的每个类别和每个类别的图片数量
def print_class_image_count(file_path, classes):
    class_image_count = {class_name: 0 for class_name in classes}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split('/')
        class_name = parts[0]
        class_image_count[class_name] += 1
    return class_image_count

train_image_count = print_class_image_count(train_txt_path, classes)
val_image_count = print_class_image_count(val_txt_path, classes)

# 打印每个类别的图片数量
print("Train dataset:")
for class_name, count in train_image_count.items():
    print(f"{class_name}: {count} images")
total_train_images = sum(train_image_count.values())
print(f"Total train images: {total_train_images}\n")

print("Validation dataset:")
for class_name, count in val_image_count.items():
    print(f"{class_name}: {count} images")
total_val_images = sum(val_image_count.values())
print(f"Total validation images: {total_val_images}")