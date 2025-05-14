import os
import shutil
import random
from build_dict import build_dict

# 定义路径
origin_path='../flower_dataset_0'
merged_data_path = '../flower_dataset_1'  # 合并后的文件夹路径
train_path = '../flower_dataset/train'  # 训练集路径
val_path = '../flower_dataset/val'  # 验证集路径

# 确保训练集和验证集目录存在
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)

image_dict={}
# 调用函数构建字典
image_dict = build_dict(origin_path)
print(image_dict)

print("总结：")
tot = 0
for label_name, label_indices in image_dict.items():
    print(f"类别 '{label_name}' 下有 {len(label_indices)} 个编号")
    tot += len(label_indices)
print(f"共:{tot}张")

# 获取合并后的文件夹中的所有图片文件
files = os.listdir(merged_data_path)

# 随机打乱文件列表
random.shuffle(files)

# 计算训练集和验证集的文件数量
num_files = len(files)
num_train = int(num_files * 0.8)

# 分配文件到训练集和验证集
for i, filename in enumerate(files):
    # 从文件名中提取图片编号
    label_idx = filename.split('_')[0]

    # 遍历image_dict，找到包含该编号的类别名称
    class_name = None
    for key, value in image_dict.items():
        if label_idx in value:
            class_name = key
            break
            
    if class_name is None:
        print(f"次数：{i},警告：图片 {filename} 的编号 {label_idx} 未在字典中找到对应的类别名称。")
        continue

    # 构造源文件路径
    src_file = os.path.join(merged_data_path, filename)

    # 根据索引将文件复制到训练集或验证集的对应类别文件夹中
    if i < num_train:
        dst_path = os.path.join(train_path, class_name)
    else:
        dst_path = os.path.join(val_path, class_name)

    # 确保目标文件夹存在
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    # 构造目标文件路径
    dst_file = os.path.join(dst_path, filename)

    # 复制文件
    shutil.copy(src_file, dst_file)

print("数据集拆分完成！")