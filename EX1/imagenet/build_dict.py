import os

def build_dict(data_path):
    image_dict={}
    # 遍历数据集目录下的所有子目录（类别文件夹）
    for label_name in os.listdir(data_path):
        # 构造当前类别的文件夹路径
        label_path = os.path.join(data_path, label_name)

        # 确保当前路径是一个目录
        if os.path.isdir(label_path):
            # 获取该类别文件夹下的所有图片文件
            files = os.listdir(label_path)

            # 初始化一个列表，用于存储当前类别的所有图片编号
            image_dict[label_name] = []

            # 遍历当前类别的所有图片
            for filename in files:
                # 从文件名中提取图片编号（假设文件名格式为 类别编号_其他内容）
                label_idx = filename.split('_')[0]
                # print(f"{label_idx}")
                # 将图片编号添加到当前类别的列表中
                image_dict[label_name].append(label_idx)
    return image_dict


if __name__ == '__main__':
    # 定义数据集路径
    data_path = '../flower_dataset_0'

    # 初始化字典，用于存储类别名称与图片编号的映射关系
    image_dict=build_dict(data_path)
    # 打印构建的字典
    print(image_dict)

    print("总结：")
    tot = 0
    for label_name, label_indices in image_dict.items():
        print(f"类别 '{label_name}' 下有 {len(label_indices)} 个编号")
        tot += len(label_indices)
    print(f"共:{tot}张")