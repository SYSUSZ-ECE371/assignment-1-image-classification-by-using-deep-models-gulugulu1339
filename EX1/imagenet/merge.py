import os
import shutil

# 输入文件夹路径和输出文件夹路径
src_path = '../flower_dataset_0'  # 原始文件夹路径
dst_path = '../flower_dataset_1'  # 输出文件夹路径

# 如果目标文件夹不存在，则创建
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

# 计数器初始化
count = 0

# 遍历源目录下的所有子目录和文件
for root, dirs, files in os.walk(src_path):
    # 处理当前目录下的所有文件
    for file in files:
        # 构造源文件和目标文件的完整路径
        src_file = os.path.join(root, file)
        dst_file = os.path.join(dst_path, file)
        # 复制文件到目标目录
        shutil.copy(src_file, dst_file)
        # 计数器加1
        count += 1
        # 打印处理进度
        print('正在处理第{}张图片: {}'.format(count, src_file))

    # 处理完成的标志
print('所有图片已处理完成！')