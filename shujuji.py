import os
import ast
import numpy as np
import random

# 定义根文件夹路径
root_dir = './chushishujuji'
matrices = []
labels = []

print("正在加载原始数据...")

# 遍历每个子文件夹 (bn1到bn5)
for branch_num in range(1, 6):
    subdir = f'bn{branch_num}'
    subdir_path = os.path.join(root_dir, subdir)

    # 检查子文件夹是否存在
    if not os.path.exists(subdir_path):
        print(f"警告: 文件夹 {subdir_path} 不存在")
        continue

    # 遍历子文件夹中的所有txt文件
    for file_name in os.listdir(subdir_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(subdir_path, file_name)

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()

            try:
                # 使用ast.literal_eval安全地解析字符串为列表
                matrix_list = ast.literal_eval(content)

                # 转换为numpy数组
                matrix = np.array(matrix_list)

                # 确保矩阵是8x8的
                if matrix.shape == (8, 8):
                    matrices.append(matrix)
                    labels.append(branch_num)
                else:
                    print(f"警告: 文件 {file_path} 的矩阵格式不正确，期望8x8，实际为{matrix.shape}")

            except (ValueError, SyntaxError) as e:
                print(f"错误: 无法解析文件 {file_path} 的内容: {e}")

print(f"成功加载 {len(matrices)} 个原始矩阵")

# 按标签分组矩阵
matrices_by_label = {}
for matrix, label in zip(matrices, labels):
    if label not in matrices_by_label:
        matrices_by_label[label] = []
    matrices_by_label[label].append(matrix)

# 创建扩充后的数据集
augmented_matrices = []
augmented_labels = []


# 行变换函数
def apply_row_operations(matrix):
    """对矩阵应用随机行变换"""
    # 复制矩阵以避免修改原始数据
    new_matrix = matrix.copy()

    # 随机选择一种或多种行变换
    operations = random.sample([
        'swap',  # 交换两行
        'flip',  # 翻转一行
        'xor'  # 将一行与另一行进行异或操作
    ], k=random.randint(1, 3))  # 随机选择1-3种操作

    for op in operations:
        if op == 'swap' and len(new_matrix) >= 2:
            # 随机选择两行进行交换
            i, j = random.sample(range(8), 2)
            new_matrix[[i, j]] = new_matrix[[j, i]]

        elif op == 'flip':
            # 随机选择一行进行翻转 (0变1，1变0)
            row = random.randint(0, 7)
            new_matrix[row] = 1 - new_matrix[row]

        elif op == 'xor' and len(new_matrix) >= 2:
            # 随机选择两行，将一行与另一行进行异或
            i, j = random.sample(range(8), 2)
            new_matrix[i] = np.bitwise_xor(new_matrix[i], new_matrix[j])

    return new_matrix


# 对每个标签的矩阵生成变体，使每个标签有10000个矩阵
for label, matrices_list in matrices_by_label.items():
    print(f"处理分支数 {label} 的矩阵...")
    original_count = len(matrices_list)

    # 每个原始矩阵需要生成的变体数量
    variants_per_matrix = 10000 // original_count

    # 如果需要额外的矩阵来达到10000个
    extra_needed = 10000 % original_count

    for i, matrix in enumerate(matrices_list):
        # 添加原始矩阵
        augmented_matrices.append(matrix)
        augmented_labels.append(label)

        # 生成变体
        variants_to_generate = variants_per_matrix - 1  # -1 因为已经添加了原始矩阵

        # 如果需要额外的矩阵，前extra_needed个矩阵多生成一个变体
        if i < extra_needed:
            variants_to_generate += 1

        for _ in range(variants_to_generate):
            variant = apply_row_operations(matrix)
            augmented_matrices.append(variant)
            augmented_labels.append(label)

    print(f"分支数 {label} 已扩充到 {len([l for l in augmented_labels if l == label])} 个矩阵")

# 转换为numpy数组
augmented_matrices = np.array(augmented_matrices)
augmented_labels = np.array(augmented_labels)

print(f"扩充后数据集大小: {len(augmented_matrices)} 个矩阵")
print(f"矩阵形状: {augmented_matrices.shape}")
print(f"标签分布: {np.bincount(augmented_labels)}")

# 保存扩充后的数据集
output_dir = './augmented_dataset'
os.makedirs(output_dir, exist_ok=True)

# 按标签分组保存
for branch_num in range(1, 6):
    branch_indices = np.where(augmented_labels == branch_num)[0]
    branch_matrices = augmented_matrices[branch_indices]

    branch_dir = os.path.join(output_dir, f'bn{branch_num}')
    os.makedirs(branch_dir, exist_ok=True)

    for i, matrix in enumerate(branch_matrices):
        # 将矩阵转换为列表形式并保存为txt文件
        matrix_list = matrix.tolist()
        file_path = os.path.join(branch_dir, f'matrix_{i:05d}.txt')

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(matrix_list))

    print(f"已保存分支数 {branch_num} 的 {len(branch_matrices)} 个矩阵")

print("数据集扩充完成!")