import os
import ast
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# ========== 第一步：加载原始数据 ==========
root_dir = './chushishujuji'
matrices = []
labels = []

print("正在加载原始数据...")

for branch_num in range(1, 6):
    subdir = f'bn{branch_num}'
    subdir_path = os.path.join(root_dir, subdir)

    if not os.path.exists(subdir_path):
        print(f"警告: 文件夹 {subdir_path} 不存在")
        continue

    for file_name in os.listdir(subdir_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(subdir_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
            try:
                matrix_list = ast.literal_eval(content)
                matrix = np.array(matrix_list)
                if matrix.shape == (8, 8):
                    matrices.append(matrix)
                    labels.append(branch_num)
                else:
                    print(f"警告: {file_path} 不是 8x8")
            except Exception as e:
                print(f"错误: {file_path} 解析失败: {e}")

print(f"成功加载 {len(matrices)} 个原始矩阵")

# 按标签分组
matrices_by_label = {}
for m, l in zip(matrices, labels):
    matrices_by_label.setdefault(l, []).append(m)

# ========== 第二步：数据扩充 ==========
def apply_row_operations(matrix):
    new_matrix = matrix.copy()
    operations = random.sample(['swap', 'flip', 'xor'], k=random.randint(1, 3))
    for op in operations:
        if op == 'swap':
            i, j = random.sample(range(8), 2)
            new_matrix[[i, j]] = new_matrix[[j, i]]
        elif op == 'flip':
            row = random.randint(0, 7)
            new_matrix[row] = 1 - new_matrix[row]
        elif op == 'xor':
            i, j = random.sample(range(8), 2)
            new_matrix[i] = np.bitwise_xor(new_matrix[i], new_matrix[j])
    return new_matrix

augmented_matrices = []
augmented_labels = []

for label, matrices_list in matrices_by_label.items():
    print(f"处理分支 {label} ...")
    original_count = len(matrices_list)
    variants_per_matrix = 10000 // original_count
    extra_needed = 10000 % original_count

    for i, matrix in enumerate(matrices_list):
        augmented_matrices.append(matrix)
        augmented_labels.append(label)
        variants_to_generate = variants_per_matrix - 1
        if i < extra_needed:
            variants_to_generate += 1
        for _ in range(variants_to_generate):
            variant = apply_row_operations(matrix)
            augmented_matrices.append(variant)
            augmented_labels.append(label)

    print(f"分支 {label} 已扩充到 {len([l for l in augmented_labels if l == label])}")

augmented_matrices = np.array(augmented_matrices)
augmented_labels = np.array(augmented_labels)

print(f"扩充后数据集: {augmented_matrices.shape}")
print(f"标签分布: {np.bincount(augmented_labels)}")

# ========== 第三步：训练分类模型 ==========
# 只用分支 2-5，映射为 0-3
mask = augmented_labels >= 2
X = augmented_matrices[mask]
y = augmented_labels[mask] - 2  # 标签转换为 0-based

print(f"分类数据集大小: {X.shape}, 标签分布: {np.bincount(y)}")

# train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

num_classes = 4
y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
y_val_onehot = keras.utils.to_categorical(y_val, num_classes)

def create_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(64,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
)

print("开始训练模型...")
history = model.fit(
    X_train_flat, y_train_onehot,
    batch_size=32, epochs=100,
    validation_data=(X_val_flat, y_val_onehot),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 评估
val_loss, val_acc = model.evaluate(X_val_flat, y_val_onehot, verbose=0)
print(f"验证集损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")

# 绘制曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率'); plt.xlabel('轮次'); plt.ylabel('准确率'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失'); plt.xlabel('轮次'); plt.ylabel('损失'); plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# 保存模型
model.save('matrix_branch_classifier.h5')
print("模型已保存 matrix_branch_classifier.h5")

# 预测示例
print("\n预测示例:")
sample_indices = np.random.choice(len(X_val), 5, replace=False)
for i, idx in enumerate(sample_indices):
    sample_matrix = X_val[idx]
    true_label = y_val[idx] + 2
    pred = model.predict(sample_matrix.reshape(1, -1), verbose=0)
    pred_label = np.argmax(pred) + 2
    confidence = np.max(pred)
    print(f"示例{i+1}: 真实={true_label}, 预测={pred_label}, 置信度={confidence:.4f}, 概率分布={pred[0]}")
