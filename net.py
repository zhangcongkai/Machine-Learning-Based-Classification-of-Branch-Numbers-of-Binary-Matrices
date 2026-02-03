import os
import ast
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保可重复性
np.random.seed(42)
tf.random.set_seed(42)


# 加载扩充后的数据集
def load_augmented_data(data_dir='./augmented_dataset'):
    matrices = []
    labels = []

    # 遍历每个子文件夹 (bn1到bn5)
    for branch_num in range(2, 6):
        subdir = f'bn{branch_num}'
        subdir_path = os.path.join(data_dir, subdir)

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

    return np.array(matrices), np.array(labels)


print("正在加载扩充后的数据集...")
X, y = load_augmented_data()

# 将标签转换为0-based索引 (2->0, 3->1, 4->2, 5->3)
y = y - 2

print(f"数据集大小: {X.shape}")
print(f"标签分布: {np.bincount(y)}")

# 将数据分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")

# 数据预处理
# 将矩阵展平为64维向量
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# 将标签转换为one-hot编码
num_classes = 4
y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
y_val_onehot = keras.utils.to_categorical(y_val, num_classes)


# 构建神经网络模型
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

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model = create_model()
model.summary()

# 设置回调函数
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
)

# 训练模型
print("开始训练模型...")
history = model.fit(
    X_train_flat, y_train_onehot,
    batch_size=32,
    epochs=100,
    validation_data=(X_val_flat, y_val_onehot),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 评估模型
print("评估模型...")
val_loss, val_accuracy = model.evaluate(X_val_flat, y_val_onehot, verbose=0)
print(f"验证集损失: {val_loss:.4f}")
print(f"验证集准确率: {val_accuracy:.4f}")

# 绘制训练历史
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# 保存模型
model.save('matrix_branch_classifier.h5')
print("模型已保存为 matrix_branch_classifier.h5")

# 进行一些预测示例
print("\n预测示例:")
sample_indices = np.random.choice(len(X_val), 5, replace=False)

for i, idx in enumerate(sample_indices):
    sample_matrix = X_val[idx]
    true_label = y_val[idx] + 2  # 转换回2-based标签

    sample_flat = sample_matrix.reshape(1, -1)

    # 预测
    prediction = model.predict(sample_flat, verbose=0)
    predicted_label = np.argmax(prediction) + 2  # 转换回2-based标签
    confidence = np.max(prediction)

    print(f"示例 {i + 1}:")
    print(f"  真实分支数: {true_label}")
    print(f"  预测分支数: {predicted_label}")
    print(f"  置信度: {confidence:.4f}")
    print(f"  预测概率分布: {[f'{p:.4f}' for p in prediction[0]]}")
    print()