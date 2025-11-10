import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 1. 读取 CSV 文件
# ================================
file_path = "your_file.csv"  # <-- 修改为你的实际文件路径
df = pd.read_csv(file_path)

# ================================
# 2. 提取特征与标签
# ================================
# 确保 riscode 列存在
if "riscode" not in df.columns:
    raise ValueError("数据中未找到名为 'riscode' 的列，请检查文件。")

X = df.drop(columns=["riscode"])
y = df["riscode"]

# 将 riscode 转换为 0 / 1（如果是10000和90000）
y = y.map({10000: 1, 90000: 0})

# ================================
# 3. 划分训练集与测试集
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 4. 训练 XGBoost 分类器
# ================================
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ================================
# 5. 模型预测与评估
# ================================
y_pred = model.predict(X_test)

# 准确率
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ 准确率(Accuracy): {acc:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵 (Confusion Matrix):")
print(cm)

# 分类报告（含精确率、召回率、F1）
print("\n分类报告 (Classification Report):")
print(classification_report(y_test, y_pred, digits=4))

# ================================
# 6. 可视化混淆矩阵
# ================================
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["10000(1)", "90000(0)"], yticklabels=["10000(1)", "90000(0)"])
plt.title("Confusion Matrix of XGBoost Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ================================
# 7. 特征重要性可视化（可选）
# ================================
plt.figure(figsize=(10, 6))
xgb_importances = model.feature_importances_
indices = np.argsort(xgb_importances)[-15:]  # 显示最重要的15个特征
plt.barh(range(len(indices)), xgb_importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Top 15 Important Features")
plt.show()
