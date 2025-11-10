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
file_path = "/home/zhuzhenghao/all_combined_balancedRiscode_6000.csv"
df = pd.read_csv(file_path)

# ================================
# 2. 提取特征与标签
# ================================
if "riscode" not in df.columns:
    raise ValueError("数据中未找到名为 'riscode' 的列，请检查文件。")

X = df.drop(columns=["riscode"])
y = df["riscode"].map({10000: 1, 90000: 0})

# ================================
# 3. 自动检测并转换非数值列
# ================================
object_cols = X.select_dtypes(include=["object"]).columns
print(f"检测到 {len(object_cols)} 个非数值列，将进行独热编码: {list(object_cols)[:10]}...")

# 转换为哑变量
X = pd.get_dummies(X, drop_first=True)

# 移除有缺失值或异常值的列
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# 确保输入为 float32（避免 xgboost pandas dtype bug）
X = X.astype(np.float32)

# ================================
# 4. 划分训练集与测试集
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 5. 训练 XGBoost 分类器
# ================================
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    tree_method="hist"
)

# ✅ 关键：转换为 numpy 数组以彻底避免 pandas 兼容性问题
model.fit(X_train.to_numpy(), y_train.to_numpy())

# ================================
# 6. 模型预测与评估
# ================================
y_pred = model.predict(X_test.to_numpy())

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ 准确率(Accuracy): {acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵 (Confusion Matrix):")
print(cm)

print("\n分类报告 (Classification Report):")
print(classification_report(y_test, y_pred, digits=4))

# ================================
# 7. 可视化混淆矩阵
# ================================
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["10000(1)", "90000(0)"],
            yticklabels=["10000(1)", "90000(0)"])
plt.title("Confusion Matrix of XGBoost Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ================================
# 8. 可选：显示前15个最重要的特征
# ================================
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Top 15 Important Features")
plt.show()
