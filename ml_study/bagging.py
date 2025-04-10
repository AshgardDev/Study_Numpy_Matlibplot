import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 生成模拟回归数据
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)  # 正弦函数 + 噪声

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 单个决策树
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# BaggingRegressor
bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=10,           # 10 个基学习器
    max_samples=0.8,           # 每个子数据集用 80% 的样本
    max_features=1.0,          # 使用所有特征
    bootstrap=True,            # 有放回采样
    random_state=42
)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

# 输出结果
print("单个决策树 - MSE:", mse_dt, "R²:", r2_dt)
print("BaggingRegressor - MSE:", mse_bagging, "R²:", r2_bagging)

# 可视化
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(X_test, y_test, color='blue', label='真实值', alpha=0.5)
plt.plot(X_test, y_pred_dt, color='green', label='决策树预测')
plt.plot(X_test, y_pred_bagging, color='red', label='Bagging 预测')
plt.xlabel('X')
plt.ylabel('y')
plt.title('BaggingRegressor vs 单个决策树')
plt.legend()
plt.show()