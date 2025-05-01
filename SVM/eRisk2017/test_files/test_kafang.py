import numpy as np
from scipy.stats import chi2_contingency
from scipy import stats
import pandas as pd

# 创建一个虚拟数据集
X = np.random.randint(low=0, high=2, size=(10, 5))
y = np.random.randint(low=0, high=2, size=(10,))

# 计算每个特征与目标变量之间的卡方值和p值
for i in range(X.shape[1]):
    contingency_table = pd.crosstab(X[:,i], y)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Feature {i}: chi2={chi2}, p={p}")           

# 根据设定的显著性水平，选择卡方值和p值都超过该水平的特征
significant_features = []
alpha = 0.1
for i in range(X.shape[1]):
    contingency_table = pd.crosstab(X[:,i], y)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    if p < alpha and chi2 > stats.chi2.ppf(q=1-alpha, df=dof):
        significant_features.append(i)

print(f"Significant features: {significant_features}")
