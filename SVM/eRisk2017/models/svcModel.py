from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# 加载数据集
data = datasets.load_breast_cancer()
X = data.data
Y = data.target

# 数据归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 定义SVM模型
model = LinearSVC(C=1,max_iter=1000)

# 训练模型
model.fit(x_train, y_train)

# 测试模型
y_pred = model.predict(x_test)
accuracy = (y_pred == y_test).mean()
print('Accuracy:', accuracy)
# Accuracy: 0.9473684210526315
