import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_genres = pd.read_csv('data_genres_after_PU_learning.csv')
X_ = data_genres
label = np.array(data_genres['label'])
X_ = X_.drop(['anime_id', 'label'], axis=1)
X_ = X_.reset_index(drop=True)
X_ = X_.astype(float)
X_ = np.array(X_)
# Для начала отмасштабируем выборку
rows, cols = X_.shape

# центрирование - вычитание из каждого значения среднего по столбцу
means = X_.mean(0)
for i in range(rows):
    for j in range(cols):
        X_[i, j] -= means[j]

# деление каждого значения на стандартное отклонение
std = np.std(X_, axis=0)
for i in range(cols):
    for j in range(rows):
        X_[j][i] /= std[i]
        
# Найдем собственные векторы и собственные значения (англ. Eigenvalues)
covariance_matrix = X_.T.dot(X_)
eig_values, eig_vectors = np.linalg.eig(covariance_matrix)
# сформируем список кортежей (собственное значение, собственный вектор)
eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:, i]) for i in range(len(eig_values))]
# и отсортируем список по убыванию собственных значений
eig_pairs.sort(key=lambda x: x[0], reverse=True)    
eig_sum = sum(eig_values)
# Доля дисперсии, описываемая каждой из компонент
var_exp = [(i / eig_sum) * 100 for i in sorted(eig_values, reverse=True)]
# Кумулятивная доля дисперсии по компонентам
cum_var_exp = np.cumsum(var_exp)
print('Кумулятивная доля дисперсии по компонентам', cum_var_exp)
# Сформируем вектор весов из собственных векторов, соответствующих первым двум главным компонентам
W = np.hstack((eig_pairs[0][1].reshape(cols,1), eig_pairs[1][1].reshape(cols,1), eig_pairs[2][1].reshape(cols,1)))
# print(f'Матрица весов W:\n', W)
Z = X_.dot(W)
y = label

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

fig = plt.figure(figsize = (12, 9))
ax = fig.add_subplot(111, projection='3d')

for c, i, s, t in zip("rk", [0, 1], [1, 1], [0.1, 1]):
    ax.scatter(Z[y == i, 0], Z[y == i, 1], Z[y == i, 2], c=c, s=s, alpha=t)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

plt.show()
