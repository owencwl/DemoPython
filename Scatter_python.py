from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 身高、体重、脚尺寸数据
x = [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75]
y = [180, 190, 170, 165, 100, 150, 130, 150]
z = [12, 11, 12, 10, 6, 8, 7, 9]

# 男性用红色园圈表示
ax.scatter(x[:4], y[:4], z[:4], c='r', marker='o', s=100)
# 女性用蓝色三角表示
ax.scatter(x[4:], y[4:], z[4:], c='b', marker='^', s=100)

ax.set_xlabel('Height (feet)')
ax.set_ylabel('Weight (lbs)')
ax.set_zlabel('Foot size (inches)')

# 显示散点图
plt.show()