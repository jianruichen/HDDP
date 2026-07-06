import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建数据
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(-5, 5, 100)
U, V = np.meshgrid(u, v)
X = np.sqrt(1 + V**2) * np.cos(U)
Y = np.sqrt(1 + V**2) * np.sin(U)
Z = V

# 创建图像
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
ax.plot_surface(X, Y, Z, cmap='viridis')

# 设置坐标轴范围
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])

# 设置坐标轴标签
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 设置标题
ax.set_title("双曲空间的立体图形 (双曲面模型)")

# 显示图像
plt.show()