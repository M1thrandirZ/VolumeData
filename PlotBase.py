import DataBase as DB
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


def DrawData_Scatter(data: DB.VolumeData, t):  # 传入定义好的VolumeData类的实例，t为绘制点的最小数值
    """对dataMatrix进行plot"""
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    # 制作数据的列向量，4列分别为x坐标、y坐标、z坐标、数值
    # 先计算满足条件t的数据个数
    con = 0
    for i in range(data.count):
        if data.dataArray[i] > t:
            con = con + 1
    # 构造用于绘制的数组
    dataColum = np.zeros((con, 4))  # 四列向量，前三个是点的空间位置，最后一个是数值
    k = 0
    for i in range(0, data.dataDimension[2]):
        for j in range(0, data.dataDimension[1]):
            for z in range(0, data.dataDimension[0]):
                if data.dataMatrix[z, j, i] > t:
                    dataColum[k, 0] = z
                    dataColum[k, 1] = j
                    dataColum[k, 2] = i
                    dataColum[k, 3] = data.dataMatrix[z, j, i]
                    k = k + 1

    ax.scatter(dataColum[:, 0], dataColum[:, 1], dataColum[:, 2], c=dataColum[:, 3] / 255, cmap="cool", alpha=0.2)

    plt.show()


def DrawData_voxels(data: DB.VolumeData, t):  # 传入定义好的VolumeData类的实例，t为绘制点的最小数值
    """对dataMatrix进行plot"""
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    # 制作标记有体素的三维矩阵
    filled = data.dataMatrix > t
    # 制作颜色的数组，颜色与体素数值相关
    # color = np.zeros(filled.shape + (4,))  # 为每一个位置的点，定义颜色（RGBA）
    # temp = color[..., 0]
    # color[..., 0] = data.dataMatrix / 255  # 红色分量
    # color[..., 1] = 0  # 绿色分量
    # color[..., 2] = 0  # 蓝色分量
    # color[..., 3] = np.clip(data.dataMatrix / 255,0,0.5)  # 透明度分量，上限截取在0.5

    hsv = np.zeros(filled.shape + (3,))
    hsv[..., 0] = data.dataMatrix / 255  # hue,色相，范围[0,1]
    hsv[..., 1] = 0.5  # saturation，饱和度，范围[0,1]
    hsv[..., 2] = 0.8  # value，明度，范围[0,1]
    color = matplotlib.colors.hsv_to_rgb(hsv)

    ax.voxels(filled
              , facecolors=color
              # , edgecolors=np.clip(2 * color - 0.5, 0, 1)
              )
    plt.show()
