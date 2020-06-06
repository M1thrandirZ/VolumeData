# 体数据的相关操作
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
from scipy.interpolate import griddata


class VolumeData:
    """体数据的类"""

    # 初始化函数
    def __init__(self, filepath, filedimention):
        # 简单判断输入是否正确
        if isinstance(filepath, str) and isinstance(filedimention, np.ndarray):
            self.dataName = str(filepath).split('/')[-1]  # 文件名
            self.dataDimension = filedimention  # 数据维度，x、y、z和每个位置的字节数（8或16）
            self.dataPath = filepath  # 数据路径
            f = open(file=self.dataPath, mode='rb')
            dataArray = f.read()
            f.close()
            # 制作体数据，将数值放在对应的坐标位置
            dataMatrix = np.zeros(len(dataArray))
            for k in range(0, len(dataArray)):
                dataMatrix[k] = dataArray[k]
            dataMatrix = dataMatrix.reshape((filedimention[0], filedimention[1], filedimention[2]))
            self.dataArray = dataArray  # 数据的线性结构
            self.dataMatrix = dataMatrix  # 带位置信息的数据结构
            self.count = len(dataArray)  # 数据点的个数
        else:
            print("输入不正确")

    # 保存为二进制文件
    def Save_Data(self):
        if isinstance(self.dataArray, bytes):
            root = tk.Tk()
            root.withdraw()
            file_path = fd.asksaveasfilename()
            f = open(file=file_path, mode='wb')
            f.write(self.dataArray)
            f.close()

    # 对体数据进行插值，边界处的数据保持不变，仅在体数据内部插值，修改dataMatrix和dataArray
    def Interpolation(self):
        # 构造插值后的矩阵
        # res = np.zeros((self.dataDimension[0] * 2 - 1, self.dataDimension[1] * 2 - 1, self.dataDimension[2] * 2 - 1))
        # # 先将6个面上的原数据填入
        # res[0::2, 0::2, 0] = self.dataMatrix[:, :, 0]
        # res[0::2, 0::2, self.dataDimension[2] * 2 - 2] = self.dataMatrix[:, :, self.dataDimension[2] - 1]
        # res[0::2, 0, 0::2] = self.dataMatrix[:, 0, :]
        # res[0::2, self.dataDimension[1] * 2 - 2, 0::2] = self.dataMatrix[:, self.dataDimension[1]-1, :]
        # res[0, 0::2, 0::2] = self.dataMatrix[0, :, :]
        # res[self.dataDimension[0] * 2 - 2, 0::2, 0::2] = self.dataMatrix[self.dataDimension[0]-1, :, :]

        # #插入原始的数据点
        # for i in range(0,self.dataDimension[2]):
        #     res[0::2, 0::2, 2*i] = self.dataMatrix[:, :, i]

        # self.dataMatrix = res

