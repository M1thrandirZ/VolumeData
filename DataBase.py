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
            self.dataArray_int = dataMatrix  # 数据一维形式，int类型
            dataMatrix = dataMatrix.reshape((filedimention[0], filedimention[1], filedimention[2]))
            self.dataArray_bytes = dataArray  # 数据一维形式，bytes类型
            self.dataMatrix = dataMatrix  # 带位置信息的数据结构
            self.count = len(dataArray)  # 数据点的个数
        else:
            print("输入不正确")

    # 保存为二进制文件
    def Save_Data(self):
        if isinstance(self.dataArray_bytes, bytes):
            root = tk.Tk()
            root.withdraw()
            file_path = fd.asksaveasfilename()
            f = open(file=file_path, mode='wb')
            f.write(self.dataArray_bytes)
            f.close()

    # 对体数据进行插值，t为扩大系数
    def Interpolation(self, t):

        # 让原数据坐标膨胀
        datagrid = np.argwhere(self.dataMatrix > -1) * t
        # 数据点的值
        datavalue = self.dataArray_int
        # 构造插值后数据点的坐标
        res_x, res_y, res_z = np.mgrid[0:self.dataDimension[0] * t:1, 0:self.dataDimension[1] * t:1,
                              0:self.dataDimension[2] * t:1]
        self.dataArray_int = griddata(datagrid, datavalue, (res_x, res_y, res_z), method='nearest').reshape((1,-1))  # 更新实例的int型一维数据
        self.dataArray_bytes = self.dataArray_int.tobytes()  # 更新实例的bytes型一维数据
        self.count = len(self.dataArray_int)  # 更新实例的数据数量
        temp = np.array(
            [self.dataDimension[0] * t, self.dataDimension[1] * t, self.dataDimension[2] * t, self.dataDimension[3]])
        self.dataDimension = temp  # 更新数据维度属性
        self.dataMatrix = self.dataArray_int.reshape(  # 更新体数据的三维尺寸
            (self.dataDimension[0], self.dataDimension[0], self.dataDimension[0]))

    # 对数据进行缩减，t为缩减系数
    def downsample(self, t):

        # 降采样之后数据的尺寸
        newX = self.dataDimension[0] // t
        newY = self.dataDimension[1] // t
        newZ = self.dataDimension[2] // t

        if newX > 0 and newY > 0 and newZ > 0:
            res = np.zeros((newX, newY, newZ))
            for z in range(newZ):
                for y in range(newY):
                    for x in range(newX):
                        res[x, y, z] = self.dataMatrix[x * t, y * t, z * t]

        self.dataMatrix = res  # 更新体数据
        self.dataArray_int = self.dataMatrix.reshape((1, -1))  # 更新int型的一维数据
        self.dataArray_bytes = self.dataArray_int.tobytes()  # 更新bytes型的一维数据
        self.count = len(self.dataArray_int)  # 更新数据长度
        self.dataDimension = np.array([newX, newY, newZ, self.dataDimension[3]])  # 更新数据维度

