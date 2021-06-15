# 体数据的相关操作
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
from scipy.interpolate import griddata
# import mcubes as mc
import struct
import vtk
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk as numpy2vtk
from vtkmodules.util.numpy_support import vtk_to_numpy as vtk2numpy


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
            if filedimention[3] == 8:  # 一个字节代表一个数值，如果每个数据是8bit
                dataMatrix = np.zeros(len(dataArray))
                for k in range(0, len(dataArray)):
                    dataMatrix[k] = dataArray[k]
                self.dataArray_int = dataMatrix  # 数据一维形式，int类型
                dataMatrix = dataMatrix.reshape((filedimention[0], filedimention[1], filedimention[2]), order='F')
                self.dataArray_bytes = dataArray  # 数据一维形式，bytes类型
                self.dataMatrix = dataMatrix  # 带位置信息的数据结构
                self.count = len(dataArray)  # 数据点的个数
            else:  # 如果每个数据是16bit
                count = int(len(dataArray) / 2)
                dataMatrix = np.zeros(count)
                for k in range(0, count):
                    dataMatrix[k] = dataArray[2 * k] * 16 ** 2 + dataArray[2 * k + 1]

                self.dataArray_int = dataMatrix  # 数据一维形式，int类型
                dataMatrix = dataMatrix.reshape((filedimention[0], filedimention[1], filedimention[2]))
                self.dataArray_bytes = dataArray  # 数据一维形式，bytes类型
                self.dataMatrix = dataMatrix  # 带位置信息的数据结构
                self.count = count  # 数据点的个数

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
        self.dataArray_int = griddata(datagrid, datavalue, (res_x, res_y, res_z), method='nearest').reshape(
            (1, -1))  # 更新实例的int型一维数据
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

    # 提取等值面，t是等值面的值，返回定点和三角形，精度不太高
    # def MarshingCubes(self, t):
    #     vertices, triangles = mc.marching_cubes(self.dataMatrix, t)
    #     return vertices, triangles

    # 生成一个或一组符合高斯分布的数
    def GenGauss(self, loc, scale, size):
        return np.random.normal(loc, scale, size)

    # 将体数据的每一个点都替换成一个高斯分布的随机值
    def GaussDataMatrix(self, scale):
        for i in range(0, self.count):
            if self.dataArray_int[i] != 0:
                self.dataArray_int[i] = self.GenGauss(self.dataArray_int[i], scale, 1)
        self.dataMatrix = self.dataArray_int.reshape(
            (self.dataDimension[0], self.dataDimension[1], self.dataDimension[2]))

    # 对矩阵中某一个位置进行高斯化
    def GaussData(self, location: np.ndarray, scale):
        if self.dataMatrix[location[0], location[1], location[2]] != 0:
            print("[" + str(location[0]) +
                  "," + str(location[1]) +
                  "," + str(location[2]) +
                  "]位置的数据值为" +
                  str(self.dataMatrix[location[0], location[1], location[2]]))
            self.dataMatrix[location[0], location[1], location[2]] = self.GenGauss(
                self.dataMatrix[location[0], location[1], location[2]], scale, 1)
            print("高斯化之后的值为" + str(self.dataMatrix[location[0], location[1], location[2]]))
        else:
            print("这个位置的数据为0")

    # 修改体数据里某一位置的数值，只对非零数据进行修改
    def ChangeData(self, location: np.ndarray, scale):
        if self.dataMatrix[location[0], location[1], location[2]] != 0:
            print("[" + str(location[0]) +
                  "," + str(location[1]) +
                  "," + str(location[2]) +
                  "]位置的数据值为" +
                  str(self.dataMatrix[location[0], location[1], location[2]]))
            self.dataMatrix[location[0], location[1], location[2]] = scale
            print("修改值为" + str(scale))
        else:
            print("这个位置的数据为0，不修改")

    # 修改某一区域内所有数据数值，start为区域起点，regionRange为区域向三个维度扩展的范围，scale为要修改的数值
    def ChangeRegionData(self, start: np.ndarray, regionRange: np.ndarray, scale):
        for i in range(0, regionRange[0]):
            for j in range(0, regionRange[1]):
                for k in range(0, regionRange[2]):
                    self.ChangeData(np.array([start[0] + i, start[1] + j, start[2] + k]), scale)
