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
                    # dataMatrix[k] = dataArray[2 * k] * 16 ** 2 + dataArray[2 * k + 1]
                    dataMatrix[k] = int.from_bytes(dataArray[2 * k:2 * k + 2], byteorder='big', signed=False)

                self.dataArray_int = dataMatrix  # 数据一维形式，int类型
                dataMatrix = dataMatrix.reshape((filedimention[0], filedimention[1], filedimention[2]), order='F')
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

    def RandomUnstructuredDataSet(self, n: int, threshold=0):
        """
        随机抽取规则网格数据(非0数据)转换为非结构化体数据，n为抽取的样本数
        :param n:采样点的个数
        :param threshold: 阈值，小于这个值的数据点会被剔除，默认只剔除值为0的数据点
        :return:每一行是一个点
        """
        # 获取三个方向上的维度
        XDimension = self.dataDimension[0]
        YDimension = self.dataDimension[1]
        ZDimension = self.dataDimension[2]

        # 获得随机生成的位置下标
        Xindex = np.random.randint(0, XDimension, n)
        Yindex = np.random.randint(0, YDimension, n)
        Zindex = np.random.randint(0, ZDimension, n)

        UnstructuredDataSet = np.zeros((4, n)).astype(np.int)  # 前三行储存x、y、z位置，第四行储存值

        UnstructuredDataSet[0, :] = Xindex[:]
        UnstructuredDataSet[1, :] = Yindex[:]
        UnstructuredDataSet[2, :] = Zindex[:]

        zeroIndice = np.zeros(1).astype(np.int)
        # 在原matrix中取值，但要记录一下哪些位置的值小于threshold
        for i in range(0, n):
            value = self.dataMatrix[
                UnstructuredDataSet[0, i], UnstructuredDataSet[1, i], UnstructuredDataSet[2, i]]
            UnstructuredDataSet[3, i] = value

            if value <= threshold:
                zeroIndice = np.append(zeroIndice, i)

        zeroIndice = np.delete(zeroIndice, 0)  # 删除第一个初始值

        # 剔除值为0的部分
        UnstructuredDataSet = np.delete(UnstructuredDataSet, zeroIndice, 1)

        return UnstructuredDataSet.T

    def GenUnstructuredGrid(self, n: int, threshold=0):
        """
        将RandomUnstructuredDataSet方法抽取的点组成UnstructuredGrid数据
        :param n:抽取个数，剔除小于threshold的点，总个数比n小
        :param threshold:小于threshold的点将被剔除
        :return:vtkUnstructuredGrid
        """
        if threshold == 0:
            randomData = self.RandomUnstructuredDataSet(n)  # 先随机抽取n个非零值的数据点
        else:
            randomData = self.RandomUnstructuredDataSet(n, threshold)  # 先随机抽取n个非零值的数据点
        pointCount = np.size(randomData, 0)

        points = vtk.vtkPoints()
        for j in range(0, pointCount):
            points.InsertNextPoint(randomData[j, 0], randomData[j, 1], randomData[j, 2])

        # todo:重新写一个函数，每次抽取一个点的时候其实是抽取了8个点形成的一个voxel

        # 只是纯数据点，没有拓扑结构
        # voxel=vtk.vtkVoxel()
        # # numberOfIds=voxel.GetNumberOfIds()
        # for i in range(0,pointCount):
        #     voxel.GetPointIds().SetId(i,i)

        scalarArray = numpy2vtk(num_array=randomData[:, 3], array_type=vtk.VTK_FLOAT)

        uGrid = vtk.vtkUnstructuredGrid()
        uGrid.SetPoints(points)
        uGrid.GetPointData().SetScalars(scalarArray)
        # uGrid.InsertNextCell(voxel.GetCellType(), voxel.GetPointIds())

        return uGrid

    def ExtractVoxelsToUnstructuredGrid(self, n: int):

        # 获取三个方向上的维度
        XDimension = self.dataDimension[0]
        YDimension = self.dataDimension[1]
        ZDimension = self.dataDimension[2]
        UGrid = vtk.vtkUnstructuredGrid()  # 最后输出的非结构化体数据
        voxelArray = vtk.vtkCellArray()  # 体素的集合
        points = vtk.vtkPoints()  # 点的集合
        scalars = np.empty(0, dtype=float)

        for v in range(0, n):

            pointValues = np.zeros(8)
            voxel = vtk.vtkVoxel()  # 一个体素

            # 获得随机生成的位置下标，然后每个维度+1就可以得到一个voxel
            Xindex = np.random.randint(0, XDimension - 1)
            Yindex = np.random.randint(0, YDimension - 1)
            Zindex = np.random.randint(0, ZDimension - 1)

            # 8个顶点
            points.InsertNextPoint(Xindex, Yindex, Zindex)
            points.InsertNextPoint(Xindex + 1, Yindex, Zindex)
            points.InsertNextPoint(Xindex, Yindex + 1, Zindex)
            points.InsertNextPoint(Xindex, Yindex, Zindex + 1)
            points.InsertNextPoint(Xindex + 1, Yindex, Zindex + 1)
            points.InsertNextPoint(Xindex, Yindex + 1, Zindex + 1)
            points.InsertNextPoint(Xindex + 1, Yindex + 1, Zindex)
            points.InsertNextPoint(Xindex + 1, Yindex + 1, Zindex + 1)

            # 8个顶点的标量值
            pointValues[0] = self.dataMatrix[Xindex, Yindex, Zindex]
            pointValues[1] = self.dataMatrix[Xindex + 1, Yindex, Zindex]
            pointValues[2] = self.dataMatrix[Xindex, Yindex + 1, Zindex]
            pointValues[3] = self.dataMatrix[Xindex, Yindex, Zindex + 1]
            pointValues[4] = self.dataMatrix[Xindex + 1, Yindex, Zindex + 1]
            pointValues[5] = self.dataMatrix[Xindex, Yindex + 1, Zindex + 1]
            pointValues[6] = self.dataMatrix[Xindex + 1, Yindex + 1, Zindex]
            pointValues[7] = self.dataMatrix[Xindex + 1, Yindex + 1, Zindex + 1]

            scalars = np.hstack((scalars, pointValues))  # 连接两个array

            for i in range(8):
                voxel.GetPointIds().SetId(i, i + v)

            voxelArray.InsertNextCell(voxel)

        scalarArray = numpy2vtk(num_array=scalars, array_type=vtk.VTK_FLOAT)
        UGrid.SetPoints(points)
        UGrid.GetPointData().SetScalars(scalarArray)
        UGrid.SetCells(voxel.GetCellType(), voxelArray)

        return UGrid
