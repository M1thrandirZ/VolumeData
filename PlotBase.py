import DataBase as DB
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk as numpy2vtk


# 散点图画体数据，仅绘制大于t的数据
def DrawData_Scatter(data: DB.VolumeData, t):  # 传入定义好的VolumeData类的实例，t为绘制点的最小数值
    """对dataMatrix进行plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 制作数据的列向量，4列分别为x坐标、y坐标、z坐标、数值
    # 先计算满足条件t的数据个数
    con = 0
    for i in range(data.count):
        if data.dataArray_bytes[i] > t:
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


# voxel的方式画体数据，仅绘制大于t的数据
def DrawData_voxels(data: DB.VolumeData, t):  # 传入定义好的VolumeData类的实例，t为绘制点的最小数值
    """对dataMatrix进行plot"""
    fig = plt.figure(figsize=(10, 8))
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
    hsv[..., 0] = data.dataMatrix / 255  # hue,色相
    hsv[..., 1] = 0.5  # saturation，饱和度，范围[0,1]
    hsv[..., 2] = 0.8  # value，明度，范围[0,1]
    color = matplotlib.colors.hsv_to_rgb(hsv)

    ax.voxels(filled
              , facecolors=color
              # , edgecolors=np.clip(2 * color - 0.5, 0, 1)
              )
    plt.show()


# 面绘制，t为等值面的值，精度不好，不如VTK
def DrawISO(data: DB.VolumeData, t):
    # verices顶点，triangle三角形，顶点向量的标号，每一行代表一个三角形
    ver, tri = data.MarshingCubes(t)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(np.shape(tri)[0]):
        xArray = np.array([ver[tri[i, 0], 0], ver[tri[i, 1], 0], ver[tri[i, 2], 0]])
        yArray = np.array([ver[tri[i, 0], 1], ver[tri[i, 1], 1], ver[tri[i, 2], 1]])
        zArray = np.array([ver[tri[i, 0], 2], ver[tri[i, 1], 2], ver[tri[i, 2], 2]])
        # 判断三个点是否有相同的，有则这个三角形没法画
        if not (
                (ver[tri[i, 0], :] == ver[tri[i, 1], :]).all()
                or (ver[tri[i, 0], :] == ver[tri[i, 2], :]).all()
                or (ver[tri[i, 1], :] == ver[tri[i, 2], :]).all()):
            # print(i)
            try:
                # 精度不够，matplotlib处理的时候不知道怎么回事，会认为相近的两个点是同一个点。
                ax.plot_trisurf(xArray, yArray, zArray, linewidth=2, antialiased=True, color='#0000ee88')
            except Exception as e:
                print("i=%s,%s" % (i, e))
                pass
            continue

    plt.show()


def VTKMarshingCubes(data: DB.VolumeData, t):
    # # 1. 读取数据
    # cube = vtk.vtkCubeSource()
    # cube.Update()  # 记得加这句不加看不到模型
    # # 2. 建图（将点拼接成立方体）
    # cube_mapper = vtk.vtkPolyDataMapper()
    # cube_mapper.SetInputData(cube.GetOutput())
    # # 3. 根据2创建执行单元
    # cube_actor = vtk.vtkActor()
    # cube_actor.SetMapper(cube_mapper)
    #
    # cube_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    # # 4. 渲染（将执行单元和背景组合在一起按照某个视角绘制）
    # renderer = vtk.vtkRenderer()
    # renderer.SetBackground(0.0, 0.0, 0.0)  # 背景只有一个所以是Set()
    # renderer.AddActor(cube_actor)  # 因为actor有可能为多个所以是add()
    #
    # # 5. 显示渲染窗口
    # render_window = vtk.vtkRenderWindow()
    # render_window.SetWindowName("My First Cube")
    # render_window.SetSize(400, 400)
    # render_window.AddRenderer(renderer)  # 渲染也会有可能有多个渲染把他们一起显示
    # # 6. 创建交互控键（可以用鼠标拖来拖去看三维模型）
    # interactor = vtk.vtkRenderWindowInteractor()
    # interactor.SetRenderWindow(render_window)
    # interactor.Initialize()
    # render_window.Render()
    # interactor.Start()

    # 从numpy得到vtk的数组数据类型
    vtkdataArray = numpy2vtk(num_array=data.dataMatrix.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    # 定义一种source，vtkImageData
    vtkImageData = vtk.vtkImageData()
    # 定义vtkImageData的各种属性
    vtkImageData.SetDimensions(np.shape(data.dataMatrix))
    vtkImageData.SetSpacing(1, 1, 1)
    vtkImageData.GetPointData().SetScalars(vtkdataArray)

    # 从vtkImageData返回numpy数组的过程
    # vtkPointData_temp=vtkImageData.GetPointData()
    # vtkFloatArray_temp=vtkPointData_temp.GetScalars()
    # numpy_temp=vtk2numpy(vtkFloatArray_temp)
    # res=numpy_temp==self.dataMatrix.flatten()
    # print(res)

    # 定义vtkMarchingCubes这个filter
    vtkMC = vtk.vtkMarchingCubes()
    vtkMC.SetInputData(vtkImageData)  # 设置输入数据（vtkImageData）
    vtkMC.SetNumberOfContours(1)  # 设置等值面的数量
    vtkMC.SetValue(0, t)  # 设置等值面的值（等值面索引，等值面值）

    # 1原始版本
    vtkMC.ComputeGradientsOn()  # 计算梯度
    vtkMC.ComputeNormalsOff()  # 计算法向量时，绘图质量反而下降
    vtkMC.ComputeScalarsOff()  # 开关无影响

    vtkNorm = vtk.vtkVectorNorm()  #
    vtkNorm.SetInputConnection(vtkMC.GetOutputPort())
    vtkNorm.Update()

    cubeMapper = vtk.vtkDataSetMapper()
    cubeMapper.SetInputConnection(vtkNorm.GetOutputPort())

    # 2使用vtkMarchingCubes计算标量值
    # vtkMC.ComputeGradientsOn()
    # vtkMC.ComputeNormalsOff()
    # vtkMC.ComputeScalarsOn()
    #
    # cubeMapper = vtk.vtkDataSetMapper()
    # cubeMapper.SetInputConnection(vtkMC.GetOutputPort())

    # 3不计算标量值
    # vtkMC.ComputeGradientsOn()
    # vtkMC.ComputeNormalsOn()
    # vtkMC.ComputeScalarsOff()
    #
    # cubeMapper = vtk.vtkDataSetMapper()
    # cubeMapper.SetInputConnection(vtkMC.GetOutputPort())

    cubeActor = vtk.vtkActor()
    cubeActor.SetMapper(cubeMapper)

    cubeCamera = vtk.vtkCamera()
    cubeCamera.SetPosition(1, 1, 1)
    cubeCamera.SetFocalPoint(0, 0, 0)

    Render = vtk.vtkRenderer()
    Render.AddActor(cubeActor)
    Render.SetActiveCamera(cubeCamera)
    Render.ResetCamera()
    Render.SetBackground(0.7, 0.7, 0.7)

    Win = vtk.vtkRenderWindow()
    Win.AddRenderer(Render)
    Win.SetSize(3000, 3000)

    Inter = vtk.vtkRenderWindowInteractor()
    Inter.SetRenderWindow(Win)

    # 设置小的坐标系，跟随交互改变
    axes = vtk.vtkAxesActor()
    widet = vtk.vtkOrientationMarkerWidget()
    widet.SetOrientationMarker(axes)  # 设置谁讲被在挂件中显示
    widet.SetOutlineColor(0.5, 0.5, 0.5)  # 挂件被选中时的外框颜色
    widet.SetInteractor(Inter)  # 选择vtkRenderWindowInteractor
    widet.SetViewport(0, 0, 0.3, 0.3)  # 挂件显示的位置和大小，(xmin,ymin,xmax,ymax)
    widet.SetEnabled(1)  # 使能
    widet.InteractiveOn()  # 交互开

    Win.Render()
    Inter.Start()
