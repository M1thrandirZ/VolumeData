import vtkmodules.all as vtk

import DataBase as DB
import PlotBase as PB
import numpy as np
import sys

# 体绘制部分
test = DB.VolumeData("/Users/zhangjunda/Desktop/volume_data/raw/Tooth_256_256_161_16.raw", np.array([256, 256, 161, 16]))

# 高斯化
# test.GaussDataMatrix(5)
# test.GaussData(np.array([120,64,32]),10)

# test.ChangeRegionData(np.array([128, 64, 32]), np.array([3, 3, 3]), 255)

# test.Save_Data()
# test.Interpolation(2)
# test.downsample(2)
# PB.DrawData_Scatter(test, 200)
# PB.DrawData_voxels(test, 100)
# PB.DrawISO(test,50)
# vert,tri=test.MarshingCubes(200)
# PB.DrawVTKMarshingCubes(test,100)
# PB.DrawHistogram(test)
# PB.Draw2DHistogram(test,100)

# test.GenUnstructuredGrid(1000)
PB.DrawDelaunay3D(test,2000)

# test.ExtractVoxelsToUnstructuredGrid(100)

# PB.DrawVTKVolumeRendering(test) # 体绘制
# PB.DrawVTKUnstructuredVolumeRendering(test.ExtractVoxelsToUnstructuredGrid(100))
# sys.exit()

#图片处理部分
