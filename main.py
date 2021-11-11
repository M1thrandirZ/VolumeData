import vtkmodules.all as vtk

import DataBase as DB
import PlotBase as PB
import numpy as np
import sys

# 体绘制部分
test = DB.VolumeData("/Users/zhangjunda/Desktop/volume_data/raw/BluntFin_256_128_64_8.raw", np.array([256, 128, 64, 8]))
# test.SaveVTKFile()
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
# PB.DrawVTKMarshingCubes(test,800)
# PB.DrawHistogram(test)
# PB.Draw2DHistogram(test,100)

# PB.DrawDelaunay3D(test,800)

# test.ExtractVoxelsToUnstructuredGrid(100)

PB.DrawVTKVolumeRendering(test) # 体绘制
# PB.DrawVTKUnstructuredVolumeRendering(test.GenTetraUnstructuredGrid(1600))
# sys.exit()

#图片处理部分
