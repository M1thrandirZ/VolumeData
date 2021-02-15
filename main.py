import DataBase as DB
import PlotBase as PB
import numpy as np
import sys

test = DB.VolumeData("/Users/zhangjunda/Desktop/volume_data/raw/BluntFin_256_128_64_8.raw", np.array([256, 128, 64, 8]))

# test.Save_Data()
# test.Interpolation(2)
# test.downsample(2)
# PB.DrawData_Scatter(test, 200)
# PB.DrawData_voxels(test, 200)
# PB.DrawISO(test,50)
# vert,tri=test.MarshingCubes(200)
# PB.DrawVTKMarshingCubes(test,150)
# PB.DrawHistogram(test)
# PB.Draw2DHistogram(test,100)
PB.DrawVTKVolumeRendering(test)
# sys.exit()
