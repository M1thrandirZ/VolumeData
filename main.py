import DataBase as DB
import PlotBase as PB
import numpy as np
import sys

test = DB.VolumeData("/Users/zhangjunda/Desktop/volume_data/raw/Carp_256_256_512_16.raw", np.array([256, 256, 512, 16]))

# test.Save_Data()
# test.Interpolation(2)
# test.downsample(2)
# PB.DrawData_Scatter(test, 200)
# PB.DrawData_voxels(test, 200)
# PB.DrawISO(test,50)
# vert,tri=test.MarshingCubes(200)
PB.VTKMarshingCubes(test,1600)


# sys.exit()
