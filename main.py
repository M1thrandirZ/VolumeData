import DataBase as DB
import PlotBase as PB
import numpy as np
import sys

test = DB.VolumeData("/Users/zhangjunda/Desktop/volume_data/raw/Bucky32_32_32_8.raw", np.array([32, 32, 32, 8]))

# test.Save_Data()
test.Interpolation(2)
# test.downsample(2)
# PB.DrawData_Scatter(test, 200)
# PB.DrawData_voxels(test, 200)

# sys.exit()
