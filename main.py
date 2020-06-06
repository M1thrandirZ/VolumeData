import DataBase as DB
import PlotBase as PB
import numpy as np

test = DB.VolumeData("/Users/zhangjunda/Desktop/volume_data/raw/Bucky32_32_32_8.raw", np.array([32, 32, 32, 8]))
# PB.DrawData_Scatter(test,150)
# PB.DrawData_voxels(test, 150)
#test.Save_Data(test.dataArray)
test.Interpolation()
