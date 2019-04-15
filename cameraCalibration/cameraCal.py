import numpy as np
from scipy.interpolate import griddata

data = np.loadtxt("positions.txt", delimiter=',', skiprows=1)

points = []
xValues = []
yValues = []
widthGrid, heightGrid = np.mgrid[0:640:40, 0:480:40]

nrPoints = data.shape[0]
for i in range(nrPoints):
  points.append((data[i,0], data[i,1]))
  xValues.append(data[i,2])
  yValues.append(data[i,3])

xValues = np.array(xValues)
yValues = np.array(yValues)
xGrid = griddata(points, xValues, (widthGrid, heightGrid), method='linear')
yGrid = griddata(points, yValues, (widthGrid, heightGrid), method='linear')

np.savetxt("widthGrid.csv", widthGrid)
np.savetxt("heightGrid.csv", heightGrid)
np.savetxt("xGrid.csv", xGrid)
np.savetxt("yGrid.csv", yGrid)
