import robotic as ry
import numpy as np
from segmenation.utils import generate_uniform_cube_points, filter_points_with_multiple_capsules

print('robotic version:', ry.__version__, ry.compiled())

CAMERA = "camera"  # or "cameraWrist" 

C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandasTable.g'))

C.view(True)

capsule_data = []
for i in C.getFrameNames():
    print(i, C.getFrame(i).getShapeType())

    if "coll" not in i:
        C.delFrame(i)
    else:
        capsule_data.append((
            C.getFrame(i).getPose(), 
            C.getFrame(i).getSize()[0],  # half length
            C.getFrame(i).getSize()[1]   # radius
        ))

C.view(True)

pcl = C.addFrame('pcl')

points = generate_uniform_cube_points(1000000, (-.75, -1, .25), (.75, .5, 1.75))
pcl.setPointCloud(points)
C.view(True)

pose = C.getFrame("r_panda_coll3").getPose()
size = C.getFrame("r_panda_coll3").getSize()

print(pose, size)

points = filter_points_with_multiple_capsules(points, capsule_data, True)
pcl.setPointCloud(points)

C.view(True)