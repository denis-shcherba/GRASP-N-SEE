import robotic as ry
import numpy as np

print('robotic version:', ry.__version__, ry.compiled())

CAMERA = "l_cameraWrist"  # or "cameraWrist" 

C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandasTableCameraCalibrated.g'))

C.addFrame("exampleObject", "r_gripper").setShape(ry.ST.capsule, [.03, .02]).setColor([.1, 1, .1])

C.view(True)
qHome = C.getJointState()

komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=0, enableCollisions=False)
komo.addObjective(
    times=[], 
    feature=ry.FS.jointState, 
    frames=[],
    type=ry.OT.sos, 
    scale=[1e-1], 
    target=qHome
)
komo.addObjective([], ry.FS.positionRel, ['r_gripper', 'l_gripper'], ry.OT.eq, [1, 1, 0], [0 , 0, 0])
komo.addObjective([], ry.FS.negDistance, ['r_gripper', 'l_gripper'], ry.OT.ineq, [1], [-.5])
komo.addObjective([], ry.FS.scalarProductZZ, ['l_gripper', 'exampleObject'], ry.OT.eq, [1])

ret = ry.NLP_Solver(komo.nlp(), verbose=4) .solve()
print(ret)

qKomo = komo.getPath()
print(type(qKomo), len(qKomo))
C.setJointState(qKomo)
C.view(True)


pcl = C.addFrame('pcl', CAMERA)
C.addFrame("cameraWP", CAMERA).setShape(ry.ST.marker, [.1]) 

C.view(True)


C.setJointState(qHome)
bot = ry.BotOp(C, useRealRobot=False)

q = bot.get_qHome()
q[1] = q[1] + .2


pcl = C.getFrame("pcl")
pcl.setShape(ry.ST.pointCloud, [2]) # the size here is pixel size for display
bot.sync(C)

count = 0

while bot.getKeyPressed()!=ord('q'):
    image, depth, points = bot.getImageDepthPcl(CAMERA)
    pcl.setPointCloud(points, image)
    pcl.setColor([1,0,0])
    bot.sync(C, .1)
    
    bot.moveTo(qKomo[-1])

image, depth, points = bot.getImageDepthPcl(CAMERA)
import matplotlib.pyplot as plt

plt.imshow(image)
plt.title("Camera Image")
plt.axis('off')
plt.show()
