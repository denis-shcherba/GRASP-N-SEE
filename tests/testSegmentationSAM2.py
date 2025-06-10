import robotic as ry
import numpy as np
from segmenation.utils import show_points, show_masks
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
komo.addObjective([], ry.FS.negDistance, ['r_gripper', 'l_gripper'], ry.OT.ineq, [1], [-.5])
komo.addObjective([], ry.FS.scalarProductZZ, ['l_gripper', 'exampleObject'], ry.OT.eq, [1])
komo.addObjective([], ry.FS.positionRel, ['exampleObject', CAMERA], ry.OT.eq, [1, 1 , 0], [0, 0, 0])


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




bot.moveTo(qKomo[-1])
while bot.getTimeToEnd() > 0:
    bot.sync(C, .1)
image, depth = bot.getImageAndDepth(CAMERA)

points = ry.depthImage2PointCloud(depth, bot.getCameraFxycxy(CAMERA))

pcl.setPointCloud(points, image)
pcl.setColor([1,0,0])
bot.sync(C, .1)




import matplotlib.pyplot as plt

plt.imshow(image)
plt.title("Camera Image")
plt.axis('off')
plt.show()

import torch


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "/home/denis/git/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

print(image.shape)
predictor.set_image(image)

input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])  # center of the image
input_label = np.array([1])

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()  

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

filtered_depth_zero = depth.copy()
filtered_depth_zero[masks[0] == 0] = 0 # Where mask is 0, set depth_image to 0
points = ry.depthImage2PointCloud(filtered_depth_zero, bot.getCameraFxycxy(CAMERA))

pcl.setPointCloud(points, image)
pcl.setColor([1,0,0])
bot.sync(C, .1)

C.view(True)