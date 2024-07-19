import json
import torch
import numpy as np
import plotly.graph_objects as go
from nerfstudio.data.utils.colmap_parsing_utils import read_model, qvec2rotmat
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F

# Load the JSON data
with open("/Users/vaibhavholani/development/computer_vision/dConstruct/data/2024-06-19/undistorted/meta.json", "r") as file:
    meta_data = json.load(file)

# Create a dictionary to map image names to LiDAR positions
image_to_lidar_position = {}
for i, entry in enumerate(meta_data["data"]):
    for lidar_position, image_name in entry["imgName"].items():
        image_to_lidar_position[image_name] = i

# Extract the calibration data
calibration_info = meta_data["calibrationInfo"]

# Create the camera extrinsic matrices dictionary
camera_extrinsics = {}
for cam_serial, calib in calibration_info.items():
    qw = calib["qw"]
    qx = calib["qx"]
    qy = calib["qy"]
    qz = calib["qz"]
    qvec = torch.Tensor([qw, qx, qy, qz])

    x = calib["x"]
    y = calib["y"]
    z = calib["z"]
    tvec = (x, y, z)

    rot_matrix = qvec2rotmat((qw, qx, qy, qz))
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rot_matrix
    extrinsic_matrix[:3, 3] = [x, y, z]
    
    # Convert to a PyTorch tensor
    extrinsic_tensor = torch.tensor(extrinsic_matrix, dtype=torch.float64)
    extrinsic_tensor.requires_grad_(True)


    camera_extrinsics[cam_serial] = extrinsic_tensor


# Load the COLMAP model
cameras, images, points3D = read_model("/Users/vaibhavholani/development/computer_vision/dConstruct/data/2024-06-19/subregion1/colmap/BA/prior_sparse")


# For each image, create its extrinsic matrix and store it with PyTorch
for image in images.values():
    image.extrinsic_matrix = torch.eye(4, dtype=torch.float64)
    image.extrinsic_matrix[:3, :3] = torch.tensor(image.qvec2rotmat(), dtype=torch.float64)
    image.extrinsic_matrix[:3, 3] = torch.tensor(image.tvec, dtype=torch.float64)
    # Set the bottom row
    
    image.extrinsic_matrix.requires_grad_(False)

    # Invert the extrinsic matrix (from c2w to w2c)
    image.extrinsic_matrix = torch.inverse(image.extrinsic_matrix)

    # Transform the axis to match the LiDAR coordinate system
    transform = torch.tensor([[-1, 0, 0, 0],
                                        [0, 0, -1, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, 0, 1]], dtype=torch.float64)
    
    image.extrinsic_matrix = transform @ image.extrinsic_matrix


# Initialize a dictionary to store lists of images for each LiDAR position
images_for_each_lidar_position = {}

# Iterate over the images from COLMAP
for image in images.values():
    image_name = image.name
    lidar_position = image_to_lidar_position.get(image_name)
    
    if lidar_position is not None:
        if lidar_position not in images_for_each_lidar_position:
            images_for_each_lidar_position[lidar_position] = []
        images_for_each_lidar_position[lidar_position].append(image)



# Convert the dictionary to a list of lists
images_for_each_lidar_position_list = list(images_for_each_lidar_position.values())

# Print the result for verification
for idx, lidar_images in enumerate(images_for_each_lidar_position_list):
    print(f"LiDAR Position {idx+1}: {len(lidar_images)} images")

# Initialize extrinsic matrices for LiDAR positions
lidar_extrinsic_matrices = []
for images in images_for_each_lidar_position_list:
    image = images[0]

    lidar_extrinsic = image.extrinsic_matrix.clone().detach()
    lidar_extrinsic.requires_grad_(True)

    lidar_extrinsic_matrices.append(lidar_extrinsic)

    
# extract lidar camera poses

# Update the cameras with the correct extrinsic matrices
for lidar_images in images_for_each_lidar_position_list:
    for camera_img in lidar_images:
        camera_img_name = camera_img.name
        camera_name = camera_img_name.split("/")[0]
        camera_id = camera_img.camera_id
        if camera_name in camera_extrinsics:
            cameras[camera_id].extrinsic = camera_extrinsics[camera_name]


def dummy_loss():
    total_loss = torch.tensor(0.0, requires_grad=True)
    
    for lidar_pose, imgs in enumerate(images_for_each_lidar_position_list):
        for img in imgs:
            cam_extrinsic = cameras[img.camera_id].extrinsic
            c2w_reconstructed = torch.matmul(lidar_extrinsic_matrices[lidar_pose], cam_extrinsic)
            loss = torch.nn.functional.mse_loss(c2w_reconstructed, img.extrinsic_matrix)
            total_loss = total_loss + loss 
    
    return total_loss

# Define the initial learning rates
learning_rate_lidar = 0.05
learning_rate_camera = 0.00  # Initially 0

num_iterations = 3000

camera_params = list(camera_extrinsics.values())

# Create optimizer with parameter groups
optimizer = torch.optim.Adam([
    {'params': lidar_extrinsic_matrices, 'lr': learning_rate_lidar},
    {'params': camera_params, 'lr': learning_rate_camera}
])

# Create a scheduler that reduces the learning rate when the loss stops improving

scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(1 / 2) ** (1 / (num_iterations / 10))
    )

for i in range(num_iterations):

    optimizer.zero_grad()
    loss = dummy_loss()
    loss.backward()
    optimizer.step()

    # Update the learning rate scheduler
    scheduler.step()

    if i % 10 == 0:
        print(f"Iteration {i}: Loss = {loss.item()}")

    # After the first 100 iterations, update the learning rate for the camera extrinsic matrices
    if i == 100:
        for param_group in optimizer.param_groups:
            if param_group['lr'] == learning_rate_camera:
                param_group['lr'] = learning_rate_lidar

print("Optimization complete.")


# Create all new world-to-camera matrices
# new_world_to_camera_matrices = []

# for lidar_pose, imgs in enumerate(images_for_each_lidar_position_list):
#     for img in imgs:
#         cam_extrinsic = qtvec_to_ext(cameras[img.camera_id].extrinsic["qvec"], cameras[img.camera_id].extrinsic["tvec"])
#         lidar_extrinsic = qtvec_to_ext(lidar_qtvec[lidar_pose]["qvec"], lidar_qtvec[lidar_pose]["tvec"])
#         c2w_reconstructed = torch.matmul(lidar_extrinsic, cam_extrinsic)
#         new_world_to_camera_matrices.append(c2w_reconstructed)
    
# # Extract worldTcamera poses from meta.json
# worldTcam_poses = []
# for entry in meta_data["data"]:
#     for cam_serial, pose in entry["worldTcam"].items():
#         qw = pose["qw"]
#         qx = pose["qx"]
#         qy = pose["qy"]
#         qz = pose["qz"]
#         px = pose["px"]
#         py = pose["py"]
#         pz = pose["pz"]

#         rot_matrix = qvec2rotmat((qw, qx, qy, qz)) # 3x3 matrix not 4x4
#         worldTcam_matrix = np.eye(4)
#         worldTcam_matrix[:3, :3] = rot_matrix
#         worldTcam_matrix[:3, 3] = [px, py, pz]

#         # convert into cameraTworld
#         worldTcam_poses.append(worldTcam_matrix)

# lidar_extrinsics = []
# for lidar_pose in lidar_qtvec:
#     lidar_extrinsic = qtvec_to_ext(lidar_pose["qvec"], lidar_pose["tvec"])
#     lidar_extrinsics.append(lidar_extrinsic)

# camera_extrinsics = []
# for cam_pose in cam_qtvec.values():
#     cam_extrinsic = qtvec_to_ext(cam_pose["qvec"], cam_pose["tvec"])
#     camera_extrinsics.append(cam_extrinsic)


# Save all the lidar extrinsic matrices and camera extrinsic matrices to a file
# torch.save({
#     "lidar_extrinsics": lidar_extrinsics,
#     "cam_extrinsics": camera_extrinsics,
#     "new_world_to_camera_matrices": new_world_to_camera_matrices
# }, "extrinsic_matrices.pth")


# TODO: Check if Noise is added to the data, what's the effect on the optimization