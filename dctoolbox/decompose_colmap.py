import json
import torch
import numpy as np
import plotly.graph_objects as go
from nerfstudio.data.utils.colmap_parsing_utils import read_model, qvec2rotmat, rotmat2qvec
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F


def quat2rotation(qvec: torch.FloatTensor) -> torch.FloatTensor:
    # Extract the quaternion components
    w, x, y, z = qvec[0], qvec[1], qvec[2], qvec[3]
    
    # Create the rotation matrix
    rotmat = torch.tensor([
        [
            1 - 2 * y**2 - 2 * z**2,
            2 * x * y - 2 * w * z,
            2 * x * z + 2 * w * y
        ],
        [
            2 * x * y + 2 * w * z,
            1 - 2 * x**2 - 2 * z**2,
            2 * y * z - 2 * w * x
        ],
        [
            2 * x * z - 2 * w * y,
            2 * y * z + 2 * w * x,
            1 - 2 * x**2 - 2 * y**2
        ]
    ], dtype=torch.float64)
    
    return rotmat

    
    

def rotation2quat(R: torch.Tensor) -> torch.Tensor:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flatten()

    K = torch.tensor([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0

    eigvals, eigvecs = torch.linalg.eigh(K)
    qvec = eigvecs[:, torch.argmax(eigvals)]
    qvec = qvec[[3, 0, 1, 2]]
    
    if qvec[0] < 0:
        qvec *= -1

    return qvec

def qtvec_to_ext(quat, trans):
    """
    Convert a quaternion and translation vector to an extrinsic matrix
    """

    rot_matrix = quat2rotation(quat)
    extrinsic_matrix = torch.eye(4, dtype=torch.float64)
    extrinsic_matrix[:3, :3] = rot_matrix
    extrinsic_matrix[:3, 3] = trans

    return extrinsic_matrix
     

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
cam_qtvec = {}
for cam_serial, calib in calibration_info.items():
    qw = calib["qw"]
    qx = calib["qx"]
    qy = calib["qy"]
    qz = calib["qz"]
    qvec = (qw, qx, qy, qz)

    x = calib["x"]
    y = calib["y"]
    z = calib["z"]
    tvec = (x, y, z)

    quat = torch.tensor(qvec, dtype=torch.float64)
    quat.requires_grad_(True)
    trans = torch.tensor(tvec, dtype=torch.float64)
    trans.requires_grad_(True)


    cam_qtvec[cam_serial] = {"qvec": quat, "tvec": trans}

# Load the COLMAP model
cameras, images, points3D = read_model("/Users/vaibhavholani/development/computer_vision/dConstruct/data/2024-06-19/subregion1/colmap/BA/prior_sparse")

# For each image, create its extrinsic matrix and store it with PyTorch
for image in images.values():
    image.extrinsic_matrix = torch.eye(4, dtype=torch.float64)
    image.extrinsic_matrix[:3, :3] = torch.tensor(image.qvec2rotmat(), dtype=torch.float64)
    image.extrinsic_matrix[:3, 3] = torch.tensor(image.tvec, dtype=torch.float64)
    # Set the bottom row
    image.extrinsic_matrix[3, :] = torch.tensor([0, 0, 0, 1], dtype=torch.float64)
    
    image.extrinsic_matrix.requires_grad_(False)

    # Invert the extrinsic matrix (from c2w to w2c)
    image.extrinsic_matrix = torch.inverse(image.extrinsic_matrix)

    # Transform the axis to match the LiDAR coordinate system
    transform = torch.tensor([[-1, 0, 0, 0],
                                        [0, 0, -1, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, 0, 1]], dtype=torch.float64)
    
    image.extrinsic_matrix = transform @ image.extrinsic_matrix
    
    image.qvec_new = rotation2quat(image.extrinsic_matrix[:3, :3])
    image.tvec_new = image.extrinsic_matrix[:3, 3]


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
lidar_qtvec = []
for image in images_for_each_lidar_position_list:

    qvec_tensor = torch.tensor(image[0].qvec, dtype=torch.float64)
    qvec_tensor.requires_grad_(True)

    tvec_tensor = torch.tensor(image[0].tvec, dtype=torch.float64)
    tvec_tensor.requires_grad_(True)

    lidar_extrinsic = {"qvec": qvec_tensor, "tvec": tvec_tensor}

    lidar_qtvec.append(lidar_extrinsic)

    
# extract lidar camera poses

# Update the cameras with the correct extrinsic matrices
for lidar_images in images_for_each_lidar_position_list:
    for camera_img in lidar_images:
        camera_img_name = camera_img.name
        camera_name = camera_img_name.split("/")[0]
        camera_id = camera_img.camera_id
        if camera_name in cam_qtvec:
            cameras[camera_id].extrinsic = cam_qtvec[camera_name]

def dummy_loss():
    total_loss = torch.tensor(0.0, requires_grad=True)
    
    for lidar_pose, imgs in enumerate(images_for_each_lidar_position_list):
        for img in imgs:
            # colmap
            cam_extrinsic = qtvec_to_ext(cameras[img.camera_id].extrinsic["qvec"], cameras[img.camera_id].extrinsic["tvec"])
            lidar_extrinsic = qtvec_to_ext(lidar_qtvec[lidar_pose]["qvec"], lidar_qtvec[lidar_pose]["tvec"])

            c2w_reconstructed = torch.matmul(lidar_extrinsic, cam_extrinsic)
            c2w_reconstructed_qvec = rotation2quat(c2w_reconstructed[:3, :3])
            c2w_reconstructed_tvec = c2w_reconstructed[:3, 3]
            loss = torch.sum(torch.abs(c2w_reconstructed_qvec - img.qvec_new)) + torch.sum(torch.abs(c2w_reconstructed_tvec - img.tvec_new))
            total_loss = total_loss +  loss
            
    return total_loss

# Define the initial learning rates
learning_rate_lidar = 0.05
learning_rate_camera = 0.00  # Initially 0

num_iterations = 3000

# Create a list of parameters to optimize
lidar_qtvec_params = []
for lidar_extrinsic in lidar_qtvec:
    lidar_qtvec_params.append(lidar_extrinsic["qvec"])
    lidar_qtvec_params.append(lidar_extrinsic["tvec"])

camera_qtvec_params = []
for cam_extrinsic in cam_qtvec.values():
    camera_qtvec_params.append(cam_extrinsic["qvec"])
    camera_qtvec_params.append(cam_extrinsic["tvec"])

# Create optimizer with parameter groups
optimizer = torch.optim.Adam([
    {'params': lidar_qtvec_params, 'lr': learning_rate_lidar},
    {'params': camera_qtvec_params, 'lr': learning_rate_camera}
])

# Create a scheduler that reduces the learning rate when the loss stops improving
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
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
    if i == 200:
        for param_group in optimizer.param_groups:
            if param_group['lr'] == learning_rate_camera:
                param_group['lr'] = learning_rate_lidar

print("Optimization complete.")


# Create all new world-to-camera matrices
new_world_to_camera_matrices = []

for lidar_pose, imgs in enumerate(images_for_each_lidar_position_list):
    for img in imgs:
        cam_extrinsic = qtvec_to_ext(cameras[img.camera_id].extrinsic["qvec"], cameras[img.camera_id].extrinsic["tvec"])
        lidar_extrinsic = qtvec_to_ext(lidar_qtvec[lidar_pose]["qvec"], lidar_qtvec[lidar_pose]["tvec"])
        c2w_reconstructed = torch.matmul(lidar_extrinsic, cam_extrinsic)
        new_world_to_camera_matrices.append(c2w_reconstructed)
    
# Extract worldTcamera poses from meta.json
worldTcam_poses = []
for entry in meta_data["data"]:
    for cam_serial, pose in entry["worldTcam"].items():
        qw = pose["qw"]
        qx = pose["qx"]
        qy = pose["qy"]
        qz = pose["qz"]
        px = pose["px"]
        py = pose["py"]
        pz = pose["pz"]

        rot_matrix = qvec2rotmat((qw, qx, qy, qz)) # 3x3 matrix not 4x4
        worldTcam_matrix = np.eye(4)
        worldTcam_matrix[:3, :3] = rot_matrix
        worldTcam_matrix[:3, 3] = [px, py, pz]

        # convert into cameraTworld
        worldTcam_poses.append(worldTcam_matrix)

lidar_extrinsics = []
for lidar_pose in lidar_qtvec:
    lidar_extrinsic = qtvec_to_ext(lidar_pose["qvec"], lidar_pose["tvec"])
    lidar_extrinsics.append(lidar_extrinsic)

camera_extrinsics = []
for cam_pose in cam_qtvec.values():
    cam_extrinsic = qtvec_to_ext(cam_pose["qvec"], cam_pose["tvec"])
    camera_extrinsics.append(cam_extrinsic)


# Save all the lidar extrinsic matrices and camera extrinsic matrices to a file
torch.save({
    "lidar_extrinsics": lidar_extrinsics,
    "cam_extrinsics": camera_extrinsics,
    "new_world_to_camera_matrices": new_world_to_camera_matrices
}, "extrinsic_matrices.pth")


# TODO: Check if Noise is added to the data, what's the effect on the optimization