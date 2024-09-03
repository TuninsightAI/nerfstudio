import json
from pathlib import Path
import numpy as np
from nerfstudio.data.utils.colmap_parsing_utils import rotmat2qvec

def update_cam_meta(cam_meta_path: Path, c2ws_path: Path, intrinsics_path: Path, convert_to_pose: bool =True):
    """
    Update the camMeta.json file with learned intrinsic and extrinsic data.

    Args:
        cam_meta_path (Path): Path to the camMeta.json file.
        c2ws_path (Path): Path to the c2ws.json file.
        intrinsics_path (Path): Path to the intrinsics.json file.
        convert_to_pose (bool): Convert the 4x4 transformation matrix to pose parameters (default: True).
    
    """
    # Convert paths to Path objects
    cam_meta_path = Path(cam_meta_path)
    c2ws_path = Path(c2ws_path)
    intrinsics_path = Path(intrinsics_path)
    
    # Load JSON data from files
    with cam_meta_path.open('r') as file:
        cam_meta = json.load(file)
    
    with c2ws_path.open('r') as file:
        c2ws = json.load(file)
    
    with intrinsics_path.open('r') as file:
        intrinsics = json.load(file)
    
    # Get the camera serial number from camMeta
    cam_serial = cam_meta['calibration']['cam_serial']
    
    # Step 2: Retrieve the intrinsic matrix for the camera
    if cam_serial in intrinsics:
        intrinsic_matrix = np.array(intrinsics[cam_serial])
    else:
        raise KeyError(f"Intrinsic matrix for camera serial {cam_serial} not found in intrinsics.json.")
    
    # Step 3: Add learned intrinsic data to the calibration section
    cam_meta['calibration']['has_learned_intrinsics'] = True
    cam_meta['calibration']['learned_intrinsics'] = intrinsic_matrix.tolist()  # Convert to list

    # Define the transformation matrix using NumPy
    transform = np.array([[-1, 0, 0, 0],
                          [0, 0, -1, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]], dtype=np.float32)

    # Step 4: Loop through each entry in camMeta.json and look up corresponding c2w matrix
    for data in cam_meta['data']:
        # Extract the filename (e.g., "0000000003.jpeg")
        filename = data['filename']
        # Use pathlib to get the suffix (e.g., "0000000003")
        file_suffix = Path(filename).stem
        # Construct the key to look up in c2ws.json
        key_to_find = f"{cam_serial}/{file_suffix}"
        
        # Check if this key exists in c2ws.json
        if key_to_find in c2ws:
            # Update the entry with hasLearnedExtrinsic
            data['hasLearnedExtrinsic'] = True
            
            # Retrieve and transform the c2w matrix
            c2w_matrix = np.array(c2ws[key_to_find])
            transformed_c2w = transform @ c2w_matrix  # Apply the transformation
            
            if convert_to_pose:
                # Convert 4x4 matrix into pose parameters
                qvec = rotmat2qvec(transformed_c2w[:3, :3])
                data['c2w'] = {
                    'qw': float(qvec[0]),
                    'qx': float(qvec[1]),
                    'qy': float(qvec[2]),
                    'qz': float(qvec[3]),
                    'px': float(transformed_c2w[0, 3]),
                    'py': float(transformed_c2w[1, 3]),
                    'pz': float(transformed_c2w[2, 3])
                }
            else:
                # Store the transformed c2w matrix
                data['c2w'] = transformed_c2w.tolist()  # Convert to list
    
    # Step 5: Define the output file path using the camera serial number
    output_file_path = Path(f"{cam_serial}_camMeta.json")
    
    # Step 6: Write back the modified camMeta to the new file
    with output_file_path.open('w') as file:
        json.dump(cam_meta, file, indent=4)
    
    print(f"Updated file saved as: {output_file_path}")

def update_all_cam_meta(raw_dir: str, c2ws_path: str, intrinsics_path: str, convert_to_pose: bool =True):
    """
    Update all camMeta.json files in a directory with learned intrinsic and extrinsic data.

    Args:
        raw_dir (str): Path to the raw directory containing the camMeta.json files.
        c2ws_path (str): Path to the c2ws.json file.
        intrinsics_path (str): Path to the intrinsics.json file.
        convert_to_pose (bool): Convert the 4x4 transformation matrix to pose parameters (default: True).
    
    """

    # Convert paths to Path objects
    raw_dir = Path(raw_dir)
    c2ws_path = Path(c2ws_path)
    intrinsics_path = Path(intrinsics_path)
    
    # Get a list of all the files in the directories
    slam_meta_file = raw_dir / 'slamMeta.json'

    # Load JSON data from files
    with slam_meta_file.open('r') as file:
        slam_meta = json.load(file)

        # load 4 camMeta folder names
        cam_meta_files = [raw_dir / cam_serial / 'camMeta.json' for cam_serial in slam_meta['camSerial']]

    # Loop through each camMeta file and update it
    for cam_meta_path in cam_meta_files:
        update_cam_meta(cam_meta_path, c2ws_path, intrinsics_path, convert_to_pose)

if __name__ == '__main__':

    update_all_cam_meta('/home/vi/workspace/dConstruct/data/02-09-2024-pixel_lvl1_water2_resampled_prune_x2_EXTRA/raw/',
                            '/home/vi/workspace/dConstruct/data/02-09-2024-pixel_lvl1_water2_resampled_prune_x2_EXTRA/outputs/git_8e2af30/export/c2ws.json',
                            '/home/vi/workspace/dConstruct/data/02-09-2024-pixel_lvl1_water2_resampled_prune_x2_EXTRA/outputs/git_8e2af30/export/intrinsics.json', convert_to_pose=True)

