import json
from pathlib import Path

def update_cam_meta(cam_meta_path, c2ws_path, intrinsics_path):
    # Convert paths to Path objects
    cam_meta_path = Path(cam_meta_path)
    c2ws_path = Path(c2ws_path)
    
    # Load JSON data from files
    with cam_meta_path.open('r') as file:
        cam_meta = json.load(file)
    
    with c2ws_path.open('r') as file:
        c2ws = json.load(file)
    
    # Get the camera serial number from camMeta
    cam_serial = cam_meta['calibration']['cam_serial']
    
    # Step 2: Loop through each entry in camMeta.json and look up corresponding c2w matrix
    for data in cam_meta['data']:
        # Extract the filename (e.g., "0000000003.jpeg")
        filename = data['filename']
        # Use pathlib to get the suffix (e.g., "0000000003")
        file_suffix = Path(filename).stem
        # Construct the key to look up in c2ws.json
        key_to_find = f"{cam_serial}/{file_suffix}"
        
        # Check if this key exists in c2ws.json
        if key_to_find in c2ws:
            # Update the entry with hasLearnedPose and c2w matrix
            data['hasLearnedPose'] = True
            data['c2w'] = c2ws[key_to_find]
    
    # Step 3: Define the output file path using the camera serial number
    # output_file_path = cam_meta_path.parent / f"{cam_serial}_camMeta.json"
    output_file_path = Path(f"{cam_serial}_camMeta.json")
    
    # Step 4: Write back the modified camMeta to the new file
    with output_file_path.open('w') as file:
        json.dump(cam_meta, file, indent=4)
    
    print(f"Updated file saved as: {output_file_path}")

# Example Usage
update_cam_meta('/home/vi/workspace/dConstruct/data/02-09-2024-pixel_lvl1_water2_resampled_prune_x2_EXTRA/raw/DECXIN2023012346/camMeta.json', 
                '/home/vi/workspace/dConstruct/data/02-09-2024-pixel_lvl1_water2_resampled_prune_x2_EXTRA/outputs/git_8e2af30/export/c2ws.json', 
                '/home/vi/workspace/dConstruct/data/02-09-2024-pixel_lvl1_water2_resampled_prune_x2_EXTRA/outputs/git_8e2af30/export/intrinsics.json')
