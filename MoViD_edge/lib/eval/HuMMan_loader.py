import os
import cv2
import json
import numpy as np
import open3d as o3d

class HuMManLoader:
    def __init__(self, root_dir,start='p000439_a000050',end='p000506_a000224'):
        self.root_dir = root_dir
        self.session_list = self._get_sessions_in_range(start, end)
    
    def _get_sessions_in_range(self, start, end):
        """Generate a list of sessions in the given range."""
        sessions = []
        
        # Assuming sessions are named in a pattern like "p000439_a000050", 
        # extract session number and use it to compare.
        session_start_num = int(start.split('_')[0].split('p')[1])
        session_end_num = int(end.split('_')[0].split('p')[1])
        a_end_num = int(end.split('_')[1].split('a')[1])

        # Iterate over directories in the root_dir and pick those that match the range
        for session in sorted(os.listdir(self.root_dir)):
            if not os.path.isdir(os.path.join(self.root_dir, session)) or session == '.cache':
                continue
            if self._is_session_in_range(session, session_start_num, session_end_num,a_end_num):
                sessions.append(session)
        
        return sessions
    
    def _is_session_in_range(self, session_name, start_num, end_num,a_end_num):
        """Check if the session is within the specified range."""
        # Extract the numeric part after "p" and "a" from the session name.
        session_num = int(session_name.split('_')[0].split('p')[1])
        return start_num <= session_num < end_num or (session_num == end_num and a_end_num >= int(session_name.split('_')[1].split('a')[1]))

    def _get_sequence_path(self, seq_name):
        """Get path to the sequence directory."""
        seq_path = os.path.join(self.root_dir, seq_name)
        if not os.path.exists(seq_path):
            raise FileNotFoundError(f"Sequence '{seq_name}' not found in {self.root_dir}.")
        return seq_path

    def load_color_image(self, seq_name, kinect_id, frame_id):
        """Load the color image for a given sequence, Kinect ID, and frame."""
        seq_path = self._get_sequence_path(seq_name)
        color_path = os.path.join(seq_path, f'kinect_color/kinect_{kinect_id:03d}/{frame_id:06d}.png')
        color_bgr = cv2.imread(color_path)
        if color_bgr is None:
            raise FileNotFoundError(f"Color image not found: {color_path}")
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        return color_rgb
    
    def load_color_image_all_frame(self, seq_name,kinect_id):
        """Load the color image for a given sequence, Kinect ID, and frame."""
        seq_path = self._get_sequence_path(seq_name)
        color_path = os.path.join(seq_path, f'kinect_color/kinect_{kinect_id:03d}')
        color_rgbs=[]
        for frame in sorted(os.listdir(color_path)):
            color_bgr=cv2.imread(os.path.join(color_path, frame))
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            color_rgbs.append(color_rgb)
        color_rgbs=np.array(color_rgbs)

        return color_rgbs

    def load_images(self, seq_name, kinect_id):
        """Load the color image for a given sequence, Kinect ID, and frame."""
        seq_path = self._get_sequence_path(seq_name)
        color_path = os.path.join(seq_path, f'kinect_color/kinect_{kinect_id:03d}')
        images=[]
        for frame in sorted(os.listdir(color_path)):
            images.append(os.path.join(color_path, frame))
        return images

    def load_mask(self, seq_name, kinect_id, frame_id, manual=False):
        """Load the mask (manual or not) for a given sequence, Kinect ID, and frame."""
        seq_path = self._get_sequence_path(seq_name)
        mask_type = 'mask_manual' if manual else 'mask'
        mask_path = os.path.join(seq_path, f'kinect_{mask_type}/kinect_{kinect_id:03d}/{frame_id:06d}.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask image not found: {mask_path}")
        return mask

    def load_smpl_params(self, seq_name, frame_id):
        """Load the SMPL parameters for a given sequence and frame."""
        seq_path = self._get_sequence_path(seq_name)
        smpl_path = os.path.join(seq_path, f'smpl_params/{frame_id:06d}.npz')
        smpl_params = np.load(smpl_path)
        return smpl_params

    def load_smpl_of_all_frame(self, seq_name):
        """Load the SMPL parameters for a given sequence and frame."""
        seq_path = self._get_sequence_path(seq_name)
        smpl_path = os.path.join(seq_path, f'smpl_params')
        smpl_poses=[]
        global_orient=[]
        betas=[]
        transl = []
        for frame in sorted(os.listdir(smpl_path)):
            if frame.split('.')[-1]!='npz':
                continue
            smpl_params=np.load(os.path.join(smpl_path, frame))
            smpl_poses.append(smpl_params['body_pose'])
            global_orient.append(smpl_params['global_orient'])
            betas.append(smpl_params['betas'])
            transl.append(smpl_params['transl'])
        smpl_poses=np.array(smpl_poses)
        global_orient=np.array(global_orient)
        betas=np.array(betas)
        transl=np.array(transl)
        return smpl_poses,global_orient,betas,transl


    def load_cameras(self, seq_name):
        """Load the camera parameters from the cameras.json file."""
        seq_path = self._get_sequence_path(seq_name)
        cameras_path = os.path.join(seq_path, 'cameras.json')
        with open(cameras_path, 'r') as f:
            cameras = json.load(f)
        return cameras

    def load_depth_image(self, seq_name, kinect_id, frame_id):
        """Load the depth image for a given sequence, Kinect ID, and frame."""
        seq_path = self._get_sequence_path(seq_name)
        depth_path = os.path.join(seq_path, f'kinect_depth/kinect_{kinect_id:03d}/{frame_id:06d}.png')
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        return depth_image

    def depth_to_point_cloud(self, seq_name, kinect_id, frame_id):
        """Convert depth image to point cloud using camera intrinsics."""
        depth_image = self.load_depth_image(seq_name, kinect_id, frame_id)
        cameras = self.load_cameras(seq_name)
        camera_params = cameras[f'kinect_depth_{kinect_id:03d}']
        K, R, T = camera_params['K'], camera_params['R'], camera_params['T']

        # Initialize open3d camera parameters
        open3d_camera = o3d.camera.PinholeCameraParameters()
        open3d_camera.intrinsic.set_intrinsics(
            width=640, height=576, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

        # Generate point cloud
        depth_image_o3d = o3d.geometry.Image(depth_image)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image_o3d, open3d_camera.intrinsic, depth_trunc=5.0)
        return pcd

    def load_textured_mesh(self, seq_name, frame_id):
        """Load the textured mesh for a given sequence and frame."""
        seq_path = self._get_sequence_path(seq_name)
        textured_mesh_path = os.path.join(seq_path, f'textured_meshes/{frame_id:06d}.obj')
        if not os.path.exists(textured_mesh_path):
            raise FileNotFoundError(f"Textured mesh not found: {textured_mesh_path}")
        mesh = o3d.io.read_triangle_mesh(textured_mesh_path)
        return mesh

# Example usage:
if __name__ == '__main__':
    # Set your root directory path here
    root_dir = '/dataset/OpenXDLab___HuMMan/humman_release_v1.0_recon'

    loader = HuMManLoader(root_dir)
    
    # Example: Load color image
    seq_name = 'p000439_a000050'
    kinect_id = 0
    frame_id = 0
    color_image = loader.load_color_image(seq_name, kinect_id, frame_id)
    print(f"Color Image shape: {color_image.shape}")

    # # Example: Load mask image
    # mask_image = loader.load_mask(seq_name, kinect_id, frame_id, manual=True)
    # print(f"Mask Image shape: {mask_image.shape}")

    # Example: Load SMPL parameters
    smpl_params = loader.load_smpl_params(seq_name, frame_id)
    print(f"SMPL Global Orientation: {smpl_params['global_orient']}")

    # # Example: Load depth image and convert to point cloud
    # pcd = loader.depth_to_point_cloud(seq_name, kinect_id, frame_id)
    # print(f"Point Cloud size: {len(pcd.points)}")
    
    # # Example: Load textured mesh
    # mesh = loader.load_textured_mesh(seq_name, frame_id)
    # print(f"Mesh vertices: {len(mesh.vertices)}")
