import robotic as ry
import numpy as np
import numpy as np

# --- Helper function for quaternion rotation (remains unchanged) ---
def rotate_vector_by_quaternion(vec, quat):
    """
    Rotates a 3D vector by a quaternion.

    Args:
        vec (np.ndarray): A 3D vector (x, y, z).
        quat (np.ndarray): A quaternion (w, x, y, z).

    Returns:
        np.ndarray: The rotated 3D vector.
    """
    # Convert vector to a pure quaternion (0, x, y, z)
    vec_quat = np.array([0.0, vec[0], vec[1], vec[2]])

    # Normalize the quaternion (important for rotation)
    quat_norm = np.linalg.norm(quat)
    if quat_norm == 0:
        # If quaternion is zero (invalid), return original vector for safety
        return vec
    quat = quat / quat_norm

    # Conjugate of the quaternion
    quat_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])

    # Perform the rotation: q * p * q_conjugate
    # Quaternion multiplication (Hamilton product)
    # q1 = (w1, v1), q2 = (w2, v2)
    # q1 * q2 = (w1*w2 - v1.v2, w1*v2 + w2*v1 + v1 x v2)

    # First product: quat * vec_quat
    w1, x1, y1, z1 = quat
    w2, x2, y2, z2 = vec_quat

    prod1_w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
    prod1_x = w1 * x2 + x1 * w2 + (y1 * z2 - z1 * y2)
    prod1_y = w1 * y2 + y1 * w2 + (z1 * x2 - x1 * z2)
    prod1_z = w1 * z2 + z1 * w2 + (x1 * y2 - y1 * x2)
    prod1 = np.array([prod1_w, prod1_x, prod1_y, prod1_z])

    # Second product: prod1 * quat_conj
    w1, x1, y1, z1 = prod1
    w2, x2, y2, z2 = quat_conj

    prod2_w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
    prod2_x = w1 * x2 + x1 * w2 + (y1 * z2 - z1 * y2)
    prod2_y = w1 * y2 + y1 * w2 + (z1 * x2 - x1 * z2)
    prod2_z = w1 * z2 + z1 * w2 + (x1 * y2 - y1 * x2)
    
    # The rotated vector is the vector part of the resulting quaternion
    return np.array([prod2_x, prod2_y, prod2_z])


### Core Capsule Mask Generation

def _get_single_capsule_mask(point_cloud, p1, p2, radius):
    """
    Helper function to get a boolean mask for points inside a single capsule,
    given its two endpoints and radius.

    Args:
        point_cloud (np.ndarray): (N, 3) array of points.
        p1 (np.ndarray): First endpoint of capsule axis.
        p2 (np.ndarray): Second endpoint of capsule axis.
        radius (float): Radius of the capsule.

    Returns:
        np.ndarray: A boolean array of shape (N,) where True indicates
                    the point is inside the capsule.
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    
    # Vector representing the capsule's axis
    v = p2 - p1
    v_sq_len = np.dot(v, v)

    # If p1 and p2 are essentially the same, treat it as a sphere
    if v_sq_len < 1e-9: # Use a small epsilon for floating point comparison
        distances_sq = np.sum((point_cloud - p1)**2, axis=1)
        return distances_sq <= radius**2

    # Vector from p1 to each point in the point cloud
    w = point_cloud - p1

    # Calculate t for each point: t = (w . v) / (v . v)
    t = np.dot(w, v) / v_sq_len

    # Clamp t to [0, 1] to find the closest point on the line *segment*
    t_clamped = np.clip(t, 0, 1)

    # Calculate the closest point on the line segment for each point in the cloud
    closest_points_on_segment = p1 + np.outer(t_clamped, v)

    # Calculate squared distances from each point to its closest point on the segment
    distances_sq = np.sum((point_cloud - closest_points_on_segment)**2, axis=1)
    
    # Return a boolean mask: True if distance is within radius
    return distances_sq <= radius**2

def get_capsule_mask_from_pose(point_cloud, capsule_pose, half_length, radius):
    """
    Calculates a boolean mask indicating which points are inside a capsule
    defined by its pose (position + quaternion), half-length, and radius.

    Args:
        point_cloud (np.ndarray): (N, 3) array of points.
        capsule_pose (np.ndarray): 7D array (x, y, z, w, qx, qy, qz).
        half_length (float): Half the length of the cylindrical part.
        radius (float): The radius of the capsule.

    Returns:
        np.ndarray: A boolean array of shape (N,) where True indicates
                    the point is inside the capsule.
    """
    # Extract position and quaternion from the pose array
    capsule_center = np.asarray(capsule_pose[:3])
    # Ensure quaternion is (w, x, y, z)
    capsule_quat = np.asarray(capsule_pose[3:])

    # Define the local vectors for the capsule endpoints along its axis (typically Z-axis)
    local_p_plus = np.array([0.0, 0.0, half_length])
    local_p_minus = np.array([0.0, 0.0, -half_length])

    # Rotate these local vectors by the capsule's orientation quaternion
    rotated_p_plus = rotate_vector_by_quaternion(local_p_plus, capsule_quat)
    rotated_p_minus = rotate_vector_by_quaternion(local_p_minus, capsule_quat)

    # Calculate the world coordinates of the capsule's endpoints
    p1 = capsule_center + rotated_p_minus
    p2 = capsule_center + rotated_p_plus

    # Use the core logic to get the mask
    return _get_single_capsule_mask(point_cloud, p1, p2, radius)

def filter_points_with_multiple_capsules(point_cloud, capsules_data, return_inside_points=False):
    """
    Filters points from a point cloud based on multiple capsules.
    A point is considered 'inside' if it's within *any* of the provided capsules.

    Args:
        point_cloud (np.ndarray): A NumPy array of shape (N, 3) representing the
                                  point cloud.
        capsules_data (list): A list of tuples, where each tuple defines a capsule:
                              (capsule_pose, half_length, radius).
                              - capsule_pose (np.ndarray): 7D array (x, y, z, w, qx, qy, qz).
                              - half_length (float): Half the length of the cylindrical part.
                              - radius (float): The radius of the capsule.
        return_inside_points (bool): If True, returns points that are inside *any* capsule.
                                     If False (default), returns points that are NOT inside *any* capsule.

    Returns:
        np.ndarray: A NumPy array containing the filtered points.
    """
    num_points = point_cloud.shape[0]
    
    # Initialize a boolean mask. True indicates a point is inside at least one capsule.
    # We use `dtype=bool` for efficiency with boolean operations.
    overall_inside_mask = np.full(num_points, False, dtype=bool)

    # Iterate through each capsule and combine their "inside" masks
    for pose, half_length, radius in capsules_data:
        # Get the mask for points inside the current capsule
        current_capsule_mask = get_capsule_mask_from_pose(point_cloud, pose, half_length, radius)
        
        # Use logical OR to accumulate points that are inside *any* capsule.
        # If a point was already marked True (inside a previous capsule), it stays True.
        overall_inside_mask = overall_inside_mask | current_capsule_mask
    
    # Apply the final mask based on whether you want inside or outside points
    if return_inside_points:
        return point_cloud[overall_inside_mask]
    else:
        # Use logical NOT (~) to invert the mask for points outside
        return point_cloud[~overall_inside_mask]


def generate_uniform_cube_points(num_points, min_coords=(-1, -1, -1), max_coords=(1, 1, 1)):
    """
    Generates a NumPy array of points uniformly distributed within a cube.

    Args:
        num_points (int): The desired number of points.
        min_coords (tuple): A tuple (x_min, y_min, z_min) defining the
                            minimum corner of the cube. Defaults to (-1, -1, -1).
        max_coords (tuple): A tuple (x_max, y_max, z_max) defining the
                            maximum corner of the cube. Defaults to (1, 1, 1).

    Returns:
        np.ndarray: A NumPy array of shape (num_points, 3) representing
                    the point cloud.
    """
    min_coords = np.asarray(min_coords)
    max_coords = np.asarray(max_coords)

    # Calculate the range for each dimension
    ranges = max_coords - min_coords

    # Generate random numbers between 0 and 1 for each coordinate of each point
    # Shape will be (num_points, 3)
    random_points_0_1 = np.random.rand(num_points, 3)

    # Scale and shift the points to fit within the specified cube
    point_cloud = random_points_0_1 * ranges + min_coords

    return point_cloud