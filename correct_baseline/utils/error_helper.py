import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
import hdbscan
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dgl.geometry import farthest_point_sampler
import torch
import os
import glob
import h5py
import scipy
import cc3d
from sklearn.decomposition import PCA
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import label
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.ndimage as ndi
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from skimage.measure import marching_cubes
from skimage.morphology import ball
from scipy.spatial import cKDTree
#from pytorch3d.ops import sample_farthest_points
#from torchmcubes import marching_cubes, grid_interp



def sample_points_from_surface(sample: np.ndarray, n: int = 100000) -> np.ndarray:
    """
    从体素表面采样 n 个点：使用 marching cubes + farthest_point_sampler。
    如果点数不足 n，则重复补齐。

    Args:
        sample: 3D numpy array, shape [D, H, W]
        n: number of output points

    Returns:
        resample_points: (n, 3) numpy array, float32
    """
    try:
        verts, _ = marching_cubes(torch.from_numpy(sample).float(), 0.5)
    except RuntimeError as e:
        raise ValueError("Marching cubes failed. Ensure input contains valid surface data.") from e

    points = verts.unsqueeze(0).float()  # (1, N, 3)
    num_points = points.shape[1]

    if num_points == 0:
        raise ValueError("No surface points found.")

    if num_points >= n:
        # 用 farthest_point_sampler 采样 n 个点
        resample_idx = farthest_point_sampler(points, n)  # (1, n)
        resample_points = points[:, resample_idx[0], :]  # (1, n, 3)
    else:
        # 不足 n 个点，补齐
        points_np = points[0].cpu().numpy()  # (num_points, 3)
        repeat_times = n // num_points
        remainder = n % num_points

        repeated = [points_np] * repeat_times
        if remainder > 0:
            extra = points_np[np.random.choice(num_points, size=remainder, replace=True)]
            repeated.append(extra)

        padded_points = np.concatenate(repeated, axis=0).astype(np.float32)  # (n, 3)
        resample_points = torch.from_numpy(padded_points).unsqueeze(0)  # (1, n, 3)

    return resample_points[0].cpu().numpy().astype(np.float32)





def sample_points_from_surface_tensor(sample: torch.Tensor, n: int = 100000) -> torch.Tensor:
    if not isinstance(sample, torch.Tensor):
        raise TypeError("Input 'sample' must be a torch.Tensor.")
    if sample.ndim != 3:
        raise ValueError("Input 'sample' must be a 3D tensor [D, H, W].")

    try:
        verts, _ = marching_cubes(sample.float(), 0.5)
    except RuntimeError as e:
        raise ValueError("Marching cubes failed. Ensure input contains valid surface data and is on CPU/CUDA.") from e

    points = verts.unsqueeze(0).float()
    num_points = points.shape[1]

    if num_points == 0:
        print("Warning: No surface points found. Returning an empty tensor.")
        return torch.empty((0, 3), dtype=torch.float32, device=sample.device)

    if num_points >= n:
        resample_idx = farthest_point_sampler(points, n)
        resample_points = points[:, resample_idx[0], :]
    else:
        repeated_points = points.repeat(1, n // num_points, 1)

        remainder = n % num_points
        if remainder > 0:
            perm = torch.randperm(num_points, device=sample.device)[:remainder]
            extra_points = points[:, perm, :]
            resample_points = torch.cat((repeated_points, extra_points), dim=1)
        else:
            resample_points = repeated_points

    return resample_points.squeeze(0).float()











def calculate_centroid_distance(arr, id1, id2):
    positions1 = np.argwhere(arr == id1)
    positions2 = np.argwhere(arr == id2)

    centroid1 = np.mean(positions1, axis=0)
    centroid2 = np.mean(positions2, axis=0)

    distance = np.linalg.norm(centroid1 - centroid2)

    return distance

def get_candidate_centroid_distance(arr,id1):
    candidates = []
    for id in np.unique(arr)[0:]:
        distance = calculate_centroid_distance(arr,id1,id)
        if distance<=10:
            candidates.append(id)

    return candidates        


def process_all_components_with_safe_merge(seg, **kwargs):
    cc = cc3d.connected_components(seg, connectivity=6)
    out = np.zeros_like(seg, dtype=np.uint8)
    label_offset = 1
    for label in range(1, cc.max()+1):
        region = (cc == label)
        result = detect_and_correct_merge_wmz(region, **kwargs)
        if np.count_nonzero(result) == 0:
            result = region.astype(np.uint8)
        result_ids = np.unique(result)
        result_ids = result_ids[result_ids > 0]
        for rid in result_ids:
            mask = (result == rid)
            if np.sum(mask) >= 50:
                out[mask] = label_offset
                label_offset += 1
    return out


def detect_and_correct_merge_cluster_wmz_HDBSCAN(sample, boundary=False):
    dist = ndimage.distance_transform_edt(sample)
    dist_thresholded = dist.copy()
    dist_thresholded[dist < 5] = 0  
    

    points = np.stack(np.where(sample > 0), axis=1)
    if len(points) < 10:
        return sample.astype(np.int32)
    
   
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3)
    labels = clusterer.fit_predict(points)

    cc3d_out = np.zeros_like(sample, dtype=np.int32)
    for i, (x, y, z) in enumerate(points):
        if labels[i] != -1:
            cc3d_out[x, y, z] = labels[i] + 1  


    unique_labels = np.unique(cc3d_out)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) <= 1:
        return sample.astype(np.int32)

    if boundary:
        sample_point_cloud = sample_points(get_boundary(sample), n=10000, all=True)[0]
    else:
        sample_point_cloud = sample_points(sample, n=10000, all=True)[0]
    sample_point_cloud = sample_point_cloud.astype(np.int32)

    sample_dist_min_list = []
    for i in unique_labels:
        component = (cc3d_out == i)
        component_boundary = get_boundary(component)
        component_boundary_points = sample_points(component_boundary, n=10000, all=True)[0]
    
        if component_boundary_points.size == 0:
            min_distances = np.full(len(sample_point_cloud), np.inf)
        else:
            distances = scipy.spatial.distance.cdist(sample_point_cloud, component_boundary_points)
            min_distances = np.min(distances, axis=1)
        sample_dist_min_list.append(min_distances)
    dist_min = np.stack(sample_dist_min_list)
    cls = np.argmin(dist_min, axis=0)
    result = np.zeros_like(sample, dtype=np.int8)
    for i in range(cls.max() + 1):
        points_in_class = sample_point_cloud[cls == i]
        result[points_in_class[:, 0], points_in_class[:, 1], points_in_class[:, 2]] = i + 1
    label_counts = np.bincount(result.ravel())
    small_labels = np.where(label_counts < 50)[0]
    for small_label in small_labels:
        if small_label == 0:
            continue
        small_component_points = np.argwhere(result == small_label)
        if small_component_points.size == 0:
            continue
        distances = []
        for large_label in range(1, cc3d_out.max() + 1):
            if label_counts[large_label] >= 50:
                large_component_points = np.argwhere(result == large_label)
                dist = np.min(scipy.spatial.distance.cdist(small_component_points, large_component_points))
                distances.append((large_label, dist))
        if distances:
            closest_label = min(distances, key=lambda x: x[1])[0]
            result[result == small_label] = closest_label
    unique_labels = np.unique(result)
    unique_labels = unique_labels[unique_labels > 0]
    label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)}
    for old_label, new_label in label_mapping.items():
        result[result == old_label] = new_label
    
    return result


def rotate_voxel(voxel, axis):
    axis = np.array(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Axis vector cannot be zero.")
    axis = axis / norm
    rotation = R.align_vectors([axis], [[0, 0, 1]])[0]
    R_matrix = rotation.as_matrix()

    original_shape = voxel.shape
    original_center = np.array([(d - 1) / 2 for d in original_shape])
    coords = np.argwhere(voxel)
    if coords.size == 0:
        return np.zeros_like(voxel)
    rel_coords = coords - original_center
    rotated = (R_matrix @ rel_coords.T).T + original_center
    rounded = np.round(rotated).astype(int)
    
    min_coords = np.min(rounded, axis=0)
    max_coords = np.max(rounded, axis=0)
    new_shape = max_coords - min_coords + 1
    offset = min_coords
    new_voxel = np.zeros(new_shape, dtype=voxel.dtype)
    grid = np.indices(new_shape)
    grid_abs = grid + offset[:, None, None, None]
    inv_rel = (grid_abs.transpose(1,2,3,0) - original_center)
    original_coords = np.round((R_matrix.T @ inv_rel[..., None]).squeeze(-1) + original_center).astype(int)
    valid = np.all((original_coords >= 0) & (original_coords < original_shape), axis=-1)
    d, h, w = original_coords[valid].T
    new_voxel[tuple(grid[:, valid])] = voxel[d, h, w]
    new_voxel = ndi.grey_dilation(new_voxel, footprint=np.ones((3,3,3)))
    new_voxel = ndi.grey_erosion(new_voxel, footprint=np.ones((3,3,3)))
    return new_voxel




def load_segment(seg_id, seg_txt, seg_h5_fid, resample=1000, dataset='mouse'):
    # load the segment
    seg = seg_txt[seg_txt[:, 0] == seg_id, :][0]
    if 'fafb' in dataset:
        # used for localize cell in Nueroglacier
        cood = seg[1:7:2][::-1] * 2 + [2048, 0, 1]
    else:
        cood = seg[1:7:2][::-1] 
    # load the points
    sample = np.array(seg_h5_fid[seg[1]:seg[2], seg[3]:seg[4], seg[5]:seg[6]])
    
    sample[sample != seg_id] = 0


    return np.array(cood), sample


def load_segment_bbox(seg_id, seg_txt, seg_h5_fid, resample=1000, dataset='mouse'):
    # load the segment
    seg = seg_txt[seg_txt[:, 0] == seg_id, :][0]
    if 'fafb' in dataset:
        # used for localize cell in Nueroglacier
        cood = seg[1:7:2][::-1] * 2 + [2048, 0, 1]
    else:
        cood = seg[1:7:2][::-1] 
    # load the points
    sample = np.array(seg_h5_fid[seg[1]:seg[2], seg[3]:seg[4], seg[5]:seg[6]])
    
    sample[sample != seg_id] = 0

    bbox = [seg[1],seg[2], seg[3],seg[4], seg[5],seg[6]]

    return bbox, sample


# find the boundary of the sample
def get_boundary(sample):
    boundary = np.zeros_like(sample)
    boundary[1:-1,1:-1,1:-1] = (sample[1:-1,1:-1,1:-1] != sample[:-2,1:-1,1:-1]) | \
                                (sample[1:-1,1:-1,1:-1] != sample[2:,1:-1,1:-1]) | \
                                (sample[1:-1,1:-1,1:-1] != sample[1:-1,:-2,1:-1]) | \
                                (sample[1:-1,1:-1,1:-1] != sample[1:-1,2:,1:-1]) | \
                                (sample[1:-1,1:-1,1:-1] != sample[1:-1,1:-1,:-2]) | \
                                (sample[1:-1,1:-1,1:-1] != sample[1:-1,1:-1,2:])
    boundary = boundary & (sample>0)
    boundary = boundary.astype(np.uint8)
    return boundary

def sample_points(sample, n=10000000, all=True):
    points = sample.nonzero()
    points = np.stack(points, axis=1).reshape(1, -1, 3).astype(np.float32)
    num_pts = points.shape[1]
    if all:
        return points
    # 如果点数不够，重复采样
    if num_pts < n:
        reps = int(np.ceil(n / num_pts))
        points = np.tile(points, (1, reps, 1))  # 沿第2维重复
        num_pts = points.shape[1]
    # farthest point sampling
    resample_idx = farthest_point_sampler(torch.from_numpy(points), n)
    resample_points = points[:, resample_idx[0], :]
    return resample_points




#加速版本
def detect_and_correct_merge_wmz(sample, boundary=False, iteration=3, connectivity=6, erosion_shape="default"):
    # Step 1: erosion
    if erosion_shape != "default":
        structure = ball(radius=3)
        sample_erosion = ndimage.binary_erosion(sample, structure=structure, iterations=iteration)
    else:
        sample_erosion = ndimage.binary_erosion(sample, iterations=iteration)

    # Step 2: connected components
    cc3d_out = cc3d.connected_components(sample_erosion, connectivity=connectivity)
    n_labels = cc3d_out.max()
    if n_labels < 2:
        return sample.astype(np.int32)

    # Step 3: sample full mask points
    if boundary:
        sample_point_cloud = sample_points(get_boundary(sample), n=5000, all=True)[0]
    else:
        sample_point_cloud = sample_points(sample, n=5000, all=True)[0]
    sample_point_cloud = sample_point_cloud.astype(np.int32)

    # Step 4: build merged boundary point cloud & label map
    all_boundary_points = []
    component_ids = []

    for i in range(1, n_labels + 1):
        component_mask = (cc3d_out == i)
        pts = sample_points(get_boundary(component_mask), n=2000, all=True)[0]
        all_boundary_points.append(pts)
        component_ids.append(np.full(len(pts), i, dtype=np.int32))

    all_boundary_points = np.concatenate(all_boundary_points, axis=0)
    component_ids = np.concatenate(component_ids, axis=0)

    # Step 5: KDTree for matching
    tree = cKDTree(all_boundary_points)
    dists, idx = tree.query(sample_point_cloud, k=1)
    cls = component_ids[idx]

    # Step 6: fill result mask
    result = np.zeros_like(sample, dtype=np.int8)
    result[sample_point_cloud[:, 0], sample_point_cloud[:, 1], sample_point_cloud[:, 2]] = cls

    # Step 7: filter small fragments and reassign
    label_counts = np.bincount(result.ravel())
    small_labels = np.where(label_counts < 500)[0]

    # Precompute large components
    large_components = {
        label: np.argwhere(result == label)
        for label in range(1, result.max() + 1)
        if label_counts[label] >= 500
    }

    for small_label in small_labels:
        if small_label == 0:
            continue
        small_pts = np.argwhere(result == small_label)
        if small_pts.size == 0:
            continue
        min_dist = float('inf')
        closest_label = None
        for large_label, large_pts in large_components.items():
            dist = np.min(scipy.spatial.distance.cdist(small_pts, large_pts))
            if dist < min_dist:
                min_dist = dist
                closest_label = large_label
        if closest_label is not None:
            result[result == small_label] = closest_label

    # Step 8: relabel to [1, 2, 3, ...]
    unique_labels = np.unique(result)
    unique_labels = unique_labels[unique_labels > 0]
    label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)}

    # Apply mapping vectorized
    max_label = result.max()
    mapping_array = np.arange(max_label + 1)
    for old_label, new_label in label_mapping.items():
        mapping_array[old_label] = new_label
    result = mapping_array[result]

    return result.astype(np.int32)


""" def detect_and_correct_merge_wmz(sample, boundary=False, iteration=3, connectivity=6,erosion_shape="default"):
    # 3D erosion
    sample_erosion = ndimage.binary_erosion(sample, iterations=iteration)
    if erosion_shape != "default":
        structure = ball(radius=3)
        sample_erosion = ndimage.binary_erosion(sample, structure = structure,iterations=iteration)
    # Connected components analysis
    cc3d_out = cc3d.connected_components(sample_erosion, connectivity=connectivity)
    
    if len(np.unique(cc3d_out)) <= 2:  # If only one connected component or empty
        return sample.astype(np.int32)  # Return the input as a single-class array

    # Find nearest points to separated components in the original sample
    if boundary:
        sample_point_cloud = sample_points(get_boundary(sample), n=10000, all=True)[0]
    else:
        sample_point_cloud = sample_points(sample, n=10000, all=True)[0]
    sample_point_cloud = sample_point_cloud.astype(np.int32)

    sample_dist_min_list = []

    for i in range(1, cc3d_out.max() + 1):
        component_boundary_points = sample_points(get_boundary(cc3d_out == i), n=10000, all=True)[0]
        distances = scipy.spatial.distance.cdist(sample_point_cloud, component_boundary_points)
        sample_dist_min_list.append(np.min(distances, axis=1))

    if len(sample_dist_min_list) == 0:
        return sample.astype(np.int8)

    # Compute the class for each point based on the minimum distance
    dist_min = np.stack(sample_dist_min_list)
    cls = np.argmin(dist_min, axis=0)

    # Create a result array to store the segmented regions
    result = np.zeros_like(sample, dtype=np.int8)

    # Assign unique IDs to each class in the result array
    for i in range(cls.max() + 1):
        points_in_class = sample_point_cloud[cls == i]
        result[points_in_class[:, 0], points_in_class[:, 1], points_in_class[:, 2]] = i + 1  # Assign cls id (1-based)

    # After relabeling, we need to check for components with fewer than 10 voxels
    label_counts = np.bincount(result.ravel())  # Count how many voxels belong to each label
    small_labels = np.where(label_counts < 500)[0]  # Find labels with fewer than 10 voxels

    # For each small label, find the nearest component with more than 10 voxels
    for small_label in small_labels:
        if small_label == 0:  # Skip background label (0)
            continue
        # Find the boundary points for the small component
        small_component_points = np.argwhere(result == small_label)
        if small_component_points.size == 0:
            continue
        
        # For each small component point, find the closest component with more than 10 voxels
        distances = []
        for large_label in range(1, cc3d_out.max() + 1):
            if label_counts[large_label] >= 500:  # Only consider larger components
                large_component_points = np.argwhere(result == large_label)
                dist = np.min(scipy.spatial.distance.cdist(small_component_points, large_component_points))
                distances.append((large_label, dist))
        
        # If there are valid distances, relabel the small component to the closest larger component
        if distances:
            closest_label = min(distances, key=lambda x: x[1])[0]
            result[result == small_label] = closest_label

    # Reassign labels sequentially from 1 to n
    unique_labels = np.unique(result)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background label (0)
    
    # Create a mapping from original labels to new labels
    label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)}

    # Apply the label mapping to the result array
    for old_label, new_label in label_mapping.items():
        result[result == old_label] = new_label
    return result """



def detect_and_correct_merge_wmz1(sample, boundary=False, iteration=3, connectivity=6):
    # 3D erosion
    sample_erosion = ndimage.binary_erosion(sample, iterations=iteration)
    
    # Connected components analysis
    cc3d_out = cc3d.connected_components(sample_erosion, connectivity=connectivity)
    
    if len(np.unique(cc3d_out)) <= 2:  # If only one connected component or empty
        return sample.astype(np.int32)  # Return the input as a single-class array

    # Find nearest points to separated components in the original sample
    if boundary:
        sample_point_cloud = sample_points(get_boundary(sample), n=10000, all=True)[0]
    else:
        sample_point_cloud = sample_points(sample, n=10000, all=True)[0]
    sample_point_cloud = sample_point_cloud.astype(np.int32)

    sample_dist_min_list = []

    for i in range(1, cc3d_out.max() + 1):
        component_boundary_points = sample_points(get_boundary(cc3d_out == i), n=10000, all=True)[0]
        distances = scipy.spatial.distance.cdist(sample_point_cloud, component_boundary_points)
        sample_dist_min_list.append(np.min(distances, axis=1))

    if len(sample_dist_min_list) == 0:
        return sample.astype(np.int8)

    # Compute the class for each point based on the minimum distance
    dist_min = np.stack(sample_dist_min_list)
    cls = np.argmin(dist_min, axis=0)

    # Create a result array to store the segmented regions
    result = np.zeros_like(sample, dtype=np.int8)

    # Assign unique IDs to each class in the result array
    for i in range(cls.max() + 1):
        points_in_class = sample_point_cloud[cls == i]
        result[points_in_class[:, 0], points_in_class[:, 1], points_in_class[:, 2]] = i + 1  # Assign cls id (1-based)

    return result



def detect_and_correct_merge(sample, boundary=False, iteration=3, connectivity=6):
    # 3d erosion
    sample_erosion = ndimage.binary_erosion(sample, iterations=iteration)
    # cc3d 
    cc3d_out = cc3d.connected_components(sample_erosion, connectivity=connectivity)
    if len(cc3d_out) < 1:
        return [sample]
    if cc3d_out.max() == 1:
        return [sample]

    # for the original sample, find the nearest point to the two separated components
    if boundary:
        sample_point_cloud = sample_points(get_boundary(sample), n=10000, all=True)[0]
    else:
        sample_point_cloud = sample_points(sample, n=10000, all=True)[0]
    sample_point_cloud = sample_point_cloud.astype(np.int32)

    sample_dist_min_list = []
    # print('cc max', cc3d_out.max())
    for i in range(1, cc3d_out.max() + 1):
        sample1_point_cloud = sample_points(get_boundary(cc3d_out == i), n=10000, all=True)[0]
        sample1_dist = scipy.spatial.distance.cdist(sample_point_cloud, sample1_point_cloud)
        sample1_dist_min = np.min(sample1_dist, axis=1)
        sample_dist_min_list.append(sample1_dist_min)
    # sample2_point_cloud = sample_points(get_boundary(cc3d_out == 2), n=10000, all=True)[0]
    # sample2_dist = scipy.spatial.distance.cdist(sample_point_cloud, sample2_point_cloud)
    # sample2_dist_min = np.min(sample2_dist, axis=1)
    if len(sample_dist_min_list) == 0:
        return [sample]
    dist_min = np.stack(sample_dist_min_list)
    cls = np.argmin(dist_min, axis=0)

    # separate sample according to the cls, cls can be more than 2 classes
    max_cls = cls.max()
    sample_cls = []

    # print('max cls', max_cls)
    # print('len dis min', len(dist_min))
    for i in range(max_cls + 1):
        sample1 = np.zeros_like(sample)
        sample1[sample_point_cloud[cls == i, 0], sample_point_cloud[cls == i, 1], sample_point_cloud[cls == i, 2]] = 1
        sample_cls.append(sample1)

    return sample_cls

#################
# find the part inside the ellipsoid
# given a point, check if it is inside the ellipsoid
def inside_ellipsoid_point(point, center, radii, evecs):
    # transform the point to the ellipsoid coordinate
    point = point - center
    point = np.dot(point, evecs)
    point = point / radii
    if np.sum(point ** 2) <= 1:
        return True
    else:
        return False
    
def inside_ellipsoid(sample, id, center, radii, evecs):
    # transform the point to the ellipsoid coordinate
    
    tmp_points = (sample == id).nonzero()
    tmp_points = np.array(tmp_points).T
    n = 0
    for p in tmp_points:
        if inside_ellipsoid_point(p, center, radii, evecs):
            n += 1
    if n / tmp_points.shape[0] > 0.5:
        return True
    else:
        return False
    


def detect_and_correct_split_ellipsoid_fit(id, sample):
    # Dilate only the region of interest with specific ID and keep its value
    sample = sample.astype(np.int_)
    dilated_sample = np.copy(sample)
    mask = (sample == id)
    dilated_mask = ndimage.binary_dilation(mask, iterations=1)
    dilated_sample[dilated_mask] = id
    dilated_sample[dilated_sample!=id] = 0
    smp_size = sample.shape
    resample_points = sample_points(get_boundary(dilated_sample), n=10000, all=True)[0]
    """ if min(pdlated_sample.shape) < 11:
        return [id] """
    
    if len(resample_points) < 100:  # Ensure enough points for fitting
        return [id]
    
    center, evecs, radii, v, mae = ellipsoid_fit(resample_points)
    if mae > 0.5:
        return [id]

    corner1 = center - radii
    corner2 = center + radii

    # Adjust cropping region to be within bounds of the entire sample
    min_region = np.maximum(np.floor(corner1).astype(np.int32), [0, 0, 0])
    max_region = np.minimum(np.ceil(corner2).astype(np.int32), smp_size)

    extend_sample = sample[min_region[0]:max_region[0], min_region[1]:max_region[1], min_region[2]:max_region[2]]
    # Relocate the center based on the cropped region
    center = center - np.array([min(corner1[i], 0) for i in range(3)])

    ids = np.unique(extend_sample)
    ids = ids[ids > 0]

    output_samples = []
    output_ids = []
    for i in ids:
        if inside_ellipsoid(extend_sample, i, center, radii, evecs):
            tmp = np.zeros_like(extend_sample)
            tmp[extend_sample == i] = i
            output_samples.append(tmp)
            output_ids.append(i)
    return output_ids



# define a function to detect and correct the split error
def detect_and_correct_split(id, sample, seg_txt, fid):
    # dilate the sample
    sample = ndimage.binary_dilation(sample, iterations=3)
    smp_size = sample.shape
    if min(smp_size) < 11:
        return [sample], [id]
    resample_points = sample_points(get_boundary(sample), n=10000, all=True)[0]
    # remove the points at the boundary
    center, evecs, radii, v, mae = ellipsoid_fit(resample_points)
    if mae > 0.5:
        return [sample], [id]
    seg = seg_txt[seg_txt[:, 0] == id, :][0]
    cood = seg[1:7:2]
    # according to the fitted ellipsoid, crop the region 
    corner1 = center - radii
    corner2 = center + radii
    size = sample.shape


    # decide the min_region and max_region
    min_region = np.array([min(corner1[i], 0) for i in range(3)]) + cood 
    min_region = np.floor(min_region).astype(np.int32)

    max_region = np.array([max(corner2[i], size[i]) for i in range(3)]) + cood
    max_region = np.ceil(max_region).astype(np.int32)
    
    # crop the region
    # if the size is too large, don't crop
    if (max_region[0] - min_region[0]) * (max_region[1] - min_region[1]) * (max_region[2] - min_region[2]) > 1000**3:
        return [sample], [id]
    extend_sample = np.array(fid[min_region[0]:max_region[0], min_region[1]:max_region[1], min_region[2]:max_region[2]])

    # relocate the center
    center = center - np.array([min(corner1[i], 0) for i in range(3)])

    ids = np.unique(extend_sample)
    ids = ids[ids > 0]

    output_samples = []
    output_ids = []
    for i in ids:
        if inside_ellipsoid(extend_sample, i, center, radii, evecs):
            tmp = np.zeros_like(extend_sample)
            tmp[extend_sample == i] = i
            output_samples.append(tmp)
            output_ids.append(i)
    return output_samples, output_ids

##################


def ellipsoid_fit(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                    x * x + z * z - 2 * y * y,
                    2 * x * y,
                    2 * x * z,
                    2 * y * z,
                    2 * x,
                    2 * y,
                    2 * z,
                    1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                    [v[3], v[1], v[5], v[7]],
                    [v[4], v[5], v[2], v[8]],
                    [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    sgns = np.sign(evals).reshape(-1, 3)
    d = X - center.reshape(-1, 3) # Shift data to origin
    d = np.dot(d, evecs.T)  # Rotate to cardinal axes of the conic
    d = d / radii.reshape(-1, 3)#  np.array([d[:, 0] / radii[0], d[:, 1] / radii[1], d[:, 2] / radii[2]])  # Normalize to the conic radii
    mae = np.mean(np.abs(1 - np.sum((d ** 2) * sgns, axis=1)))  # Calculate mae

    return center, evecs, radii, v, mae

def get_bb(seg, do_count=False):
    # get bounding box and count of a segment
    dim = len(seg.shape)
    a=np.where(seg>0)
    if len(a[0])==0:
        return [-1]*dim*2
    out=[]
    for i in range(dim):
        out+=[a[i].min(), a[i].max()]
    if do_count:
        out+=[len(a[0])]
    return out

def splitSeg(seg, sids, th_iou = 0.7, min_sz=1000):
    mid = seg.max()
    sz = seg.shape
    for ii,sid in enumerate(sids):
        print('track %d/%d (%d)'%(ii,len(sids),mid))
        mid0 = mid
        bb = get_bb(seg==sid)
        if bb[0] > -1:
            pre_id = []
            pre_num = []
            # for each z slice in the segment
            for z in range(bb[0],bb[1]+1):
                # get 2d bbx
                bb2 = get_bb(seg[z]==sid)
                if bb2[0] > -1: # if miss slice, copy previous slice
                    seg_z = seg[z,bb2[0]:bb2[1]+1,bb2[2]:bb2[3]+1]
                    ss = ndimage.label(seg_z==sid)[0] 
                    uid,uc = np.unique(ss[ss>0], return_counts=True)
                    sm = ss.max()
                    if len(pre_id)==0:# initial
                        for i in range(1,sm+1):
                            seg_z[ss==i] = mid+i
                        pre_id = range(mid+1,mid+sm+1)
                        mid += sm
                    elif sm==1 and len(pre_id)==1: # assign to the same one
                        seg_z[ss==1] = pre_id[0]
                        # no change in pre_id
                    else: # check IoU
                        seg_z0 =seg[z-1,bb2[0]:bb2[1]+1,bb2[2]:bb2[3]+1]
                        pre_id2 = list(uid)
                        for i in range(1,sm+1):
                            uid2,uc2 = np.unique(seg_z0[ss==i],return_counts=True)
                            uc2[uid2==0] = 0
                            cm = float(uc2.max())
                            cmid = uid2[np.argmax(uc2)]
                            if (cmid in pre_id) and max(cm/pre_num[pre_id==cmid])>th_iou and cm/uc[i-1]>th_iou:
                                seg_z[ss==i] = cmid
                                pre_id2[i-1] = cmid
                            else: # start anew
                                seg_z[ss==i] = mid+1
                                pre_id2[i-1] = mid+1
                                mid += 1
                        pre_id = list(pre_id2)
                    pre_num = np.array(uc)
            # remove small seg
            ui,uc = np.unique(seg[seg>mid0], return_counts=True)
            bid = ui[uc<min_sz]
            if len(bid)>0:
                rl = np.arange(mid+1).astype(seg.dtype)
                gid = ui[uc>=min_sz]
                rl[bid] = 0
                rl[gid] = mid0+np.arange(1,len(gid)+1)
                seg = rl[seg]
                mid = seg.max()
    return seg            

# define a function plot a learned ellipsoid
def ellipsoid_plot(center, radii, rotation, ax, plot_axes=False, cage_color='b', cage_alpha=0.2):
    """Plot an ellipsoid"""
        
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    
    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    if plot_axes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                         [0.0,radii[1],0.0],
                         [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cage_color)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cage_color, alpha=cage_alpha)

def isotropic_rescale(points, scale=[1,1,80/64]):
    points = np.array(points).astype(np.float32)
    scale = np.array(scale).astype(np.float32)
    points *= scale
    return points


# define a function to compute the number of connected components
def get_num_cc(seg):
    min_size = min(seg.shape)
    if min_size <= 5:
        return 1
    # first, apply distance transform
    dist = ndimage.distance_transform_edt(seg)
    dist[dist < 5] = 0

    # run DBSCAN
    points = np.stack(np.where(dist>0), axis=1)
    if len(points) < 10:
        return 1
    clustering = DBSCAN(eps=3, min_samples=10).fit(points)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters_

# define a function to compute the max area of 6 faces
def get_max_area(sample):
    # get the max area of 6 faces
    z, y, x = sample.shape
    max_area = 0
    face_list = []
    cood_list = []
    # max_cood = None
    face0 = sample[0:1,:,:]
    # center of mass
    cood0 = np.stack(np.where(face0>0), axis=1)
    if len(cood0) > 0:
        cood0 = cood0.mean(axis=0)
        face_list.append(face0)
        cood_list.append(cood0)
    
    face1 = sample[-1:,:,:]
    cood1 = np.stack(np.where(face1>0), axis=1)
    if len(cood1) > 0:
        cood1 = cood1.mean(axis=0)
        cood1[0] = z - 1
        face_list.append(face1)
        cood_list.append(cood1)

    face2 = sample[:,0:1,:]
    cood2 = np.stack(np.where(face2>0), axis=1)
    if len(cood2) > 0:
        cood2 = cood2.mean(axis=0)
        face_list.append(face2)
        cood_list.append(cood2)
    
    face3 = sample[:,-1:,:]
    cood3 = np.stack(np.where(face3>0), axis=1)
    if len(cood3) > 0:
        cood3 = cood3.mean(axis=0)
        cood3[1] = y - 1
        face_list.append(face3)
        cood_list.append(cood3)

    face4 = sample[:,:,0:1]
    cood4 = np.stack(np.where(face4>0), axis=1)
    if len(cood4) > 0:
        cood4 = cood4.mean(axis=0)
        face_list.append(face4)
        cood_list.append(cood4)

    face5 = sample[:,:,-1:]
    cood5 = np.stack(np.where(face5>0), axis=1)
    if len(cood5) > 0:
        cood5 = cood5.mean(axis=0)
        cood5[2] = x - 1
        face_list.append(face5)
        cood_list.append(cood5)
    # face_list = [face0, face1, face2, face3, face4, face5]
    # cood_list = [cood0, cood1, cood2, cood3, cood4, cood5]

    for face, cood in zip(face_list, cood_list):
        face_area = (face>0).sum()
        if face_area > max_area:
            max_area = face_area
            max_cood = cood
    return max_area, max_cood

def split_detect(sample):
    m = 1414
    std = 395
    max_area, _ = get_max_area(sample)
    if max_area > m - 2 * std:
        return True
    else:
        return False

# random 3d coordinates
def random_3d_coordinates(sample, num_points):
    z, y, x = sample.shape
    z = np.random.randint(0, z, num_points)
    y = np.random.randint(0, y, num_points)
    x = np.random.randint(0, x, num_points)
    return np.stack([z,y,x], axis=1)

# padding the sample so that the length of the sample is the same as the longest sample
def pad_sample(sample):
    max_len = max(sample.shape)
    d, h, w = sample.shape
    pad1 = ((max_len - d) // 2, max_len - d - (max_len - d) // 2)
    pad2 = ((max_len - h) // 2, max_len - h - (max_len - h) // 2)
    pad3 = ((max_len - w) // 2, max_len - w - (max_len - w) // 2)
    return np.pad(sample, (pad1, pad2, pad3), 'constant')

# visualization
def show_points_from_sample(sample):
    # sample is voxels
    # get boundary of sample
    boundary = get_boundary(sample)
    # resample boundary wirh 2048 points
    points = np.where(boundary > 0)
    points = np.stack(points, axis=1).reshape(1, -1, 3).astype(np.float32)
    num = max(2048, len(points[0]))
    resample_idx = farthest_point_sampler(torch.from_numpy(points), num)
    resample_points = points[0, resample_idx[0], :]
    # show points
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(resample_points[:, 0], resample_points[:, 1], resample_points[:, 2], c='r', s=1)
    plt.show()


def detect_and_correct_merge_cluster_wmz(sample, boundary=False):
    # 应用距离变换并阈值处理以获取内部区域
    dist = ndimage.distance_transform_edt(sample)
    dist_thresholded = dist.copy()
    dist_thresholded[dist < 5] = 0  # 保留距离较大的中心区域
    
    # 提取非零点坐标作为点云
    points = np.stack(np.where(sample > 0), axis=1)
    # points = np.stack(np.where(sample>0))
    if len(points) < 10:
        return sample.astype(np.int32)
    
    # 使用DBSCAN聚类点云
    clustering = DBSCAN(eps=3, min_samples=10).fit(points)
    labels = clustering.labels_

    # 将聚类结果映射到3D数组中
    cc3d_out = np.zeros_like(sample, dtype=np.int32)
    for i, (x, y, z) in enumerate(points):
        if labels[i] != -1:
            cc3d_out[x, y, z] = labels[i] + 1  # 有效聚类标签从1开始

    # 获取实际存在的聚类标签（排除背景0和噪声-1）
    unique_labels = np.unique(cc3d_out)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) <= 1:
        return sample.astype(np.int32)

    # 生成样本点云（原始或边界）
    if boundary:
        sample_point_cloud = sample_points(get_boundary(sample), n=10000, all=True)[0]
    else:
        sample_point_cloud = sample_points(sample, n=10000, all=True)[0]
    sample_point_cloud = sample_point_cloud.astype(np.int32)

    # 计算各聚类区域的边界点到样本点的距离（处理空边界）
    sample_dist_min_list = []
    for i in unique_labels:
        component = (cc3d_out == i)
        component_boundary = get_boundary(component)
        component_boundary_points = sample_points(component_boundary, n=10000, all=True)[0]
    
        if component_boundary_points.size == 0:
            min_distances = np.full(len(sample_point_cloud), np.inf)
        else:
            distances = scipy.spatial.distance.cdist(sample_point_cloud, component_boundary_points)
            min_distances = np.min(distances, axis=1)
        sample_dist_min_list.append(min_distances)
    
    # 分配样本点到最近的聚类
    dist_min = np.stack(sample_dist_min_list)
    cls = np.argmin(dist_min, axis=0)
    
    # 创建结果数组并分配标签
    result = np.zeros_like(sample, dtype=np.int8)
    for i in range(cls.max() + 1):
        points_in_class = sample_point_cloud[cls == i]
        result[points_in_class[:, 0], points_in_class[:, 1], points_in_class[:, 2]] = i + 1
    
    # 合并小区域到最近的较大区域
    label_counts = np.bincount(result.ravel())
    small_labels = np.where(label_counts < 50)[0]
    
    for small_label in small_labels:
        if small_label == 0:
            continue
        small_component_points = np.argwhere(result == small_label)
        if small_component_points.size == 0:
            continue
        
        distances = []
        for large_label in range(1, cc3d_out.max() + 1):
            if label_counts[large_label] >= 50:
                large_component_points = np.argwhere(result == large_label)
                dist = np.min(scipy.spatial.distance.cdist(small_component_points, large_component_points))
                distances.append((large_label, dist))
        
        if distances:
            closest_label = min(distances, key=lambda x: x[1])[0]
            result[result == small_label] = closest_label
    
    # 重新编号标签
    unique_labels = np.unique(result)
    unique_labels = unique_labels[unique_labels > 0]
    label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)}
    for old_label, new_label in label_mapping.items():
        result[result == old_label] = new_label
    
    return result



# define a function to random generate split error samples
def random_split_error(sample, margin=5):
    # add margin to sample
    d, h, w = sample.shape
    # margin = 5
    sample = np.pad(sample, ((margin, margin), (margin, margin), (margin, margin)), 'constant', constant_values=0)
    # sample 2048 points and label
    coods = random_3d_coordinates(sample, 2048)
    labels = sample[coods[:,0], coods[:,1], coods[:,2]]  

    d, h, w = sample.shape
    direction = np.random.choice([0, 1, 2]) # 0: left, 1: right, 2: both
    error_rate1 = np.random.uniform(0.1, 0.4)
    error_rate2 = np.random.uniform(0.1, 0.4)
    if direction < 2:
        error_rate1 = error_rate1 + error_rate2
    axis = np.random.choice([0, 1, 2])
    if axis == 0:
        if direction == 0 or direction == 2:
            sample[0:int(error_rate1 * d), :, :] = 0
        if direction == 1 or direction == 2:
            sample[int((1 - error_rate2) * d):d, :, :] = 0
    elif axis == 1:
        if direction == 0 or direction == 2:
            sample[:, 0:int(error_rate1 * h), :] = 0
        if direction == 1 or direction == 2:
            sample[:, int((1 - error_rate2) * h):h, :] = 0
    else:
        if direction == 0 or direction == 2:
            sample[:, :, 0:int(error_rate1 * w)] = 0
        if direction == 1 or direction == 2:
            sample[:, :, int((1 - error_rate2) * w):w] = 0
    # remove the zero slices
    # sample = sample[~np.all(sample == 0, axis=(1, 2))]
    # sample = sample[:, ~np.all(sample == 0, axis=(0, 2))]
    # sample = sample[:, :, ~np.all(sample == 0, axis=(0, 1))]
    # get boundary
    boundary = get_boundary(sample)
    # get surface
    points = np.where(boundary > 0)
    points = np.stack(points, axis=1).reshape(1, -1, 3).astype(np.float32)
    if len(points[0]) < 2048:
        return None, None, None, None
    resample_idx = farthest_point_sampler(torch.from_numpy(points), 2048)
    resample_points = points[0, resample_idx[0], :]
    return resample_points, coods, labels, sample

# define a function to random generate merge error samples
def random_merge_error(sample1, sample2):
    # make sure sample1 and sample2 are overlapped
    # compute the overlapped region
    d1, h1, w1 = sample1.shape
    d2, h2, w2 = sample2.shape
    enlarge_sample1 = np.zeros((max(d1, d2), max(h1, h2), max(w1, w2)))
    enlarge_sample1[0:d1, 0:h1, 0:w1] = sample1
    enlarge_sample2 = np.zeros((max(d1, d2), max(h1, h2), max(w1, w2)))
    enlarge_sample2[0:d2, 0:h2, 0:w2] = sample2

    overlap = np.logical_and(enlarge_sample1, enlarge_sample2)
    # if np.sum(overlap) < 10:
    #     enlarge_sample1[enlarge_sample1 > 0] = 1
    #     enlarge_sample2[enlarge_sample2 > 0] = 2
    #     sample = enlarge_sample1 + enlarge_sample2
    #     return sample
        
    # move sample2 to a random direction
    d1, h1, w1 = sample1.shape
    d2, h2, w2 = sample2.shape
    direction = np.random.rand(3)

    # move sample2 using direction, until the overlap is less than 10 
    # while np.sum(overlap) > 10:
    points = np.where(sample2 > 0)
    points = np.array(points).T
    
    # move sample2
    length = 1
    while(overlap.sum() > 10):
        
        points = points + direction * length
        points = points.astype(int)
        min_points = np.min(points, axis=0)
        if min_points[0] < 0:
            points[:, 0] = points[:, 0] - min_points[0]
        if min_points[1] < 0:
            points[:, 1] = points[:, 1] - min_points[1] 
        if min_points[2] < 0:
            points[:, 2] = points[:, 2] - min_points[2]
        
        max_points = np.max(points, axis=0)
        max_d, max_h, max_w = max_points[0], max_points[1], max_points[2]
        enlarge_sample2 = np.zeros((max(d1, max_d + 1), max(h1, max_h + 1), max(w1, max_w + 1)))
        enlarge_sample2[points[:, 0], points[:, 1], points[:, 2]] = 1

        enlarge_sample1 = np.zeros((max(d1, max_d + 1), max(h1, max_h + 1), max(w1, max_w + 1)))
        enlarge_sample1[0:d1, 0:h1, 0:w1] = sample1

        overlap = np.logical_and(enlarge_sample1, enlarge_sample2)
        length += 1

    enlarge_sample1[enlarge_sample1 > 0] = 1
    enlarge_sample2[enlarge_sample2 > 0] = 2
    sample = enlarge_sample1 + enlarge_sample2

    # get random sample and labels
    coods = random_3d_coordinates(sample, 2048)
    labels = sample[coods[:,0], coods[:,1], coods[:,2]] 

    # get boundary of sample
    sample = sample.astype(np.uint8)
    boundary = get_boundary(sample)
 
    # resample boundary wirh 2048 points
    points = np.where(boundary > 0)
    points = np.stack(points, axis=1).reshape(1, -1, 3).astype(np.float32)
    if len(points[0]) < 2048:
        return None, None, None, None
    resample_idx = farthest_point_sampler(torch.from_numpy(points), 2048)
    resample_points = points[0, resample_idx[0], :]

    return resample_points, coods, labels, sample

def normal_sample(sample, margin=5):
    # add margin
    sample = np.pad(sample, ((margin, margin), (margin, margin), (margin, margin)), 'constant', constant_values=0)
    # get random 3d points and labels
    coods = random_3d_coordinates(sample, 2048)
    labels = sample[coods[:, 0], coods[:, 1], coods[:, 2]]
    # get boundary of sample
    boundary = get_boundary(sample)
    points = np.where(boundary > 0)
    points = np.stack(points, axis=1).reshape(1, -1, 3).astype(np.float32)
    if len(points[0]) < 2048:
        return None, None, None, None
    resample_idx = farthest_point_sampler(torch.from_numpy(points), 2048)
    resample_points = points[0, resample_idx[0], :]
    return resample_points, coods, labels, sample


# generate a view with normalized depth as intensity 
def generate_view(sample, axis=0, direction=1):
    # axis: 0, 1, 2
    # direction: 1, -1
    # find index of the first non-zero value
    if direction == 1:
        idx = np.argmax(sample > 0, axis=axis)
    else:
        if axis == 0:
            idx = np.argmax(sample[::-1, :, :] > 0, axis=axis)
        elif axis == 1:
            idx = np.argmax(sample[:, ::-1, :] > 0, axis=axis)
        elif axis == 2:
            idx = np.argmax(sample[:, :, ::-1] > 0, axis=axis)
    # idx = (idx) / idx.max()
    return idx

def generate_6views(sample):
    views = []
    for axis in [0, 1, 2]:
        for direction in [0, 1]:
            view = generate_view(sample, axis, direction)
            views.append(view)
    return np.array(views)



def show_6views(sample, colorbar=False):
    if len(sample) == 6:
        views = sample
    fig = plt.figure(figsize=(12, 8))
    for axis in [0, 1, 2]:
        for direction in [0, 1]:
            if len(sample) == 6:
                view = views[axis*2+direction]
            else:
                view = generate_view(sample, axis, direction)
            plt.subplot(2, 3, 3 * direction + axis + 1)
            plt.imshow(view, cmap='jet')
            if colorbar:
                plt.colorbar()
            plt.axis('off')
    plt.show()


def save_6views_to_numpy(sample, colorbar=False):
    
    if len(sample) == 6:
        views = sample

    fig = plt.figure(figsize=(3, 2))  
    
    for axis in [0, 1, 2]:  
        for direction in [0, 1]: 
            if len(sample) == 6:
                view = views[axis * 2 + direction]
            else:
                view = generate_view(sample, axis, direction)
            
            plt.subplot(2, 3, 3 * direction + axis + 1)
            plt.imshow(view, cmap='jet')
            
            if colorbar:
                plt.colorbar()  
            
            plt.axis('off')  
    
    plt.tight_layout(pad=0.5)  

    fig.canvas.draw()  
    image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  
    image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))  

    plt.close()

    return image_data



def save_6views(sample, colorbar=False, save_path='sample.png'):
    if len(sample) == 6:
        views = sample
    
    # 减小 figsize
    fig = plt.figure(figsize=(3, 2))  # 调整为更紧凑的尺寸
    
    for axis in [0, 1, 2]:  # 遍历三个轴
        for direction in [0, 1]:  # 遍历两个方向
            if len(sample) == 6:
                view = views[axis * 2 + direction]
            else:
                view = generate_view(sample, axis, direction)
            
            # 绘制子图
            plt.subplot(2, 3, 3 * direction + axis + 1)
            plt.imshow(view, cmap='jet')
            
            if colorbar:
                plt.colorbar()  # 添加颜色条
            
            plt.axis('off')  # 关闭坐标轴
    
    # 使用 tight_layout 减少空白区域
    plt.tight_layout(pad=0.5)  # pad 控制边缘留白大小
    
    # 保存图像并关闭
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  # 进一步减少空白区域
    plt.close()

def reconstruct_surface(views):
    # reconstruct surface from 6 views
    # views: 6 views, each view is a 2d array
    # return: a 3d array
    
    h, w = views[0].shape
    cood_2d = np.mgrid[0:h, 0:w].reshape(2, -1).T
    cood_3d = None
    for axis in [0, 1, 2]:
        for direction in [0, 1]:
            z_dim = views[axis*2+direction][cood_2d[:, 0], cood_2d[:, 1]]
            if axis == 0:
                cood_3d_2 = np.concatenate([z_dim[:, None], cood_2d], axis=1)
            elif axis == 1:
                cood_3d_2 = np.concatenate([cood_2d[:, 0][:, None], z_dim[:, None], cood_2d[:, 1][:, None]], axis=1)
            elif axis == 2:
                cood_3d_2 = np.concatenate([cood_2d, z_dim[:, None]], axis=1)
            cood_3d_2 = cood_3d_2[cood_3d_2[:, axis] > 0]
            if direction == 0:
                cood_3d_2[:,axis] = h - cood_3d_2[:,axis]
            if cood_3d is None:
                cood_3d = cood_3d_2
            else:
                cood_3d = np.concatenate([cood_3d, cood_3d_2], axis=0)
    return cood_3d

class Nucleus_2d_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='../../data/Nucleus_2d_syn', train=True):
        self.id_lists = glob.glob(os.path.join(data_dir, '*.npy'))
        if train:
            self.id_lists = self.id_lists[:int(len(self.id_lists) * 0.8)]
        else:
            self.id_lists = self.id_lists[int(len(self.id_lists) * 0.8):]
        
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            # resize
            transforms.Resize((64, 64), antialias=True),
        ])

    def __getitem__(self, index):
        # load segmentation from h5
        data_dict = np.load(self.id_lists[index], allow_pickle=True).item()
        input = torch.from_numpy(np.array(data_dict['input'], dtype=np.float32))
        output = torch.from_numpy(np.array(data_dict['output'], dtype=np.float32))
        return self.transform(input), self.transform(output)
    
    def __len__(self):
        return len(self.id_lists)


def get_nonzero_coords(data):
    # 获取所有非零元素的坐标
    return np.argwhere(data != 0)

def align_object_to_z_axis(data):
    # 获取非零点的坐标
    coords = get_nonzero_coords(data)
    
    # 主成分分析
    pca = PCA(n_components=3)
    pca.fit(coords)
    
    # 获取旋转矩阵
    rotation_matrix = pca.components_
    
    # 旋转物体，使最长轴与z轴对齐
    centered_coords = coords - pca.mean_
    rotated_coords = np.dot(centered_coords, rotation_matrix)
    
    # 找到新的边界框
    min_corner = np.min(rotated_coords, axis=0).astype(int)
    max_corner = np.max(rotated_coords, axis=0).astype(int)
    
    # 创建新的空数组，大小基于新的边界框
    new_shape = max_corner - min_corner + 1
    new_data = np.zeros(new_shape, dtype=data.dtype)
    
    # 将旋转后的坐标映射回新数组
    for i, coord in enumerate(rotated_coords):
        new_coord = (coord - min_corner).astype(int)
        new_data[tuple(new_coord)] = data[tuple(coords[i])]
    
    return new_data, rotation_matrix, pca.mean_


def transform_coords_back(rotated_coords, rotation_matrix, mean_translation):
    # 反转旋转
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)
    original_centered_coords = np.dot(rotated_coords, inv_rotation_matrix)
    
    # 添加平均值偏移量以恢复原始位置
    original_coords = original_centered_coords + mean_translation
    
    return original_coords





# define a dataset to load the data
class NucleusDataset(torch.utils.data.Dataset):
    def __init__(self, id_lists=None, dataset='fafb'):
        if 'fafb' in dataset:
            seg_h5_path = '/mmfs1/data/bccv/dataset/FAFB/nucleus_seg_64nm.h5'
            seg_txt_path = '/mmfs1/data/bccv/dataset/FAFB/nucleus_seg_64nm_bb.txt'
        else:
            seg_h5_path = '/mmfs1/data/bccv/dataset/foundation/low_res/mouse_microns-phase2_256-320nm_nuclei_m65.h5'
            seg_txt_path = '/mmfs1/data/bccv/dataset/foundation/low_res/mouse_microns-phase2_256-320nm_nuclei_m65_bb.txt'

        # load segmentation from txt
        self.seg_txt = np.loadtxt(seg_txt_path, dtype=np.int32)
        self.fid = h5py.File(seg_h5_path, 'r')['main']
        self.id_lists = id_lists if id_lists is not None else self.seg_txt[:,0]

    def __getitem__(self, index):
        # load segmentation from h5
        seg_id = self.id_lists[index]
        _, sample = load_segment(seg_id, self.seg_txt, self.fid)
        if sample is not None:
            boundary = get_boundary(sample)
        else:
            boundary = None
        seg_id = torch.from_numpy(np.array(seg_id, dtype=np.int32))
        sample = torch.from_numpy(np.array(sample > 0, dtype=np.int32))
        boundary = torch.from_numpy(np.array(boundary, dtype=np.int32))
        return seg_id, sample, boundary
    
    def __len__(self):
        return len(self.id_lists)



