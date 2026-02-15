import open3d as o3d
import numpy as np

def calculate_fscore(pr: o3d.geometry.PointCloud, gt: o3d.geometry.PointCloud, th: float = 0.01):
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    d1 = pr.compute_point_cloud_distance(gt)
    d2 = gt.compute_point_cloud_distance(pr)
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))
        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
    cd = (np.mean(d1) + np.mean(d2)) / 2
    return fscore, cd

