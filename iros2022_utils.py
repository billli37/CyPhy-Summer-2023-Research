from math import factorial
from random import uniform
from tarfile import GNU_MAGIC
from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from matteo_icp import *
from lidar_utils import * 
from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
import open3d as o3d
import teaserpp_python
import time
from scipy.linalg import solve_discrete_are, solve, lstsq, sqrtm, logm

def angle_projection(theta):
    return -np.pi + np.mod(theta + np.pi, 2*np.pi)

def ang_to_rot(theta):
    a = np.stack((np.cos(theta), -np.sin(theta)), axis = 0)
    b = np.stack((np.sin(theta), np.cos(theta)), axis = 0)

    # print(np.stack((a, b), axis = 0))

    return np.stack((a, b), axis = 0)

def ang_to_rot3d(theta):
    a = np.stack((np.cos(theta), -np.sin(theta), 0.0), axis = 0)
    b = np.stack((np.sin(theta), np.cos(theta), 0.0), axis = 0)
    c = np.stack((0.0, 0.0, 1.0), axis = 0)

    return np.stack((a, b, c), axis = 0)



def compare_translations_rotations(angle_hat,angle_gd, t_hat, t_gd, pose1_heading, pose2_heading, alg):

    rot_hat = ang_to_rot(angle_hat)
    if alg == "pasta":
        rot, trans = pasta_to_optitrack(rot_hat, t_hat, pose1_heading, pose2_heading)
        
    else:
        rot, trans = o3d_to_optitrack(rot_hat, t_hat, pose1_heading, pose2_heading)
        # rot = o3d_to_optitrack(rot_hat, t_hat, pose1_heading, pose2_heading)

    
    angle = rot_to_ang(rot)
    
    return abs(np.degrees(angle_projection(angle-angle_gd))), np.linalg.norm(trans - t_gd)
    # return abs(np.degrees(angle_projection(angle-angle_gd)))


def vee_operator(matrix):
    rows, cols = matrix.shape
    if rows != cols or rows % 2 == 0:
        raise ValueError("Input matrix must be square and odd-dimensional")
    
    vector = []
    for i in range(rows):
        for j in range(i + 1, cols):
            vector.append(matrix[j, i])
    
    return np.array(vector)

def compare_rototranslation_3d(est_rot, true_rot, est_trans, true_trans):
    # YOUR CODE GOES HERE
    error_rot = np.linalg.norm(est_rot - true_rot)
    error_trans = np.linalg.norm(est_trans - np.append(true_trans, 0.0))
    return error_rot, error_trans


def rototranslation_3d(est_rot, true_rot, est_trans, true_trans):
    # Using geodesic distance in SE3
    #TODO: CHANGE
    trans_diff = est_trans - np.append(true_trans, 0.0)
    true_rot_matrix = np.transpose(ang_to_rot3d(true_rot))

    mult_inverse_rot = np.matmul(true_rot_matrix, est_rot)
    log_rot = logm(mult_inverse_rot)
    return np.sqrt(np.square(vee_operator(log_rot)) + np.square(trans_diff))

def rototranslation_2d_new(est_rot, true_rot, est_trans, true_trans):
    # Using geodesic distance in SE3
    #TODO: CHANGE
    trans_diff = est_trans - true_trans
    print(ang_to_rot3d(np.transpose(true_rot)))
    print(est_rot)
    mult_inverse_rot = ang_to_rot3d(np.transpose(true_rot)) * est_rot
    log_rot = logm(mult_inverse_rot)
    return np.sqrt(np.square(vee_operator(log_rot)) + np.append(np.square(trans_diff), 0.0))


def pasta_to_o3d(pasta_rot, pasta_trans):
    
    t_pasta_o3d = -pasta_rot.T @ pasta_trans
    rot_pasta_o3d = pasta_rot.T

    trans_matrix = np.eye(4)
    trans_matrix[:2,:2] = rot_pasta_o3d
    trans_matrix[:2,3] = t_pasta_o3d

    return trans_matrix

def o3d_to_optitrack(rot, trans, pose1_heading, pose2_heading):

    print("rot shape", rot.shape)
    print("trans shape:", trans.shape)

    r2 = ang_to_rot(np.radians(pose2_heading))
    return rot.T
    # return rot.T, -r2 @ trans

def pasta_to_optitrack(rot, trans, pose1_heading, pose2_heading):
    r2 = ang_to_rot(np.radians(pose2_heading))
    r1 = ang_to_rot(np.radians(pose1_heading))
    return rot, r1 @ trans

def shape_moments_2d(tris):

    # tris : resulting shape = (nb_simplices, 3, 2), for each simplex you have a 3 by 2 matrix containing the vertex + the next vertex + the hull centroid
    a = tris[:,1,:] - tris[:,0,:]  # roll(vertices) - vertices  (matrix of vectors)
    b = tris[:,2,:] - tris[:,0,:]  # vertices centroid - vertices  (matrix of vectors)

    areas = np.absolute(np.linalg.det(np.stack((a,b), axis=2)))  # here no need to divide by two since this is just the weighting for the average
    c = np.average(np.mean(tris, axis=1), axis=0, weights=areas)

    d = tris[:,0,:] - c[None,:]

    M = np.stack((d,a,b), axis=1)  # shape = (nb_vrtices, 3, 2)

    coeffs = np.array([
            [1,   1/3,  1/3],
            [1/3, 1/6,  1/12],
            [1/3, 1/12, 1/6]
        ])

    # To understand the following, check https://gharesifard.github.io/pdfs/IROS_2022_NOTE.pdf
    covs_11 = np.einsum("ijk,ijk->i", coeffs[None,:,:], np.einsum("ij,ik->ijk", M[:,:,0], M[:,:,0]))
    covs_12 = np.einsum("ijk,ijk->i", coeffs[None,:,:], np.einsum("ij,ik->ijk", M[:,:,0], M[:,:,1]))
    covs_21 = covs_12
    covs_22 = np.einsum("ijk,ijk->i", coeffs[None,:,:], np.einsum("ij,ik->ijk", M[:,:,1], M[:,:,1]))

    covs_1 = np.stack((covs_11,covs_12), axis=1)
    covs_2 = np.stack((covs_21,covs_22), axis=1)
    cov = np.stack((covs_1,covs_2), axis=2)
    cov = np.average(cov, axis=0, weights=areas)

    return c, cov


def shape_moments_3d(tris):

    # tris : resulting shape = (nb_simplices, 4, 3), for each simplex you have a 4 by 3 matrix containing the vertex + the next vertex + the next vertex + the hull centroid

    # Now building V
    a = tris[:,1,:] - tris[:,0,:]  # v1 - v0  # shape = (nb_simplices, 3)
    b = tris[:,2,:] - tris[:,0,:]  # v2 - v0  # shape = (nb_simplices, 3)
    c = tris[:,3,:] - tris[:,0,:]  # v3 - v0  # shape = (nb_simplices, 3)

    V = np.stack((a,b,c), axis=2) # [v1-v0  v2-v0  v3-v0]  # shape = (nb_simplices, 3, 3)

    # Compute the First moment (cent : centroid)
    volumes = np.absolute(np.linalg.det(V))  # shape = (nb_simplices,)
    centroid_simplices = np.mean(tris, axis=1)  # shape = (nb_simplices, 3)
    cent = np.average(centroid_simplices, axis=0, weights=volumes)  # first_moment = sum of vertices average of each simplex, weighted by volume

    # Compute the covariance...
    K = np.array([
                [1,   1/4,  1/4,  1/4],
                [1/4,  1/10,  1/20,  1/20],
                [1/4,  1/20,  1/10,  1/20],
                [1/4,  1/20,  1/20,  1/10]
                ])

    d = tris[:,0,:] - cent[None,:]  # v0 - q
    V_tilde = np.stack((d,a,b,c), axis=2)  # shape = (nb_simplices, 3, 4)
    temp = V_tilde @ K  # shape = (nb_simplices, 3, 4)
    covs = np.einsum("ijl, ikl -> ijk", temp, V_tilde)  # shape = (nb_simplices, 3, 3)
    cov = np.average(covs, axis=0, weights=volumes)  # shape = (3, 3)

    return cent, cov


def rectify_basis(basis):
    if np.cross(basis[:,0], basis[:,1]) > 0:
        basis[:,0] = -basis[:,0]
    return basis

def choose_direction(mu, eigvec, cloud):
    dots = np.einsum("ij,j->i", cloud - mu[None,:], eigvec[:,1])
    M = np.max(dots)
    m = np.min(dots)
    if abs(M) < abs(m):
        return rectify_basis(-eigvec)
    else:
        return rectify_basis(eigvec)

def choose_direction_3d(mu, eigvec, cloud, direction_mode = 'max'):
    "pick the direction in which the perturbation is maximum and hope it is preserved under the transformation"
    
    if direction_mode == 'max':
    
        dots_1 = np.einsum("ij,j->i", cloud - mu[None,:], eigvec[:,2])
        dots_2 = np.einsum("ij,j->i", cloud - mu[None,:], eigvec[:,1])
        M1 = np.max(dots_1)
        m1 = np.min(dots_1)

        M2 = np.max(dots_2)
        m2 = np.max(dots_2)
        if abs(M1) < abs(m1):
            eigvec[:,2] = - eigvec[:,2]
        
        if abs(M2) < abs(m2):
            eigvec[:,1] = - eigvec[:,1]

        if np.linalg.det(eigvec) < 0:
            eigvec[:,0] = - eigvec[:,0]

        return eigvec


    if direction_mode == 'mean':
        dots_1 = np.einsum("ij,j->i", cloud - mu[None,:], eigvec[:,2])
        dots_2 = np.einsum("ij,j->i", cloud - mu[None,:], eigvec[:,1])

        if np.mean(dots_1) < 0:
            eigvec[:,2] = - eigvec[:,2]
        
        if np.mean(dots_2) < 0:
            eigvec[:,1] = - eigvec[:,1]

        if np.linalg.det(eigvec) < 0:
            eigvec[:,0] = - eigvec[:,0]

        return eigvec



def choose_direction_skewness(mu, eigvec, tris, **kwargs):
    return_skewness = kwargs.pop('return_skewness', False)
    if return_skewness:
        return (directional_skewness(tris, mu, eigvec[:,1]), directional_skewness(tris, mu, eigvec[:,0]))

    if directional_skewness(tris, mu, eigvec[:,1]) >= 0:
        return rectify_basis(eigvec)
    else:
        return rectify_basis(-eigvec)

def choose_direction_oracle(eigvec, oracle):
    if np.dot(eigvec[:,1], oracle[:,1]) >= 0:
        return rectify_basis(eigvec)
    else:
        return rectify_basis(-eigvec)

def directional_skewness(tris, p, w):
    def aux_n(x):
        if x == 0: return np.array([0,0])
        elif x == 1: return np.array([1,0])
        else: return np.array([0,1])

    coeff = np.empty((3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                n = aux_n(i) + aux_n(j) + aux_n(k)
                coeff[i,j,k] = factorial(n[0]) * factorial(n[1]) / factorial(2 + n[0] + n[1])
    coeff *= 2

    d = tris[:,0,:] - p[None,:]
    a = tris[:,1,:] - tris[:,0,:]
    b = tris[:,2,:] - tris[:,0,:]
    areas = np.absolute(np.linalg.det(np.stack((a,b), axis=2)))
    M = np.stack((d,a,b), axis=1)
    M = np.einsum("ijk,k->ij", M, w / np.linalg.norm(w))

    skews = np.einsum("ijkl,ijkl->i", coeff[None,:,:,:], np.einsum("ij,ik,il->ijkl", M, M, M))
    skewness = np.average(skews, weights=areas)

    return skewness

def uniformize(cloud, point_num):
    L = np.sum(np.linalg.norm(cloud-np.roll(cloud,1,axis=0), axis=1))
    # print('L:', L)
    n = cloud.shape[0]
    # print('n:', n)
    lpp = L/point_num
    # print('lpp:', lpp)
    # print('cloudshape:', cloud.shape)

    new_cloud = np.empty((point_num,2))
    s = 0
    base_idx = 0
    vec_length = np.linalg.norm(cloud[(base_idx+1)%n]-cloud[base_idx])
    versor = (cloud[(base_idx+1)%n]-cloud[base_idx])/vec_length
    for i in range(new_cloud.shape[0]):
        new_cloud[i] = cloud[base_idx%n] + versor*s

        s += lpp
        # print(s)
        # print(lpp)
        # print(vec_length)
        while s >= vec_length:
            s -= vec_length
            base_idx += 1
            vec_length = np.linalg.norm(cloud[(base_idx+1)%n]-cloud[base_idx%n])
            versor = (cloud[(base_idx+1)%n]-cloud[base_idx%n])/vec_length
    
    # print('uniformize return')
    return new_cloud

def uniformize_in_polar(cloud, angle_num):
    polar_scan = cartesian_to_polar(cloud)
    total_range = np.sum((polar_scan[:,1]-np.roll(polar_scan[:,1],1))%(2*np.pi))
    n = cloud.shape[0]
    angle_increment = total_range/angle_num
    new_polar_cloud = np.empty((angle_num,2))
    a = 0
    base_idx = 0
    angle_diff = (polar_scan[:,1][(base_idx+1)%n]-polar_scan[:,1][base_idx])%(2*np.pi)
    xB, yB = cloud[(base_idx+1)%n]
    xA, yA = cloud[(base_idx)%n]
    alpha = (xB - xA)/(yB - yA)
    term = xA - alpha * yA

    for i in range(angle_num-1):
        new_polar_cloud[i][1] = polar_scan[base_idx%n][1] + a
        new_polar_cloud[i][0] = term/(np.cos(new_polar_cloud[i][1]) - alpha*np.sin(new_polar_cloud[i][1]))
        a += angle_increment

        while a >= angle_diff:
            a -= angle_diff
            base_idx += 1
            angle_diff = (polar_scan[:,1][(base_idx+1)%n]-polar_scan[:,1][base_idx])%(2*np.pi)
            xB, yB = cloud[(base_idx+1)%n]
            xA, yA = cloud[(base_idx)%n]
            alpha = (xB - xA)/(yB - yA)
            term = xA - (alpha * yA)

    return polar_to_cartesian(new_polar_cloud)




def kernelize(cloud, kernel, normalize=True):
    n = cloud.shape[0]
    new_cloud = np.empty_like(cloud)

    if normalize:
        Z = 0
        for elem in kernel:
            Z += elem[1]

        if Z == 0:
            Z = 1
        else:
            Z = 1/Z
    else:
        Z = 1

    for i in range(n):
        vec = np.zeros(cloud.shape[1])
        for elem in kernel:
            # print((i+elem[0])%n)
            vec += Z*elem[1]*cloud[(i+elem[0])%n]
        # exit()
        new_cloud[i] = vec

    return new_cloud

def cloud_moments_aux(cloud, dir_mode=None, **kwargs):

    if kwargs.pop('mode_3d', False):
        hull = ConvexHull(cloud)
        verts = cloud[hull.vertices,:]
        inner = np.mean(verts, axis=0)     
        simplices = cloud[hull.simplices, :]
        nb_simplices = np.shape(simplices)[0]
        inner_temp = np.broadcast_to(inner, shape = (nb_simplices, 1, 3))
        # add the inner vertex
        tris = np.concatenate((simplices, inner_temp), axis = 1) # shape = (nb_simplices, 4, 3)

        mu, sigma = shape_moments_3d(tris)
        eigval, eigvec = np.linalg.eigh(sigma)


        choose_direct_mode = kwargs.pop('direction_mode', 'max')
        print('direction_mode: ', choose_direct_mode)
        eigvec = choose_direction_3d(mu, eigvec, cloud, direction_mode=choose_direct_mode)

        return mu, sigma, eigval, eigvec


    if kwargs.pop('preconvexify', False):
        cloud = cloud[ConvexHull(cloud).vertices,:]

    if kwargs.pop('uniformize', False):
        point_num = kwargs.pop('point_num', 1000)
        cloud = uniformize(cloud, point_num)

    if kwargs.pop('kernelize', False):
        kernel = kwargs.pop('kernel')
        normalize = kwargs.pop('normalize', True)
        cloud = kernelize(cloud, kernel, normalize)
        # print(cloud)
        # plt.figure()
        # plt.scatter(cloud[:,0], cloud[:,1], s=1)
        # plt.show()
        # exit()

    if kwargs.pop('convexify', False):
        cloud = cloud[ConvexHull(cloud).vertices,:]

    if kwargs.pop('sample_mode', False):
        mu = np.mean(cloud, axis=0)
        temp = cloud-mu[None,:]
        sigma = np.mean(np.einsum('ij,ik->ijk', temp, temp), axis=0)
        tris = np.stack((cloud,np.roll(cloud,1,axis=0),np.broadcast_to(mu, (cloud.shape[0],2))), axis=1)
    else:
        if kwargs.pop("hull_mode", True):
            hull = ConvexHull(cloud)
            verts = cloud[hull.vertices,:]
            inner = np.mean(verts, axis=0)
            tris = np.stack((verts,np.roll(verts,1,axis=0),np.broadcast_to(inner, (verts.shape[0],2))), axis=1)  # resulting shape = (nb_vertices, 3, 2), for each vertex you have a 3 by 2 matrix containing the vertex + the next vertex + the hull centroid
        else:
            # Branch not implemented yet
            if kwargs.pop("hybrid", False):
                n = cloud.shape[0]
                threshold = kwargs.pop("threshold", np.inf)
                mask = np.zeros(n)
                
                mycloud = np.copy(cloud)
                i = 0
                while i < mycloud.shape[0]:
                    pass
            else:
                verts = cloud
                inner = np.zeros(2)
                tris = np.stack((verts,np.roll(verts,1,axis=0),np.broadcast_to(inner, (verts.shape[0],2))), axis=1)

        mu, sigma = shape_moments_2d(tris)

    eigval, eigvec = np.linalg.eigh(sigma)


    if kwargs.pop('return_skewness', False):
        print("SKEWNESS YEAH")
        skewness_1, skewness_2 = choose_direction_skewness(mu, eigvec, tris, return_skewness = True)
        return mu, sigma, eigval, eigvec, skewness_1, skewness_2 

    if dir_mode is not None:
        if dir_mode == "max":
            eigvec = choose_direction(mu, eigvec, cloud)
        if dir_mode == "skew":
            eigvec = choose_direction_skewness(mu, eigvec, tris)
        if dir_mode == "oracle":
            eigvec = choose_direction_oracle(eigvec, kwargs.get("oracle"))

    if kwargs.pop('return_hull', False):
        return mu, sigma, eigval, eigvec, hull
    
    return mu, sigma, eigval, eigvec

def cloud_moments(cloud, dir_mode=None, **kwargs):
    
    return_hull =  kwargs.pop('return_hull', False)
    return_skewness =  kwargs.pop('return_skewness', False)
    print("return_hull:", return_hull)
    if isinstance(cloud, list):
        n = len(cloud)
        mu = np.empty((n,2))
        sigma = np.empty((n,2,2))
        eigval = np.empty((n,2))
        eigvec = np.empty((n,2,2))
        for i in range(n):
            if return_hull:
                return "cannot process a list of clouds and return the hulls at the same time"
            if return_skewness:
                return "cannot process a list of clouds and return the skewnesses at the same time"
            mu[i], sigma[i], eigval[i], eigvec[i] = cloud_moments_aux(cloud[i], dir_mode, **kwargs)
    
    else:
        if return_hull:
            if return_skewness:
                return "cannot process return_hull and return the skewnesses at the same time"
            print("return_hull:", return_hull)
            mu, sigma, eigval, eigvec, hull = cloud_moments_aux(cloud, dir_mode, return_hull = return_hull, **kwargs)
            return mu, sigma, eigval, eigvec, hull
        

        mu, sigma, eigval, eigvec = cloud_moments_aux(cloud, dir_mode, **kwargs)

    if return_skewness:
        print("SKEWNESS YEAH")
        mu, sigma, eigval, eigvec, skewness1, skewness2 = cloud_moments_aux(cloud, dir_mode, return_skewness = True, **kwargs)
        return mu, sigma, eigval, eigvec, skewness1, skewness2
    return mu, sigma, eigval, eigvec

def rot_to_ang(R):
    R = np.clip(R, -1, 1)
    c, s = R[0,0], R[1,0]
    # if s >= 0:
    if s.any() >= 0:
        return np.arccos(c)
    else:
        return -np.arccos(c)

def rototranslation_2d(cloud_a, cloud_b, dir_mode=None, **kwargs):

    return_hull =  kwargs.pop('return_hull', False)
    wrapping_90 = kwargs.pop('wrapping_90', False)
    return_skewness =  kwargs.pop('return_skewness', False)



    if return_hull:
        mu_a, sigma_a, _, eigvec_a, hull_a = cloud_moments(cloud_a, dir_mode, return_hull = return_hull, **kwargs)
        mu_b, sigma_b, _, eigvec_b, hull_b = cloud_moments(cloud_b, dir_mode, return_hull = return_hull, **kwargs)


        angles = [0, np.pi/2, -np.pi/2, np.pi]
        volumes = []
        # pick the rotation depending on how much the convex hull volumes correspond
        # first option, classical mode
        V_a = eigvec_a
        V_b = eigvec_b
    
        R_hat = V_a @ V_b.T
        p_hat = mu_a - R_hat @ mu_b
        ang_hat = rot_to_ang(R_hat)



        #if wrapping_90 and (ang_hat > np.pi/2 or ang_hat < -np.pi/2):
        #    ang_hat += np.pi
        #    R_hat = ang_to_rot(ang_hat)
        #    p_hat = mu_a - R_hat @ mu_b
        #    return p_hat, R_hat, ang_hat


        # rotate the point cloud
        if np.abs(ang_hat) > 1.4*np.pi/2:
            volumes.append(0)

        else:
            cloud_b_rotated = p_hat.T + cloud_b @ R_hat.T
            hull_b_rotated = ConvexHull(cloud_b_rotated)
            hull_intersec = hull_intersection(hull_a, hull_b_rotated)
            volume_classical = hull_intersec.volume
            volumes.append(volume_classical)
        
        show =  kwargs.pop('show', False)
        if show:
            print("ANGLE ADDED:", angles[0])
            print("volume:",  volume_classical)
            plot_cloud_data([cloud_a, cloud_b_rotated], show = True)
        
        # second option:
        
        for ang_diff in angles[1:]:
            
            new_ang = angle_projection(ang_diff + ang_hat)
            if wrapping_90 and (np.abs(new_ang) > np.pi/2):
                volumes.append(0)
                continue
            new_rot = ang_to_rot(new_ang)
            new_trans = mu_a - new_rot @ mu_b

            # rotate the point cloud
            cloud_b_rotated = new_trans.T + cloud_b @ new_rot.T
            hull_b_rotated = ConvexHull(cloud_b_rotated)
            hull_intersec = hull_intersection(hull_a, hull_b_rotated)
            volume_reverse = hull_intersec.volume
            volumes.append(volume_reverse)

            if show:
                print("ANGLE ADDED:", ang_diff)
                print("volume:",  volume_reverse)
                plot_cloud_data([cloud_a, cloud_b_rotated], show = True)
                


        #print(volumes)
        best_match = np.argmax(volumes)
        angle = angles[best_match]
        new_ang = angle + ang_hat
        R_hat = ang_to_rot(new_ang)
        p_hat = mu_a - R_hat @ mu_b
        
        return p_hat, R_hat, rot_to_ang(R_hat)



    if return_skewness:
        print("SKEWNESS YEAH")
        mu_a, sigma_a, _, eigvec_a, skewness_1a, skewness_2a = cloud_moments(cloud_a, dir_mode, return_skewness = True, **kwargs)
        mu_b, sigma_b, _, eigvec_b, skewness_1b, skewness_2b = cloud_moments(cloud_b, dir_mode, return_skewness = True, **kwargs)

        if np.abs(skewness_1a) > np.abs(skewness_2a):
            if skewness_1a >= 0:
                eigvec_a = rectify_basis(eigvec_a)
            if skewness_1a < 0:
                eigvec_a = rectify_basis(-eigvec_a)
            if skewness_1b >= 0:
                eigvec_b = rectify_basis(eigvec_b)
            if skewness_1b < 0:
                eigvec_b = rectify_basis(-eigvec_b)
        
        else:
            if skewness_2a >= 0:
                eigvec_a = rectify_basis(eigvec_a)
            if skewness_2a < 0:
                eigvec_a = rectify_basis(-eigvec_a)
            if skewness_2b >= 0:
                eigvec_b = rectify_basis(eigvec_b)
            if skewness_2b < 0:
                eigvec_b = rectify_basis(-eigvec_b)
            

    else:
        mu_a, sigma_a, _, eigvec_a = cloud_moments(cloud_a, dir_mode, **kwargs)
        mu_b, sigma_b, _, eigvec_b = cloud_moments(cloud_b, dir_mode, **kwargs)

    # V_a = np.roll(eigvec_a, 1, axis=1)
    # V_b = np.roll(eigvec_b, 1, axis=1)
    V_a = eigvec_a
    V_b = eigvec_b
    
    R_hat = V_a @ V_b.T
    p_hat = mu_a - R_hat @ mu_b
    ang_hat = rot_to_ang(R_hat)
    #print(kwargs.pop('direction_mode', 'max'))
    
    if wrapping_90 and (np.abs(ang_hat) > np.pi/2):
        ang_hat += np.pi
        R_hat = ang_to_rot(ang_hat)
        p_hat = mu_a - R_hat @ mu_b
        return p_hat, R_hat, ang_hat

    if kwargs.pop('mode_3d', False):
        return p_hat, R_hat

    
    else:

        ang_hat = rot_to_ang(R_hat)
        return p_hat, R_hat, ang_hat

def merge_snapshots(data):
    new_data = np.empty((len(data), 360))
    for i in range(new_data.shape[0]):
        for j in range(new_data.shape[1]):
            new_data[i,j] = np.mean(data[i][data[i][:,j] != np.inf ,j])
    new_data[np.isnan(new_data)] = np.inf
    return new_data

"""
def depth_to_cartesian(data, A1=False):
    n = data.shape[0]

    angles = np.arange(data.shape[1])*2*np.pi/data.shape[1]
    if A1:
        angles /= 1.5
    rays = np.stack((np.cos(angles), np.sin(angles)), axis=1)

    cloud_data = []
    for i in range(n):
        pose_cloud = rays * data[i][:,None]
        pose_cloud = pose_cloud[data[i] != np.inf, :]
        cloud_data.append(pose_cloud)

    return cloud_data
"""
def relative_poses_estimate(cloud_data):
    n = len(cloud_data)
    pose_hat = np.empty((n,n,3))
    for i in range(n):
        for j in range(n):
            p_hat, _, ang_hat = rototranslation_2d(cloud_data[i], cloud_data[j], dir_mode="skew")
            pose_hat[i,j] = np.concatenate((p_hat, [ang_hat]))

    return pose_hat

def relative_poses_estimate_incremental(cloud_data, **kwargs):
    mode = kwargs.pop('mode', 'hull')
    n = len(cloud_data)
    pose_hat = np.empty((n-1,3))
    mu = np.empty((n,2))
    eigvec = np.empty((n,2,2))
    mu[0], _, _, eigvec[0] = cloud_moments(cloud_data[0], dir_mode="skew")
    oracle = eigvec[0]
    for i in range(n-1):
        mu[i+1], _, _, eigvec[i+1] = cloud_moments(cloud_data[i+1], dir_mode="oracle", oracle=oracle)
        oracle = eigvec[i+1]
        if mode == 'hull':
            p_hat, _, ang_hat = rototranslation_2d(cloud_data[i], cloud_data[i+1], dir_mode="oracle", oracle=oracle)
        elif mode == 'icp':
            p_hat, _, ang_hat = pose_hat_icp_aux(cloud_data[i], cloud_data[i+1], False)
        elif mode == 'icp-guess':
            p_hat, _, ang_hat = rototranslation_2d(cloud_data[i], cloud_data[i+1], dir_mode="oracle", oracle=oracle)
            pose_guess = np.concatenate((p_hat, [ang_hat*180/np.pi]))
            p_hat, _, ang_hat = pose_hat_icp_guess_aux(cloud_data[i], cloud_data[i+1], pose_guess, False)
        elif mode == 'gicp':
            p_hat, _, ang_hat = pose_hat_icp_aux(cloud_data[i], cloud_data[i+1], True)
        elif mode == 'gicp-guess':
            p_hat, _, ang_hat = rototranslation_2d(cloud_data[i], cloud_data[i+1], dir_mode="oracle", oracle=oracle)
            pose_guess = np.concatenate((p_hat, [ang_hat*180/np.pi]))
            p_hat, _, ang_hat = pose_hat_icp_guess_aux(cloud_data[i], cloud_data[i+1], pose_guess, True)
        else:
            print("FUCK YOU")
            exit()

        pose_hat[i] = np.concatenate((p_hat, [ang_hat]))

    if kwargs.pop('moments', False):
        return pose_hat, mu, eigvec
    else:
        return pose_hat

def pose_hat_ndt_aux(cloud_src, cloud_dst):
    p, R = ndt_align(cloud_src, cloud_dst)
    p_hat = -R.T @ p
    R_hat = R.T
    theta_hat = rot_to_ang(R_hat)
    return p_hat, R_hat, theta_hat

def pose_hat_icp_aux(cloud_src, cloud_dst, gicp=False):
    if not gicp:
        p, R = icp_align(cloud_src, cloud_dst)
    else:
        p, R = gicp_align(cloud_src, cloud_dst)
    p_hat = -R.T @ p
    R_hat = R.T
    theta_hat = rot_to_ang(R_hat)
    return p_hat, R_hat, theta_hat

def pose_hat_icp_guess_aux(cloud_src, cloud_dst, pose_guess, gicp=False):
    p_hat_guess = pose_guess[:2]
    theta_hat_guess = pose_guess[2]*np.pi/180
    R_hat_guess = np.array([
        [np.cos(theta_hat_guess), -np.sin(theta_hat_guess)],
        [np.sin(theta_hat_guess), np.cos(theta_hat_guess)]
    ])

    mid_cloud = np.einsum("ij,kj->ki", R_hat_guess, cloud_dst) + p_hat_guess[None,:]
    if not gicp:
        p_hat_icp, R_hat_icp = icp_align(cloud_src, mid_cloud)
    else:
        p_hat_icp, R_hat_icp = gicp_align(cloud_src, mid_cloud)
    p_hat_com = p_hat_guess - p_hat_icp
    R_hat_com = R_hat_icp.T @ R_hat_guess
    theta_hat_com = rot_to_ang(R_hat_com)

    return p_hat_com, R_hat_com, theta_hat_com

def relative_poses_estimate_icp(cloud_data, gicp=False):
    n = len(cloud_data)
    pose_hat_icp = np.empty((n,n,3))
    for i in range(n):
        for j in range(n):
            print("icp:", i,j)
            p_hat_icp, _, theta_hat_icp = pose_hat_icp_aux(cloud_data[i], cloud_data[j], gicp=gicp)
            pose_hat_icp[i,j] = np.concatenate((p_hat_icp, [theta_hat_icp*180/np.pi]))

    return pose_hat_icp

def relative_poses_estimate_icp_guess(cloud_data, pose_guess, gicp=False):
    n = len(cloud_data)
    pose_hat_icp_guess = np.empty((n,n,3))
    for i in range(n):
        for j in range(n):
            print("icp guess:", i,j)
            p_hat_icp_guess, _, theta_hat_icp_guess = pose_hat_icp_guess_aux(cloud_data[i], cloud_data[j], pose_guess[i,j], gicp=gicp)
            pose_hat_icp_guess[i,j] = np.concatenate((p_hat_icp_guess, [theta_hat_icp_guess*180/np.pi]))
            # if i == 1 or j == 1:
            #     print(i,j)
            #     print("guess:\n", pose_guess[i,j] * np.array([1.0, 1.0, 180/np.pi]))
            #     print("icp_guess:\n", pose_hat_icp_guess[i,j])

    return pose_hat_icp_guess

# gt = ground truth
def relative_pose_star_aux(gt_src, gt_dst, deg=True):
    ang_star = gt_dst[2] - gt_src[2]
    if deg:
        if ang_star < -180:
            ang_star = 360 + ang_star
        if ang_star > 180:
            ang_star = -360 + ang_star
        ang_a_rad = gt_src[2]*np.pi/180
    else:
        ang_star = angle_projection(ang_star)
        ang_a_rad = gt_src[2]
    R = np.array([
        [np.cos(ang_a_rad), np.sin(ang_a_rad)],
        [-np.sin(ang_a_rad), np.cos(ang_a_rad)]
    ])
    p_star = R @ (gt_dst[:2] - gt_src[:2])
    pose_star = np.concatenate((p_star, [ang_star]))
    return pose_star

def relative_poses_star_incremental(ground_truth):
    n = ground_truth.shape[1]
    pose_star = np.empty((n-1,3))
    for  i in range(n-1):
        pose_star[i] = relative_pose_star_aux(ground_truth[:,i], ground_truth[:,i+1])

    return pose_star

def relative_poses_star(ground_truth):
    n = ground_truth.shape[1]
    pose_star = np.empty((n,n,3))
    for i in range(n):
        for j in range(n):
            ang_star = ground_truth[2,j] - ground_truth[2,i]
            if ang_star < -180:
                ang_star = 360 + ang_star
            if ang_star > 180:
                ang_star = -360 + ang_star
            ang_a_rad = ground_truth[2,i]*np.pi/180
            R = np.array([
                [np.cos(ang_a_rad), np.sin(ang_a_rad)],
                [-np.sin(ang_a_rad), np.cos(ang_a_rad)]
            ])
            p_star = R @ (ground_truth[:2,j] - ground_truth[:2,i])
            pose_star[i,j] = np.concatenate((p_star, [ang_star]))
    
    return pose_star


def project_to_orthogonal(matrix):

    #W, V = np.linalg.eig(matrix)
    
    try: 
        W, V = np.linalg.eig(matrix)
    except np.linalg.LinAlgError:
        print(matrix)
        print("error")
    
    #return V @ np.transpose(V)
    
    try:
        answer = matrix @ np.linalg.inv(principal_square_root(matrix @ np.transpose(matrix)))
        return answer
    except np.linalg.LinAlgError:
        answer = matrix @ np.linalg.pinv(principal_square_root(matrix @ np.transpose(matrix)))
        print('Singular matrix in project_to_orthogonal')

        print('Singular matrix : ', principal_square_root(matrix @ np.transpose(matrix)))
        print('Answer', answer)
        return answer

def principal_square_root(matrix):
    #W, V = np.linalg.eig(matrix)

    #return V @ np.sqrt(np.diag(W)) @ np.linalg.inv(V)
    return sqrtm(matrix)


################################################################ SOTA Algorithms ###################################################################



# Teaser

def use_teaser(pc1, pc2, **kwargs):
    

    cbar2 = kwargs.pop('cbar2', 1)
    noise_bound = kwargs.pop('noise_bound', 1)
    estimate_scaling = kwargs.pop('estimate_scaling', True)
    rotation_estimation_algorithm = kwargs.pop('rotation_estimation_algorithm', 'GNC_TLS')
    inlier_selection_mode = kwargs.pop('inlier_selection_mode', 'KCORE_HEU')
    inlier_graph_formulation = kwargs.pop('inlier_graph_formulation', 'COMPLETE')
    gnc_factor = kwargs.pop('gnc_factor', 1.4)
    max_iterations = kwargs.pop('max_iterations', 100)
    cost_threshold = kwargs.pop('cost_threshold', 1e-12)
    max_clique_time_limit = kwargs.pop('max_clique_time_limit', 5)

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()

    if inlier_selection_mode == 'KCORE_HEU':
        mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.KCORE_HEU
    elif inlier_selection_mode == 'PMC_EXACT':
        mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    elif inlier_selection_mode == 'PMC_HEU':
        mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_HEU
    elif inlier_selection_mode == None:
        mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.NONE
    else:
        return 'Unrecognized inlier_selection_mode'
    
    solver_params.inlier_selection_mode = (mode)

    if rotation_estimation_algorithm == 'GNC_TLS':
        mode = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    elif rotation_estimation_algorithm == 'FGR':
        mode = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.FGR
    else:
        return 'Unrecognized rotation_estimation_algorithm'
        

    solver_params.rotation_estimation_algorithm = (mode)


    if inlier_graph_formulation == 'COMPLETE':
        mode = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.COMPLETE
    elif inlier_graph_formulation == 'CHAIN':
        mode = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    else:
        return 'Unrecognized inlier_graph_formulation'

    solver_params.rotation_tim_graph = (mode)
    


    
    solver_params.cbar2 = cbar2
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = estimate_scaling
    solver_params.rotation_gnc_factor = gnc_factor
    solver_params.rotation_max_iterations = max_iterations
    solver_params.rotation_cost_threshold = cost_threshold
    solver_params.max_clique_time_limit = max_clique_time_limit
    print("TEASER++ Parameters are:", solver_params)
    teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    dimension = kwargs.pop('dimension', '2d')
    N1, N2 = len(pc1), len(pc2)
    
    if dimension == '2d':

        first_pc_3d = np.hstack((pc1, np.ones((N1,1))))
        second_pc_3d = np.hstack((pc2, np.ones((N2, 1))))
        
        s = time.time()
        solver.solve(first_pc_3d.T, second_pc_3d.T)
        teaser_time = time.time() - s

        solution = solver.getSolution()

        teaser_rotation = solution.rotation[:2,:2]
        teaser_translation = solution.translation[:2]
        teaser_ang = np.degrees(rot_to_ang(teaser_rotation))

        return teaser_rotation, teaser_translation, teaser_time

    
    if dimension == '3d':

        if np.shape(pc1)[-1] != 3:
            pc1 = pc1.T

        start_teaser = time.time()
        solver.solve(pc1.T, pc2.T)
        time_teaser = time.time() - start_teaser

        solution = solver.getSolution()

        return solution.rotation, solution.translation, time_teaser






def use_generalized_icp(pc1, pc2, **kwargs):

    dimension = kwargs.pop('dimension', '2d')
    N1, N2 = len(pc1), len(pc2)

    icp_version = kwargs.pop('icp_version', 'Generalized')
    max_correspondence_distance = kwargs.pop('max_correspondence_distance', 0.5)

    if icp_version == 'PointToPoint':
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    if icp_version == 'PointToPlane':
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    if icp_version == 'Generalized':
        estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    
    trans_init = kwargs.pop('trans_init', np.eye(4))


    if dimension == '2d':
        pc1 = np.hstack((pc1, np.ones((N1,1))))
        pc2 = np.hstack((pc2, np.ones((N2,1))))
    

    # Converting the point clouds into o3d objects
    pc1_cloud = o3d.geometry.PointCloud()
    pc1_cloud.points = o3d.utility.Vector3dVector(pc1)

    pc2_cloud = o3d.geometry.PointCloud()
    pc2_cloud.points = o3d.utility.Vector3dVector(pc2)
    
    # computing the result

    if icp_version == 'PointToPoint' or 'Generalized':
        s = time.time()
        result = o3d.pipelines.registration.registration_generalized_icp(
            pc1_cloud, pc2_cloud, max_correspondence_distance, trans_init, estimation)
        icp_time = time.time() - s

    if icp_version == 'PointToPlane':
        radius = kwargs.pop('radius', 0.1)
        max_nn = kwargs.pop('max_nn', 30)
        s = time.time()
        pc1_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
        pc2_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
        result = o3d.pipelines.registration.registration_generalized_icp(
            pc1_cloud, pc2_cloud, max_correspondence_distance, trans_init, estimation)
        icp_time = time.time() - s

    if dimension == '2d':
        icp_rotation = result.transformation[:2,:2]
        icp_translation = result.transformation[:2, 3]
    
    if dimension == '3d':
        icp_rotation = result.transformation[:3,:3]
        icp_translation = result.transformation[:3, 3]

    return icp_rotation, icp_translation, icp_time


def use_icp(pc1, pc2, **kwargs):

    # dimension = kwargs.pop('dimension', '2d')
    dimension = kwargs.pop('dimension', '3d')

    N1, N2 = len(pc1), len(pc2)
    icp_version = kwargs.pop('icp_version', 'PointToPlane')
    max_correspondence_distance = kwargs.pop('max_correspondence_distance', 1)

    if icp_version == 'PointToPoint':
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    if icp_version == 'PointToPlane':
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    if icp_version == 'Robust':
        k = kwargs.pop('noise_sigma', 0.1)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k))

    trans_init = kwargs.pop('trans_init', np.eye(4))


    if dimension == '2d':
        pc1 = np.hstack((pc1, np.ones((N1,1))))
        pc2 = np.hstack((pc2, np.ones((N2,1))))
    

    # Converting the point clouds into o3d objects
    pc1_cloud = o3d.geometry.PointCloud()
    pc1_cloud.points = o3d.utility.Vector3dVector(pc1)

    pc2_cloud = o3d.geometry.PointCloud()
    pc2_cloud.points = o3d.utility.Vector3dVector(pc2)

    # computing the result

    if icp_version == 'PointToPoint':
        s = time.time()
        result = o3d.pipelines.registration.registration_icp(
            pc1_cloud, pc2_cloud, max_correspondence_distance, trans_init, estimation, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
        #result = o3d.pipelines.registration.registration_icp(
        #    pc1_cloud, pc2_cloud, max_correspondence_distance, trans_init, estimation)
        
        icp_time = time.time() - s

    
    if icp_version == 'PointToPlane' or icp_version == 'Robust':
        radius = kwargs.pop('radius', 0.1)
        max_nn = kwargs.pop('max_nn', 30)
        s = time.time()
        pc1_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
        pc2_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
        result = o3d.pipelines.registration.registration_icp(
            pc1_cloud, pc2_cloud, max_correspondence_distance, trans_init, estimation)
        icp_time = time.time() - s


    if dimension == '2d':
        icp_rotation = result.transformation[:2,:2]
        icp_translation = result.transformation[:2, 3]
    
    if dimension == '3d':
        icp_rotation = result.transformation[:3,:3]
        icp_translation = result.transformation[:3, 3]

    return icp_rotation, icp_translation, icp_time



def use_pasta(pc1, pc2, **kwargs):

    N1, N2 = len(pc1), len(pc2)
    s = time.time()
    result = rototranslation_2d(pc1, pc2, **kwargs)
    pasta_time = time.time() - s

    rotation = result[1]
    translation = result[0]

    return rotation, translation, pasta_time


def use_pasta_plus_icp(pc1, pc2, **kwargs):
    rot_pasta, trans_pasta, time_pasta = use_pasta(pc1, pc2, **kwargs)
    pasta_o3d = pasta_to_o3d(rot_pasta, trans_pasta)
    rot_pasta_icp, trans_pasta_icp, time_icp = use_icp(pc1, pc2, trans_init = pasta_o3d, **kwargs)

    return rot_pasta_icp, trans_pasta_icp, time_icp+time_pasta



def use_fpfh_ransac(pc1, pc2, **kwargs):

    dimension = kwargs.pop('dimension', '3d')
    N1, N2 = len(pc1), len(pc2)

    voxel_size = kwargs.pop('voxel_size', 0.05)
    radius_normal = kwargs.pop('radius_normal', voxel_size*2)
    radius_feature = kwargs.pop('radius_feature', voxel_size*5)
    max_nn = kwargs.pop('max_nn', 30)
    feature_matching_methods = kwargs.pop('feature_matching_methods', ['Distance', 'EdgeLength'])
    max_correspondence_distance = kwargs.pop('max_correspondence_distance', voxel_size * 1.5)
    edge_length_criterion = kwargs.pop('edge_length_criterion', 0.9)
    normal_criterion = kwargs.pop('normal_criterion', 0.5)
    estimation_method = kwargs.pop('estimation_method', 'PointToPoint')
    mutual_filter = kwargs.pop('mutual_filter', True)

    checkers = []
    if 'Distance' in feature_matching_methods:
        checkers.append(o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance))
    elif 'EdgeLength' in feature_matching_methods:
        checkers.append(o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(edge_length_criterion))
    elif 'Normal' in feature_matching_methods:
        checkers.append(o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(normal_criterion))
    else:
        return 'Incorrect Feature Matching Method'


    if estimation_method == 'PointToPoint':
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif estimation_method == 'PointToPlane':
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif estimation_method == 'GeneralizedICP':
        estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    else:
        return 'Incorrect Estimation Method'

    if dimension == '2d':
        pc1 = np.hstack((pc1, np.ones((N1,1))))
        pc2 = np.hstack((pc2, np.ones((N2,1))))
    
    # Converting the point clouds into o3d objects
    pc1_cloud = o3d.geometry.PointCloud()
    pc1_cloud.points = o3d.utility.Vector3dVector(pc1)

    pc2_cloud = o3d.geometry.PointCloud()
    pc2_cloud.points = o3d.utility.Vector3dVector(pc2)

    s = time.time()
    # Downsampling with the voxel size
    pc1_cloud_down = pc1_cloud.voxel_down_sample(voxel_size)
    pc2_cloud_down = pc2_cloud.voxel_down_sample(voxel_size)

    # Estimate normals
    pc1_cloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))
    pc2_cloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))

    # Compute fpfh_features
    pc1_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc1_cloud_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    pc2_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc2_cloud_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # Feature matching using RANSAC
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(pc1_cloud_down, pc2_cloud_down, pc1_fpfh, pc2_fpfh, mutual_filter, 
                                                                                        max_correspondence_distance, estimation, ransac_n = 3, checkers = checkers)
    fpfh_ransac_time = time.time() - s

    # if dimension == '2d':
    #     fpfh_ransac_rotation = result.transformation[:2,:2]
    #     fpfh_ransac_translation = result.transformation[:2, 3]
    

    if dimension == '3d':
        fpfh_ransac_rotation = result.transformation[:3,:3]
        fpfh_ransac_translation = result.transformation[:3, 3]

    return fpfh_ransac_rotation, fpfh_ransac_translation, fpfh_ransac_time




def use_fast_global_registration(pc1, pc2, **kwargs):

    dimension = kwargs.pop('dimension', '2d')
    N1, N2 = len(pc1), len(pc2)

    voxel_size = kwargs.pop('voxel_size', 0.05)
    radius_normal = kwargs.pop('radius_normal', voxel_size*2)
    max_correspondence_distance = kwargs.pop('max_correspondence_distance', voxel_size * 1.5)
    radius_feature = kwargs.pop('radius_feature', voxel_size*5)
    max_nn = kwargs.pop('max_nn', 30)
    estimation = o3d.pipelines.registration.FastGlobalRegistrationOption(max_correspondence_distance)

    if dimension == '2d':
        pc1 = np.hstack((pc1, np.ones((N1,1))))
        pc2 = np.hstack((pc2, np.ones((N2,1))))

    # Converting the point clouds into o3d objects
    pc1_cloud = o3d.geometry.PointCloud()
    pc1_cloud.points = o3d.utility.Vector3dVector(pc1)

    pc2_cloud = o3d.geometry.PointCloud()
    pc2_cloud.points = o3d.utility.Vector3dVector(pc2)

    s = time.time()
    # Downsampling with the voxel size
    pc1_cloud_down = pc1_cloud.voxel_down_sample(voxel_size)
    pc2_cloud_down = pc2_cloud.voxel_down_sample(voxel_size)

    # Estimate normals
    pc1_cloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))
    pc2_cloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))

    # Compute fpfh_features
    pc1_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc1_cloud_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    pc2_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc2_cloud_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # feature matching
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pc1_cloud_down, pc2_cloud_down, pc1_fpfh, pc2_fpfh, estimation)
    fgr_time = time.time() - s

    if dimension == '2d':
        fgr_rotation = result.transformation[:2,:2]
        fgr_translation = result.transformation[:2, 3]
    
    if dimension == '3d':
        fgr_rotation = result.transformation[:3,:3]
        fgr_translation = result.transformation[:3, 3]

    return fgr_rotation, fgr_translation, fgr_time


# Go-ICP

# helper function to normalize point clouds to live in specified interval
def normalize_point_clouds(pc1,pc2,lb,ub):
    dim = pc1.shape[-1]
    # you could probably do this in less lines/memory by separately finding max/min in each point cloud but I was lazy
    both_clouds = np.array([*pc1,*pc2]) # array with all the points in it
    
    maxval = np.amax(both_clouds,axis=0) # pull largest value of points along axis
    length = maxval-np.amin(both_clouds,axis=0) # compute length along same axis

    # this is the current l-infinity center of both clouds together
    center = maxval-length/2.0
    
    # center both point clouds, scale, then move to desired center
    des_center = np.ones((dim,))*(ub+lb)/2.0 #center of desired interval
    des_length = ub-lb # how wide do we want them?
    
    scale = min(min(des_length/length),1.0) # what is the worst violation?
    pc1 = (pc1-center)*scale + des_center
    pc2 = (pc2-center)*scale + des_center
    
    del both_clouds # don't need that guy anymore, save some memory
    
    return pc1, pc2, scale
    

# this is just a wrapper that builds the custom POINT3D objects for Go-ICP and stores them in a list together
def go_icp_pc_listing(point_cloud):
    dim = point_cloud.shape[-1]
    pc_3dlist = []

    # make point cloud 3D if needed
    if dim < 3:
        point_cloud = np.append(point_cloud,np.ones((point_cloud.shape[0],1)),axis=-1)

    for point in point_cloud:
        x,y,z = point
        pc_3dlist.append(POINT3D(x,y,z))

    return pc_3dlist

# OUTER WRAPPER FUNCTION HERE
# wrapper function to call Go-ICP algorithm
def use_goicp(pc1,pc2,**kwargs):
    dimension = kwargs.pop('dimension','2d')
    dim = pc1.shape[-1] # pull dimension
    N1,N2 = len(pc1),len(pc2) # pull number of points
    pc1,pc2, scale = normalize_point_clouds(pc1,pc2,-1,1) # shifts and scales points to live in [-1,1]
    pc1_3dlist, pc2_3dlist = go_icp_pc_listing(pc1),go_icp_pc_listing(pc2) #creates
    
    goicp = GoICP() # initialize solver object
    goicp.loadModelAndData(N1,pc1_3dlist,N2,pc2_3dlist) # feed it point clouds
    goicp.setDTSizeAndFactor(300,2.0) # need to tune these, distance transform parameters
    print('Building distance transform...')
    start = time.time()
    goicp.BuildDT() # compute distance transform first
    goicp.Register() # Use Go-ICP (BnB) to solve ICP problem
    runtime = time.time() - start
    print('Solved!')
    R = np.array(goicp.optimalRotation())
    t = np.array(goicp.optimalTranslation())/scale # scale is needed ???
    
    del goicp # kill the spare

    return R[0:dim,0:dim], t[0:dim], runtime





















    









################################################################ Plotting Functions #################################################################


def plot_cloud_data(cloud_data, show=False):
    n = len(cloud_data)
    for i in range(n):
        plt.figure()
        plt.scatter(cloud_data[i][:,0], cloud_data[i][:,1], zorder=10, s=9)
        plt.scatter(0, 0, zorder=10, color='yellow', edgecolor='black')
        plt.grid(zorder=0)
        plt.axis('square')
        plt.title("Pose "+str(i+1))
        plt.tight_layout()

    if show:
        plt.show()

def plot_error(pose_err):
    p_err = np.reshape(pose_err[:,:,:2], (-1,2))
    p_err_norm = np.linalg.norm(pose_err[:,:,:2], axis=2)
    ang_err_norm = np.absolute(pose_err[:,:,2])

    plt.figure()
    plt.matshow(p_err_norm, cmap='viridis')
    plt.colorbar()
    plt.title("Position error matrix (2-norm, m)\n")
    plt.xlabel("Target pose index")
    plt.ylabel("Base pose index")
    # plt.tight_layout()
    plt.figure()
    plt.matshow(ang_err_norm, cmap='viridis')
    plt.colorbar()
    plt.title("Angle error matrix (absolute, °)\n")
    plt.xlabel("Target pose index")
    plt.ylabel("Base pose index")
    # plt.tight_layout()

    p_err_norm = np.ravel(p_err_norm)
    ang_err_norm = np.ravel(ang_err_norm)

    plt.figure()
    plt.hist2d(p_err[:,0], p_err[:,1], bins=15)
    plt.title("Position estimation error")
    plt.xlabel("Error - X (m)")
    plt.ylabel("Error - Y (m)")
    plt.tight_layout()
    plt.figure()
    plt.hist(ang_err_norm, bins=15, rwidth=0.9, zorder=3)
    plt.grid(zorder=0)
    plt.title("Angle estimation error (absolute)")
    plt.xlabel("Degrees (°)")
    plt.ylabel("Frequency (no. of samples)")
    plt.tight_layout()
    plt.figure()
    plt.hist(p_err_norm, bins=15, rwidth=0.9, zorder=3)
    plt.grid(zorder = 0)
    plt.title("Position estimation error (2-norm)")
    plt.xlabel("Error (m)")
    plt.ylabel("Frequency (no. of samples)")
    plt.tight_layout()

    # plt.show()

def plot_error_incremental(pose_err):
    fig, ax = plt.subplots(2,1)
    ax[0].plot(np.linalg.norm(pose_err[:,:2], axis=1))
    ax[1].plot(np.absolute(pose_err[:,2]))
    ax[0].set_title("Position error (2-norm)")
    ax[0].set_ylabel("error (m)")
    ax[0].set_xlabel("pose index")
    ax[0].grid()
    ax[1].set_title("Angle error (absolute)")
    ax[1].set_ylabel("error (°)")
    ax[1].set_xlabel("pose index")
    ax[1].grid()
    plt.tight_layout()

def hull_intersection(hull_1, hull_2):
    equ = np.concatenate((hull_1.equations, hull_2.equations), axis=0)
    interior = np.mean(hull_1.points[hull_1.vertices], axis=0)
    # print("interior:", interior)
    # intersection = HalfspaceIntersection(equ, interior)

    # plt.figure()
    try:
        hull = ConvexHull(HalfspaceIntersection(equ, interior).intersections)
    except:
        return ConvexHull(np.array([[0.0, 0.001], [0.001, 0.0], [0.001, 0.001]]))
    # points = hul.points[hul.vertices]
    # points = intersection.intersections
    # plt.fill(points[:,0], points[:,1], color='green', alpha=0.5)
    # plt.show()

    # return intersection
    return hull

# pose hat is for single (cloud_1, cloud_2) tuple
def compute_delta_rho(cloud_1, cloud_2, pose_hat, return_hulls=False):
    hull_1 = ConvexHull(cloud_1)

    ang = pose_hat[2]*np.pi/180
    R = np.array([
        [np.cos(ang), -np.sin(ang)],
        [np.sin(ang), np.cos(ang)]
    ])
    hull_2 = ConvexHull(pose_hat[:2][None,:] + np.einsum("ij,kj->ki", R, cloud_2))
    
    hull_3 = hull_intersection(hull_1, hull_2)

    delta = hull_3.volume / max(hull_1.volume, hull_2.volume)
    hull_4 = ConvexHull(np.concatenate((hull_1.points, hull_2.points), axis=0))    # This is the union of hull 2 and 3
    diam_sq = 0
    for i in hull_4.vertices:
        chord = hull_4.points[i] - hull_4.points[hull_4.vertices[0]]
        diam_sq_new = np.dot(chord, chord)
        diam_sq = diam_sq_new if diam_sq_new > diam_sq else diam_sq
    rho = np.sqrt(diam_sq)/2

    if not return_hulls:
        return delta, rho
    else:
        return delta, rho, hull_1, hull_2, hull_3

def relative_poses_delta_rho(cloud_data, pose_hat):
    n = len(cloud_data)
    delta = np.empty((n,n))
    rho = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            delta[i,j], rho[i,j] = compute_delta_rho(cloud_data[i], cloud_data[j], pose_hat[i,j])

    return delta, rho

def pos_angle_bound_aux(cloud_src, cloud_dst, pose_star):
    delta, rho = compute_delta_rho(cloud_src, cloud_dst, pose_star)
    p_bound = 3*(1-delta)*rho
    _, sigma, eigval, _ = cloud_moments(cloud_src)
    min_eigval_diff = np.absolute(eigval[0] - eigval[1])
    cov_bound = (25*(1-delta)**2 + 8*(1-delta))*rho**2
    if 2*cov_bound > min_eigval_diff:
        print("PANIC!")
        print(min_eigval_diff, cov_bound)
        exit()
    R_I_bound = cov_bound / (min_eigval_diff - 2*cov_bound)
    ang_bound = np.arccos(np.sqrt(2) * R_I_bound)*180/np.pi

    return p_bound, R_I_bound, ang_bound

def transform_bounds(src, dst, pose):
    correction = np.array([1., 1., 180/np.pi])
    delta, rho = compute_delta_rho(src, dst, pose*correction)
    e_p = 2*(1-delta)*rho
    e_R = (2*(1-delta)+4*(1-delta)**2)*rho**2
    mu, _, eigval, _ = cloud_moments(dst)
    eigval_diff = np.absolute(eigval[0] - eigval[1])
    if eigval_diff > 2*e_R:
        ang_bound = np.arccos(1 + 0.5*np.log(1-2*e_R/eigval_diff))
        pos_bound = ang_bound*np.linalg.norm(mu) + e_p
        return pos_bound, ang_bound
    else:
        return np.inf, np.inf

def pos_angle_bound(cloud_data, pose_hat):
    delta, rho = relative_poses_delta_rho(cloud_data, pose_hat)
    p_bound = 3*(1-delta)*rho
    
    n = len(cloud_data)
    R_I_bound = np.empty((n,n))
    for i in range(n):
        _, sigma_i, eigval, _ = cloud_moments(cloud_data[i])
        min_eigval_diff = np.absolute(eigval[0] - eigval[1])
        for j in range(n):
            cov_bound = (25*(1-delta[i,j])**2 + 8*(1-delta[i,j]))*rho[i,j]**2
            if i == 0 and j == 1:
                _, sigma_j, _, _ = cloud_moments(cloud_data[i])
                print(min_eigval_diff, cov_bound, delta[i,j], rho[i,j], np.linalg.norm(sigma_i - sigma_j))
                exit()
            if 2*cov_bound > min_eigval_diff:
                print("PANIC!", i, j)
                # exit()
            R_I_bound[i,j] = cov_bound / (min_eigval_diff - 2*cov_bound)
    ang_bound = np.arccos(np.sqrt(2) * R_I_bound)*180/np.pi

    return p_bound, ang_bound

def plot_hull_intersect(hull_1=None, hull_2=None, hull_3=None):
    plt.figure()
    plt.grid()
    if hull_1 is not None:
        verts_1 = hull_1.points[hull_1.vertices]
        plt.fill(verts_1[:,0], verts_1[:,1], alpha=0.5, color='blue', zorder=10)
    if hull_2 is not None:
        verts_2 = hull_2.points[hull_2.vertices]
        plt.fill(verts_2[:,0], verts_2[:,1], alpha=0.5, color='red',zorder=10)
    if hull_3 is not None:
        verts_3 = hull_3.points[hull_3.vertices]
        plt.fill(verts_3[:,0], verts_3[:,1], facecolor='none', edgecolor='black', hatch='//', zorder=10)

    plt.axis('square')
    plt.title("Convex hull overlap")
    plt.tight_layout()

    # plt.show()
