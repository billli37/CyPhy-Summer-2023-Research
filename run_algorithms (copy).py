import numpy as np
from iros2022_utils import *
import matplotlib.pyplot as plt
import pickle
import os

# outer wrapper for using a given algorithm
# modify this function to change how each individual algorithm is called parameter-wise

def use_algorithm(name,pc1,pc2,dim,**kwargs):
    try :
        trans_init = kwargs['trans_init'] #pop a given initial transformation if it exists
    except :
        trans_init = np.eye(4) # otherwise use the identity

    if name == "teaser":
        rotation, translation, runtime = use_teaser(pc1, pc2, dimension = dim, estimate_scaling = True, 
                        max_iterations = 50, rotation_estimation_algorithm = 'GNC_TLS', noise_bound = 0.062, inlier_selection_mode = 'KCORE_HEU'
                        , cost_threshold = 1e-6, max_clique_time_limit = 100, cbar2 = 1)
    
    elif name == "icp":
        rotation, translation, runtime = use_icp(pc1, pc2, dimension=dim, trans_init = trans_init,  icp_version = 'PointToPoint')
    
    elif name == "icp_plane":
        rotation, translation, runtime = use_icp(pc1, pc2, dimension=dim, trans_init=trans_init, icp_version = 'PointToPlane')
    
    elif name == "robust_icp":
        rotation, translation, runtime = use_icp(pc1, pc2, dimension = dim, trans_init =trans_init, icp_version = 'Robust')
    
    elif name == "pasta":
        rotation, translation, runtime = use_pasta(pc1, pc2, dir_mode = "skew", return_hull = False, wrapping_90 = True, return_skewness = True)

    elif name == "pasta_plus_icp":
        rotation, translation, runtime = use_pasta_plus_icp(pc1, pc2, dir_mode = "skew", dimension=dim, icp_version='PointToPoint', return_hull = False, wrapping_90 = True)
    
    elif name == "fpfh":
        rotation, translation, runtime = use_fpfh_ransac(pc1, pc2, dimension = dim, voxel_size = 0.02, feature_matching_methods = ['Distance', 'EdgeLength'], max_correspondence_distance = 0.5, 
                    estimation_method = 'GeneralizedICP')

    elif name == "gen_icp":
        rotation, translation, runtime = use_generalized_icp(pc1, pc2, dimension = dim, icp_version = 'Generalized',trans_init =trans_init, max_correspondence_distance = 0.01)
    
    elif name == "fgr":
        rotation, translation, runtime = use_fast_global_registration(pc1, pc2, dim = dim, voxel_size = 0.002)

    elif name == "go_icp":
        rotation, translation, runtime = use_goicp(pc1,pc2,dimension=dim)

    else :
        print('Algorithm {} not found, skipping...'.format(name))
        return
    
    if dim == '2d':
        rotation_angle = rot_to_ang(rotation)
        return rotation_angle, translation, runtime
    
    else:
        return rotation, translation, runtime

# function that will look at pose data's x-y coordinates
# divides the covered x-y space into num_bins x num_bins regions
# for each region, determines which indices of pose_data are measurements in that region
# returns the list of indices

def bin_data(pose_data, num_bins):
    xmin = np.min(pose_data[:,0])
    xmax = np.max(pose_data[:,0])
    ymin = np.min(pose_data[:,1])
    ymax = np.max(pose_data[:,1])

    xlength = xmax-xmin
    ylength = ymax-ymin

    xdiv = [xmin + float(i)*xlength/num_bins for i in range(num_bins+1)]
    ydiv = [ymin + float(i)*ylength/num_bins for i in range(num_bins+1)]


    bins = [[None for i in range(num_bins)] for j in range(num_bins)]
    for i in range(num_bins):
        for j in range(num_bins):
            # extract the pose indices that are in the associated bin
            bins[i][j] = np.nonzero((xdiv[i] <= pose_data[:,0]) & (pose_data[:,0] <= xdiv[i+1]) & (ydiv[j] <= pose_data[:,1]) & (pose_data[:,1] <= ydiv[j+1]))[0]

    return bins, xdiv, ydiv

def main(algs = None,dir=None, pairs=None):
    verbose = False # will spew progress through processing pairs

    if pairs is None:
        #pairs = 'random' # do we want to pull the pairs we used for a different algorithm's run?
        #pairs = 'sequential' # do we want to compare sequential (in time) measurements?
        pairs = 'pinned_first'
        #pairs = 'xy_bins'
        #pairs = 'file'
        #pairs = 'lag'
    num_pairs = 500

    if dir is None :
        dir = './experiments' # directory for experiment and its data
    
    source_path = os.path.join(dir,'numpy')
    dest_path = os.path.join('results_'+pairs) # where to dump all the pickled results (CHANGE IF YOU DONT WANT OVERWRITES!)
    
    if pairs == 'xy_bins':
        num_bins = 3 # size of xy grid space
        show_heatmap = True # plot a quick heatmap of how many samples are in each grid space?
    elif pairs == 'file':
        pairs_alg = 'pasta' #which algorithm should we pull pair data from (if doing so)?
        pairs_path = os.path.join(dest_path,pairs_alg)
    elif pairs == 'lag':
        comparisons_per_lag = 10
        lags = [1] + list(range(2, 251, 1))
        # lags = list(range(300, 305, 1))
        num_comparisons = len(lags) * comparisons_per_lag


    # extract the pose and point cloud data from its saved pickle files
    print('Loading data...')
    pose_file = open(os.path.join('experiments_big_obstacles/numpy/true_pose_big'), 'rb')
    true_pose = pickle.load(pose_file)
    pose_file.close()

    cloud_file = open(os.path.join('experiments_big_obstacles/numpy/point_clouds_big'), 'rb')
    point_cloud_data = pickle.load(cloud_file)
    cloud_file.close()
        
    # using a dict of dicts for all of the algorithms we test
    # initial value for each algorithm just indicates if we are going to run it or not

    if algs is None :
        algs = {'teaser': False,
                'icp' : False,
                'icp_plane' : False,
                'robust_icp' : False,
                'pasta' : True,
                'pasta_plus_icp': False,
                'fpfh' : False,
                'gen_icp' : False,
                'fgr' : False,
                'go_icp' : False
                }
    
    dimension = '2d'

    for alg,check in list(algs.items()):
        if check :
            algs[alg] = {'times' : [], 'rotation_errors' : [], 'translation_errors' : [], 'transformations' : []}
        else :
            del algs[alg]
    
    
    start_index = 0
    end_index = len(true_pose)
    seed = 0
    generator = np.random.default_rng(seed)

    if pairs == 'random' :
        num_comparisons = min(num_pairs,(end_index-start_index)//2) # how many poses do we want to use?

        # we will pull 2* number of desired comparisons scans from the data
        selected_indices = generator.choice(np.arange(start_index,end_index),2*num_comparisons,replace=False)
        
        # save tuples of those indices in a list
        pairs = [(selected_indices[2*i], selected_indices[2*i+1]) for i in range(num_comparisons)]
    
    elif pairs == 'sequential' :
        num_comparisons = min(num_pairs, (end_index-start_index)-2) # how many poses do we want to compare?
        start_window = generator.integers(start_index, end_index-num_comparisons) # pick a random starting point
        
        # save tuples of indices in sequence in a list
        pairs  = [(i, i+1) for i in range(start_window,start_window+num_comparisons)]
    
    elif pairs == 'xy_bins':
        bins, xdiv, ydiv = bin_data(true_pose, num_bins)
        num_comparisons = min(num_pairs, (end_index-start_index)//2)
        bin_indices = [(i,j) for i in range(num_bins) for j in range(num_bins)]

        if show_heatmap :
            heatmap_data = np.array([[len(bins[i][j]) for i in range(num_bins)] for j in range(num_bins)])
            fig, ax = plt.subplots()
            ax.imshow(heatmap_data)
            for i in range(num_bins):
                for j in range(num_bins):
                    text = ax.text(j, i, len(bins[i][j]),
                                ha="center", va="center", color="w")

            ax.set_xticks(np.arange(0,num_bins+1)-0.5, labels = ["{:.2f}".format(div) for div in xdiv])
            ax.set_yticks(np.arange(0,num_bins+1)-0.5, labels = ["{:.2f}".format(div) for div in ydiv])
            fig.tight_layout()
            ax.legend()
            plt.show()



        pairs = []
        for i in range(num_comparisons):
            # randomly select a pair of bins
            bin_1 = generator.choice(bin_indices)
            bin_2 = generator.choice(bin_indices)

            # randomly select poses from those bins
            #print(len(bins[bin_1[0]][bin_1[1]]))
            ind_1 = generator.choice(bins[bin_1[0]][bin_1[1]])
            ind_2 = generator.choice(bins[bin_2[0]][bin_2[1]])
            #print(type(ind_1))
            pairs.append((ind_1,ind_2))


    elif pairs == 'file' :
        pairs_file = open(os.path.join(pairs_path,'pairs'), 'rb')
        pairs = pickle.load(pairs_file)
        pairs_file.close()
        num_comparisons = len(pairs)

    elif pairs == 'pinned_first':
        num_comparisons = min(num_pairs,(end_index-start_index)-1)
        pairs = [(start_index,i) for i in range(num_comparisons)]

    elif pairs == 'lag':
        rng = np.random.default_rng()
        # rng = generator
        pairs = []
        for lag in lags:  # reject pair if diff in angle is > pi/2, make sure to  
            inter_count = 0
            while inter_count < comparisons_per_lag:
                i = rng.integers(start_index, end_index-lag, 1)[0]
                
                if -np.pi/2 < angle_projection(np.radians((true_pose[i+lag] - true_pose[i])[2])) < np.pi/2:
                    pairs = pairs + [(i, i+lag)]
                    inter_count += 1
                #pairs = pairs + [(i, i+lag) for i in rng.integers(start_index, end_index-lag, comparisons_per_lag)]
        print(pairs)
        # Pair debugging


    else :
        Exception('Invalid pair type specified, exiting...')


    # now iterate through the pairs!
    error_list = []
    counter = 0
    for ind_1,ind_2 in pairs:
        counter += 1
        if True:
            print('='*60)
            print('Running pose pair {} of {} ({:.2%})...'.format(counter,num_comparisons,float(counter)/float(num_comparisons)))
            print('Poses are {} and {}'.format(ind_1,ind_2))
            print('='*60)
        
        # pull data from two indices 
        pc1 = point_cloud_data[ind_1]
        num_pts1 = len(pc1)
        pose_1 = true_pose[ind_1]

        pc2 = point_cloud_data[ind_2]
        num_pts2 = len(pc2)
        pose_2 = true_pose[ind_2]

        # compute ground truth rotation/translation between them
        true_rotation = angle_projection(np.radians((true_pose[ind_2] - true_pose[ind_1])[2]))
        
        true_translation = (true_pose[ind_2] - true_pose[ind_1])[:2]

        # run the algorithms!

        trans_init = np.eye(4) #initial transformation guess for algs that accept one

        for alg in algs.keys():
            if algs[alg] == False: #skip the algorithms that we don't want to run
                continue
            
            if verbose:
                
                print('Running algorithm {} on poses {} and {}...'.format(alg,ind_1,ind_2))
            est_rotation, est_translation, runtime = use_algorithm(alg,pc1, pc2, dimension)

            #rotation_error = abs(np.degrees(angle_projection(est_rotation-true_rotation)))
            #translation_error = compare_translations(est_translation,true_translation,pose_1[2],pose_2[2])
            rotation_error, translation_error = compare_translations_rotations(est_rotation,true_rotation, est_translation, true_translation, pose_1[2],pose_2[2], alg)
            if alg == 'pasta' and rotation_error > 90:
                error_list.append((ind_1, ind_2))
            algs[alg]['times'].append(runtime)
            algs[alg]['rotation_errors'].append(rotation_error)
            algs[alg]['translation_errors'].append(translation_error)
            algs[alg]['transformations'].append([est_rotation,est_translation])

    # now, save all the results

    # make a directory if needed
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    
    # go through each algorithm, save its results in a folder
    for alg,results in algs.items():

        # do nothing if we didn't run the algorithm
        if results == False:
            continue

        # otherwise, pickle the data from the algorithm in its own folder
        alg_path = os.path.join(dest_path,alg)
        if not os.path.exists(alg_path):
            os.mkdir(alg_path)

        # first, save the list of pairs you used for reference
        pair_file = open(os.path.join(alg_path,'pairs'),'wb')
        pickle.dump(pairs,pair_file)
        pair_file.close()
        
        for result,data in results.items():
            file = open(os.path.join(alg_path,result),'wb')
            pickle.dump(data,file)
            file.close()
    
    print(error_list)

if __name__ == '__main__':
    main()