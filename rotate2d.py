import numpy as np
import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os

def main(algs=None,dir=None,pairs=None,plots=None):
    DATA_LOADED = False
    verbose = True
    if dir is None:
        dir = './' # directory for experiment and its data
    
    if pairs is None:
        pairs_type = '/xy_bins'
    else:
        pairs_type = pairs
    #source_path = os.path.join(dir,'results'+'_'+pairs_type)
    #source_path = os.path.join(dir,'/results_fixing_'+pairs_type)
    
    if algs is None:
        # dict of algorithms to pull data from
        algs = {'teaser': False,
                'icp' : False,
                'icp_plane' : False,
                'robust_icp' : False,
                'pasta' : True,
                'pasta_plus_icp': True,
                'iPasta' : True,
                'fpfh' : False,
                'gen_icp' : False,
                'fgr' : False,
                'go_icp' : False
                }


    if plots is None :
        # dict of plots to make
        plots = {'histograms' : True,
                'trajectory' : True,
                'scans' : True,
                'lag': True,
                'time' : False}
        plots['save'] = [key for key in plots.keys()]

    # plotting an nrows x ncols grid of lidar scans
    if plots['scans']:
        cloud_data_fname = os.path.join(dir, 'experiments_big_obstacles/numpy/point_clouds')
        with open(cloud_data_fname, 'rb') as f:
            cloud_data = pickle.load(f)       
        finres = []
        
        for slice_data in cloud_data:
            x = slice_data[:,0]
            y = slice_data[:,1]    
                    
            xy_points = np.column_stack((x,y))
            finres.append(xy_points)
            
        print(len(finres))
        print(finres[0].shape)
        
        with open('rotated_pt2d_big.pickle', 'wb') as pt3d_file:
            pickle.dump(finres, pt3d_file)

if __name__ == '__main__':
    main()
