import numpy as np
import pickle
import os

def depth_to_cartesian(data, angle_range=(0,360)):
    n = data.shape[0]
    m = data.shape[1]

    min_angle_rad = angle_range[0]*np.pi/180
    max_angle_rad = angle_range[1]*np.pi/180
    angles = min_angle_rad + np.arange(m)/m * (max_angle_rad - min_angle_rad)
    rays = np.stack((np.sin(-angles), np.cos(-angles)), axis=1)

    cloud_data = []
    for i in range(n):
        pose_cloud = rays * data[i][:,None]
        pose_cloud = pose_cloud[data[i] != np.inf, :]
        cloud_data.append(pose_cloud)

    return cloud_data

def extract_move_data(dir, angle_range=(0,360), subsample_angle=False):
    true_pose = np.genfromtxt(dir + "/optitrack values.csv", delimiter=',', missing_values="inf", filling_values=np.inf, skip_header=1)[:,1:]
    depths = np.genfromtxt(dir + "/point_cloud.csv", delimiter=',', missing_values="inf", filling_values=np.inf, skip_header=1)[:,1:]
    if subsample_angle:
        indices = np.arange(0,720,2)
        depths = depths[:,indices]
    cloud_data = depth_to_cartesian(depths, angle_range)

    return true_pose, cloud_data, depths

def main(dirs = None):
    # Getting the data
    if dirs == None:
        dirs = ["./experiments"] # put the csv file directories name here
    
    dest = "numpy"
    num_files = len(dirs)
    if num_files == 1:
        path = os.path.join(dirs[0],dest)
    else :
        dir = "./coverage_obstacles_2022_09_06_merged"
        if not os.path.exists(dir):
            os.mkdir(dir)
        path = os.path.join(dir,dest)
    
    if not os.path.exists(path):
        os.mkdir(path)
    subsample_angle = False # do you want every OTHER entry in angle's depths?

    # Non compensated angle
    angle_range = (0,360)
    true_pose = [None for dir in dirs]
    global_cloud_data = []
    depths = [None for dir in dirs]
    for i in range(num_files) :
        true_pose[i], cloud_data, depths[i] = extract_move_data(dirs[i], angle_range, subsample_angle)
        global_cloud_data += cloud_data # appending the list of point clouds

    # stacking the numpy arrays of poses and depths
    true_pose = np.concatenate(true_pose,axis=0)
    depths = np.concatenate(depths,axis=0)
    print("skipping time cause I did not record it")
    """
    times = np.genfromtxt(dirs[i]+'/time.csv',delimiter=',',skip_header=1)[:,1]
    times -= times[0] # set start time to zero
    """
    print('Extracted and saved {} poses'.format(np.shape(true_pose)[0]))

    """
    with open(os.path.join(path,'time'),'wb') as f:
        pickle.dump(times,f)
    """
    # pickle the output so that we can process it later without re-extracting
    pose_file = open(os.path.join(path,'true_pose'),'wb')
    pickle.dump(true_pose,pose_file)
    pose_file.close()

    cloud_file = open(os.path.join(path,'point_clouds'),'wb')
    pickle.dump(global_cloud_data,cloud_file)
    cloud_file.close()

    depths_file = open(os.path.join(path,'depths'),'wb')
    pickle.dump(depths,depths_file)
    depths_file.close()

if __name__ == "__main__" :
    main()