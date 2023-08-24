import pickle
import numpy as np
from scipy.spatial.transform import Rotation

# Load the pickled file containing 3D coordinates
with open('experiments_big_obstacles/pt3d_big.pickle', 'rb') as file:
    finres = pickle.load(file)
    
# Create rotation object for desired rotation (5 in x axis)
rotation = Rotation.from_euler('x', 5, degrees=True)

#Apply rotation to each point in dataset
rotated_data = []
for slice_data in finres:
    rotated_points = rotation.apply(slice_data)
    rotated_data.append(rotated_points)

# Save the rotated data to a new pickeled file
with open('rotated_pt3_big.pickle', 'wb') as file:
    pickle.dump(rotated_data, file)