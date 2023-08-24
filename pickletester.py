import numpy as np
import pandas as pd
import pickle as pkl
import csv


with open('pt3d_big_TEST', 'rb') as pickle_file:
    cloud_data = pkl.load(pickle_file)
    # df = pd.DataFrame(cloud_data)
    # df.to_csv('ptc.csv')
    
with open('pt3_big_TEST.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(cloud_data)
    


# df = pd.DataFrame(cloud_data)
# df.to_csv('point_clouds.csv', index)