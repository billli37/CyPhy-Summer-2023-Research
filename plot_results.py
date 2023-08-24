import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gs
import pickle
import os

matplotlib.rcParams['pdf.fonttype'] = 42 # changed for compatability errors with IEEE

def wrap_angles(data):
    return [np.hstack((d[d < 90],180-d[d>=90])) for d in data]

# plots is a dict:
# keys: 'histograms', 'trajectories','scans','time', 'save'
# boolean values for all except 'save'
# 'save' = a list of keys to the plots you want saved, e.g.
# plots['save'] = ['histograms','trajectories'] if you only want to save histograms and trajectory plots

def main(algs=None,dir=None,pairs=None,plots=None):
    DATA_LOADED = False
    verbose = True
    if dir is None:
        dir = './data/many_obstacles_2022_09_12' # directory for experiment and its data
    
    if pairs is None:
        pairs_type = 'xy_bins'
    else:
        pairs_type = pairs
    source_path = os.path.join(dir,'results'+'_'+pairs_type)
    
    if algs is None:
        # dict of algorithms to pull data from
        algs = {'teaser': False,
                'icp' : False,
                'icp_plane' : True,
                'robust_icp' : False,
                'pasta' : False,
                'pasta_plus_icp': False,
                'fpfh' : False,
                'gen_icp' : False,
                'fgr' : False,
                'go_icp' : False
                }


    if plots is None :
        # dict of plots to make
        plots = {'histograms' : False,
                'trajectory' : True,
                'scans' : True,
                'lag': True,
                'time' : False}
        plots['save'] = [key for key in plots.keys()]

    # plotting an nrows x ncols grid of lidar scans
    if plots['scans']:
        cloud_data_fname = os.path.join('experiments_big_obstacles/numpy/pt3d_big')
        with open(cloud_data_fname, 'rb') as f:
            cloud_data = pickle.load(f)

        nrows = 2
        ncols = 2
        num_pts = len(cloud_data)
        f2,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(12,10))
        inds = np.arange(0,num_pts,num_pts//(nrows*ncols))
        for index,ax in zip(inds,axs.ravel()):
            ax.scatter(cloud_data[index][:,0], cloud_data[index][:,1], zorder=10, s=9)
            ax.scatter(0, 0, zorder=10, color='yellow', edgecolor='black')
            ax.grid(zorder=0)
            ax.axis('square')
            ax.set_title("Scan {}".format(index+1))
        f2.tight_layout()
        plot_file = 'scans_big.png'
        if 'scans' in plots['save']:
            f2.savefig(os.path.join(plot_file))
        #plt.show()


    # plot the trajectory of the robot
    if plots['trajectory']:
        pose_data_fname = os.path.join('experiments_big_obstacles/numpy/true_pose_big')
        with open(pose_data_fname, 'rb') as f:
            true_pose = pickle.load(f)

        # how many data points are there for each algorithm's results?
        num_pts = np.shape(true_pose)[0]
        f4,ax4 = plt.subplots()
        ax4.plot(-true_pose[:,0],true_pose[:,1],lw=4)
        arr_len = 0.2
        for i in range(num_pts):
            ax4.arrow(-true_pose[i,0],true_pose[i,1],-arr_len*np.cos(true_pose[i,2]),-arr_len*np.sin(true_pose[i,2]), width=0.01, color='red')
            ax4.arrow(-true_pose[i,0],true_pose[i,1],-arr_len*np.sin(true_pose[i,2]),arr_len*np.cos(true_pose[i,2]), width=0.01, color='green')
        ax4.set_title('Trajectory')
        ax4.set_xlabel('x (m)')
        ax4.set_ylabel('y (m)')
        ax4.grid()
        ax4.axis('square')
        ax4.set_xlim(-3,3)
        ax4.set_ylim(-3,3)
        
        if 'trajectory' in plots['save']:
            plot_file = 'frame_trajectory.png'
            f4.savefig(plot_file)

        f5,axs = plt.subplots(nrows=3,ncols=1)
        ylabels = ['x', 'y', 'angle']
        for i,ax in enumerate(axs.ravel()):
            ax.plot(true_pose[:,i],linewidth=3)
            ax.set_xlabel('t')
            ax.set_ylabel(ylabels[i])
            ax.grid()

        if 'trajectory' in plots['save']:
            plot_file = 'time_trajectory.png'
            f5.savefig(plot_file)

    # plot the average performance of algorithms by pair comparison lag
    if plots['lag']:
        # For now I just copy the params from run_algorithms.py // Matteo
        comparisons_per_lag = 10
        lags = [1] + list(range(2, 251, 1))
        # lags = list(range(300, 305, 1))

        angle_wrapping = False # wrap angular errors around 90?
        save_lag_plot = True # save the scan?
        save_scan = True

        outlier_limits = dict({'times' : 0.048, 'rotation_errors': 90, 'translation_errors': 3.5})

        # list of algorithms to pull data from
        '''
        algs = {'teaser': True,
                'icp' : True,
                'icp_plane' : True,
                'robust_icp' : True,
                'pasta' : True,
                'fpfh' : True,
                'gen_icp' : True,
                'fgr' : True,
                'go_icp' : True
                }
        '''

        num_trimmed = dict.fromkeys(outlier_limits.keys(),None)
        
        if not DATA_LOADED:
            # typeset your label strings for plotting here
            labels  = np.array(['TEASER++', 'ICP', 'P2Plane-ICP','R-ICP','PASTA', 'PASTA + ICP', 'FPFH','Gen-ICP','FGR','Go-ICP'])
            mask = np.array(list(algs.values())) # mask the full list by which ones we'll be loading
            # print(labels, mask)
            # exit()
            labels = list(labels[mask])

            # pull the data for all algorithms set to true
            for alg,check in list(algs.items()):
                if algs[alg]: # if we are going to plotting/using this alg's data
                    #labels.append(alg)
                    # setup its landing space for its part of the dictionary
                    algs[alg] = {'times' : [], 'rotation_errors' : [], 'translation_errors' : [], 'transformations' : [], 'rototranslation_errors' : []}

                    alg_path = os.path.join(source_path,alg)

                    # go through all the result data and save it in the dictionary
                    for result in algs[alg].keys():
                        with open('results_pinned_first/icp_plane/times','rb') as f:
                            algs[alg][result] = pickle.load(f)
                
                else : 
                    del algs[alg]

            DATA_LOADED = True
        
        # how many data points are there for each algorithm's results? (for histogram normalization)
        num_pts = len(next(iter(algs.values()))['times'])

        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        f = plt.figure(figsize=(8,10))
        xdata = lags

        # result = 'rotation_errors'
        # means = np.array([[np.mean(alg[result][i:(i+1)*comparisons_per_lag]) for alg in algs.values()] for i in range(len(lags))])
        # # means = np.array([[np.mean(alg[result][lag:lag+comparisons_per_lag]) for alg in algs.values()] for lag in lags])
        # # stds = np.array([[np.std(alg[result][lag:lag+comparisons_per_lag]) for alg in algs.values()] for lag in lags])
        # ax = f.add_subplot((211))
        # ax.plot(xdata, means, label=labels, linewidth=1)
        # ax.set_title('Rotation errors')
        # ax.set_xlabel('Lag')
        # ax.set_ylabel('Error (deg)')
        # ax.grid()
        # ax.legend()

        # result = 'translation_errors'
        # means = np.array([[np.mean(alg[result][i:(i+1)*comparisons_per_lag]) for alg in algs.values()] for i in range(len(lags))])
        # # means = np.array([[np.mean(alg[result][lag:lag+comparisons_per_lag]) for alg in algs.values()] for lag in lags])
        # # stds = np.array([[np.std(alg[result][lag:lag+comparisons_per_lag]) for alg in algs.values()] for lag in lags])
        # ax = f.add_subplot((212))
        # ax.plot(xdata, means, label=labels, linewidth=1)
        # ax.set_title('Translation errors')
        # ax.set_xlabel('Lag')
        # ax.set_ylabel('Error (m)')
        # ax.grid()
        # ax.legend()

        result = 'rototranslation_errors'
        means = np.array([[np.mean(alg[result][i:(i+1)*comparisons_per_lag]) for alg in algs.values()] for i in range(len(lags))])
        ax = f.add_subplot((211))
        ax.plot(xdata, means, label=labels, linewidth=1)
        ax.set_title('Rototranslation errors')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Error distance')
        ax.grid()
        ax.legend()

        f.tight_layout()

        if 'histograms' in plots['save'] :
            if angle_wrapping :
                plot_file = 'lags_angle_wrapped_big.png'
            else :
                plot_file = 'lags_big.png'
            f.savefig(os.path.join(plot_file))


    # plot histograms
    if plots['histograms']:
        angle_wrapping = False # wrap angular errors around 90?
        save_hist = True # save the scan?
        save_scan = True

        outlier_limits = dict({'times' : 0.048, 'rotation_errors': 180, 'translation_errors': 3.5})

        # list of algorithms to pull data from
        '''
        algs = {'teaser': True,
                'icp' : True,
                'icp_plane' : True,
                'robust_icp' : True,
                'pasta' : True,
                'fpfh' : True,
                'gen_icp' : True,
                'fgr' : True,
                'go_icp' : True
                }
        '''

        num_trimmed = dict.fromkeys(outlier_limits.keys(),None)
        
        if not DATA_LOADED:
            # typeset your label strings for plotting here
            labels  = np.array(['TEASER++', 'ICP', 'P2Plane-ICP','R-ICP','PASTA', 'PASTA + ICP', 'FPFH','Gen-ICP','FGR','Go-ICP'])
            mask = np.array(list(algs.values())) # mask the full list by which ones we'll be loading
            # print(labels, mask)
            labels = list(labels[mask])
            
            # pull the data for all algorithms set to true
            for alg,check in list(algs.items()):
                if algs[alg]: # if we are going to plotting/using this alg's data
                    #labels.append(alg)
                    # setup its landing space for its part of the dictionary
                    algs[alg] = {'times' : [], 'rotation_errors' : [], 'translation_errors' : [], 'transformations' : [], 'rototranslation_errors' : []}
                    alg_path = os.path.join(source_path,alg)

                    # go through all the result data and save it in the dictionary
                    for result in algs[alg].keys():
                        with open(os.path.join(alg_path,result),'rb') as f:
                            algs[alg][result] = pickle.load(f)
                
                else : 
                    del algs[alg]
            
            DATA_LOADED = True
        
        # how many data points are there for each algorithm's results? (for histogram normalization)
        num_pts = len(next(iter(algs.values()))['times'])

        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # make an average runtimes histogram
        result = 'times'
        f = plt.figure(figsize=(8,10))
        ax = f.add_subplot((311))

        outlier_limit = outlier_limits[result]


        # compile all of the data in a single list to plot
        #xdata = [np.array(alg[result])[(np.array(alg[result]) <= outlier_limit)] for alg in algs.values()]
        xdata = [np.array(alg[result])[(np.array(alg[result]) <= outlier_limit) & (np.array(alg[result]) >= 0)] for alg in algs.values()]
        num_trimmed[result] = [np.sum(np.array(alg[result]) <= outlier_limit) for alg in algs.values()]
        means = [np.mean(algs[alg][result]) for alg in algs.keys()]


        num_bins = 10

        ax.hist(xdata,bins=num_bins,histtype='bar',label=labels)

        # mean vline plotting
        min_ylim,max_ylim = ax.get_ylim()
        ht = max_ylim*0.9
        decrease = max_ylim*0.1
        hshift = 0.00025
        count  = 0
        for mean in means:
            if mean <= outlier_limit:
                ax.axvline(mean,color=cycle[count],linestyle='dashed',linewidth=1.5)
                ax.text(mean + hshift,ht-decrease*count,'{}: {:.3f}'.format(labels[count],mean),size='large')
            else :
                ax.text(outlier_limit,ht-decrease*count,'{}: {:.3f}'.format(labels[count],mean),size='large',horizontalalignment='right')
            count += 1

        ax.set_xlim(0,1.001*outlier_limit)
        ax.set_title('Running times')
        #ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (counts)')

        #plt.show()

        # # make an angle error histogram
        # result = 'rotation_errors'
        # ax2 = f.add_subplot((312))

        # outlier_limit = outlier_limits[result]

        # xdata = [np.array(alg[result]) for alg in algs.values()]
        # if angle_wrapping: 
        #     xdata = wrap_angles(xdata)
        # means = [np.mean(data) for data in xdata]
        # num_trimmed[result] = [np.sum(data <= outlier_limit) for data in xdata]
        # xdata = [data[data <= outlier_limit] for data in xdata]
        

        # num_bins = 10

        # ax2.hist(xdata,bins=num_bins,histtype='bar',label=labels)

        # # mean vline plotting
        # min_ylim,max_ylim = ax2.get_ylim()
        # ht = max_ylim*0.9
        # decrease = max_ylim*0.1
        # hshift = 0.1
        # count  = 0
        # for mean in means:
        #     if mean <= outlier_limit:
        #         ax2.axvline(mean,color=cycle[count],linestyle='dashed',linewidth=1.5)
        #         ax2.text(mean + hshift,ht-decrease*count,'{}: {:.2f}'.format(labels[count],mean),size='large')
        #     else :
        #         ax2.text(outlier_limit,ht-decrease*count,'{}: {:.3f}'.format(labels[count],mean),size='large',ha='right')
        #     count += 1

        # ax2.set_title('Rotation errors (degrees)')
        # #ax2.set_xlim(0,35)
        # ax2.set_xlabel('Error (degrees)')
        # ax2.set_ylabel('Frequency (counts)')
        # #plt.show()


        # make a rototranslation error histogram
        result = 'rototranslation_errors'
        ax3 = f.add_subplot((313))

        # outlier_limit = outlier_limits[result]

        xdata = [np.array(alg[result]) for alg in algs.values()]
        # if angle_wrapping: 
        #     xdata = wrap_angles(xdata)
        means = [np.mean(data) for data in xdata]
        num_trimmed[result] = [np.sum(data) for data in xdata]
        xdata = [data[data] for data in xdata]
        

        num_bins = 10

        ax3.hist(xdata,bins=num_bins,histtype='bar',label=labels)

        # mean vline plotting
        min_ylim,max_ylim = ax3.get_ylim()
        ht = max_ylim*0.9
        decrease = max_ylim*0.1
        hshift = 0.001
        count  = 0
        for mean in means:
            ax3.text(outlier_limit,ht-decrease*count,'{}: {:.3f}'.format(labels[count],mean),size='large',ha='right')
            count += 1
        ax3.set_xlim(0,outlier_limit)
        ax3.set_title('Rototranslation error distances')
        #ax3.legend()
        ax3.set_xlabel('Error distance')
        ax3.set_ylabel('Frequency (counts)')
        handles,labels = ax3.get_legend_handles_labels()



        # # make a translation error histogram
        # result = 'translation_errors'
        # ax3 = f.add_subplot((313))

        # outlier_limit = outlier_limits[result]

        # xdata = [np.array(alg[result])[np.array(alg[result]) <= outlier_limit] for alg in algs.values()]
        # num_trimmed[result] = [np.sum(np.array(alg[result]) <= outlier_limit) for alg in algs.values()]
        # means = [np.mean(algs[alg][result]) for alg in algs.keys()]

        # num_bins = 10
        # #labels = list(algs.keys())

        # ax3.hist(xdata,bins=num_bins,histtype='bar',label=labels)

        # # mean vline plotting
        # min_ylim,max_ylim = ax2.get_ylim()
        # ht = max_ylim*0.9
        # decrease = max_ylim*0.1
        # hshift = 0.001
        # count  = 0
        # for mean in means:
        #     if mean <= outlier_limit:
        #         ax3.axvline(mean,color=cycle[count],linestyle='dashed',linewidth=1.5)
        #         ax3.text(mean + hshift,ht-decrease*count,'{}: {:.3f}'.format(labels[count],mean),size='large')
        #     else :
        #         ax3.text(outlier_limit,ht-decrease*count,'{}: {:.3f}'.format(labels[count],mean),size='large',ha='right')
        #     count += 1
        # ax3.set_xlim(0,outlier_limit)
        # ax3.set_title('Translation errors (m)')
        # #ax3.legend()
        # ax3.set_xlabel('Error (m)')
        # ax3.set_ylabel('Frequency (counts)')

        # handles,labels = ax3.get_legend_handles_labels()
        # ax2.legend(loc='center right')
        f.tight_layout()

        if 'histograms' in plots['save'] :
            if angle_wrapping :
                plot_file = 'histograms_angle_wrapped_big.png'
            else :
                plot_file = 'histograms_big.png'
            f.savefig(os.path.join(plot_file))


    if plots['time']:
        if not plots['histograms']:
            labels  = np.array(['TEASER++', 'ICP', 'P2Plane-ICP','R-ICP','PASTA','FPFH','Gen-ICP','FGR','Go-ICP'])
            mask = np.array(list(algs.values())) # mask the full list by which ones we'll be loading
            labels = list(labels[mask])

        with open('results_pinned_first/icp_plane/times','rb') as f:
            time = pickle.load(f)

        # time *= 1E-9
        time = time[1:]
        time.append('')

        f5 = plt.figure(figsize=(12,10))
        ax5_rt = f5.add_subplot(211)
        # ax5_t = f5.add_subplot(212)

        xdata = [np.array(alg[result]) for alg in algs.values()]

        for alg in algs.values():
            ax5_rt.plot(time,np.array(alg['rototranslation_errors']),linewidth=2.0)
            # ax5_t.plot(time,np.array(alg['translation_errors']),linewidth=2.0)
            

        ax5_rt.legend(labels)
        ax5_rt.set_xlabel('Time (s)')
        ax5_rt.set_ylabel('Rotation error (deg)')
        # ax5_t.set_xlabel('Time (s)')
        # ax5_t.set_ylabel('Translation error (m)')

        ax5_rt.grid()
        # ax5_t.grid()

        if 'time' in plots['save']:
            plot_file = 'time_errors_big.png'
            f5.savefig(os.path.join(plot_file))

    plt.show()

if __name__ == '__main__':
    main()