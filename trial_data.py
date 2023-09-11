import scipy.io as sio
import numpy as np
from os import path
from scipy import stats
from scipy.signal import decimate

class trial_data:
    def __init__(self, base_path, file_name):
        if base_path[-1] != '/':
            base_path = base_path + '/' 
        self.file_name = file_name[:-4]
        if not path.exists( base_path + file_name ):
            raise Exception( 'Can''t find file:' + file_name )
        read_data = sio.loadmat(base_path + file_name)
        self.trials = read_data['trial_data']
        self.bin_size = self.trials[0][0]['bin_size'][0][0]
        self.unit_names_M1 = ['elec'+str(each[0])+'_'+str(each[1]) for each in self.trials[0][0]['M1_unit_guide']]
        self.unit_names_PMd = ['elec'+str(each[0])+'_'+str(each[1]) for each in self.trials[0][0]['PMd_unit_guide']]
        self.perturbation = self.trials[0][0]['perturbation'][0]
        self.perturbation_info = self.trials[0][0]['perturbation_info'][0]
        #---- for the convinience of further processing, just extracted numbers from arrays for the timing info ----#
        for each in self.trials[0]:
            each['idx_trial_start'] = each['idx_trial_start'][0][0]
            each['idx_target_on'] = each['idx_target_on'][0][0]
            each['idx_go_cue'] = each['idx_go_cue'][0][0]
            each['idx_movement_on'] = each['idx_movement_on'][0][0]
            each['idx_peak_speed'] = each['idx_peak_speed'][0][0]
            each['idx_trial_end'] = each['idx_trial_end'][0][0]
        
    def get_spikes(self, trial_result, epoch, array):
        """
        This function gives M1_spikes or PMd spikes in trials without any temporal alignement
        The strings to specify which array are:
            M1_spikes
            PMd_spikes
        """
        if (array == 'M1')|(array == 'm1'):
            array_str = 'M1_spikes'
        elif (array == 'PMD')|(array == 'pmd')|(array == 'PMd'):
            array_str = 'PMd_spikes'
        spikes = []
        for i, each in enumerate(self.trials[0]):
            if each['result'] == trial_result:
                if each['epoch'] == epoch:
                    temp = each[array_str]
                    spikes.append(temp)
        return spikes
    
    def get_cursor(self, trial_result, epoch, output):
        """
        This function gives cursor data in trials without any temporal alignement
        The strings for the type of outputs are:
            'pos'
            'vel'
            'acc'
        """
        curs_p, curs_v, curs_a = [], [], []
        for i, each in enumerate(self.trials[0]):
            if each['result'] == trial_result:
                if each['epoch'] == epoch:
                    curs_p.append(each['pos'])
                    curs_v.append(each['vel'])
                    curs_a.append(each['acc'])
        if output == 'pos':
            return curs_p
        elif output == 'vel':
            return curs_v
        elif output == 'acc':
            return curs_a
    
    def get_trial_info(self, trial_result, epoch, info_type):
        """
        info_type includes ID, target direction and timing information,
        which can be specified by 'ID', 'target_dir' and 'timing'
        """
        if info_type == 'trial_id':
            return [each['trial_id'][0][0] for each in self.trials[0] 
                    if (each['result'] == trial_result)&(each['epoch'] == epoch)]
        if info_type == 'target_direction': 
            return [int(np.round(np.rad2deg(each['target_direction'][0][0]))) for each in self.trials[0] 
                    if (each['result'] == trial_result)&(each['epoch'] == epoch)]
        if info_type == 'timing':
            timing = {}
            timing['idx_trial_start'] = [each['idx_trial_start'] for each in self.trials[0]
                                   if (each['result'] == trial_result)&(each['epoch'] == epoch)]
            timing['idx_target_on'] = [each['idx_target_on'] for each in self.trials[0]
                                   if (each['result'] == trial_result)&(each['epoch'] == epoch)]
            timing['idx_go_cue'] = [each['idx_go_cue'] for each in self.trials[0]
                                   if (each['result'] == trial_result)&(each['epoch'] == epoch)]
            timing['idx_movement_on'] = [each['idx_movement_on'] for each in self.trials[0]
                                   if (each['result'] == trial_result)&(each['epoch'] == epoch)]
            timing['idx_peak_speed'] = [each['idx_peak_speed'] for each in self.trials[0]
                                   if (each['result'] == trial_result)&(each['epoch'] == epoch)]
            timing['idx_trial_end'] = [each['idx_trial_end'] for each in self.trials[0]
                                   if (each['result'] == trial_result)&(each['epoch'] == epoch)]
            timing['t_trial_start'] = [np.round(self.bin_size*each, 3) for each in timing['idx_trial_start']]
            timing['t_target_on'] = [np.round(self.bin_size*each, 3) for each in timing['idx_target_on']]
            timing['t_go_cue'] = [np.round(self.bin_size*each, 3) for each in timing['idx_go_cue']]
            timing['t_movement_on'] = [np.round(self.bin_size*each, 3) for each in timing['idx_movement_on']]
            timing['t_peak_speed'] = [np.round(self.bin_size*each, 3) for each in timing['idx_peak_speed']]
            timing['t_trial_end'] = [np.round(self.bin_size*each, 3) for each in timing['idx_trial_end']]
            return timing
        
    def update_bin_data(self, new_bin_size):
        if ((new_bin_size*100)%(self.bin_size*100) > 0)|(new_bin_size<self.bin_size):
            print('The new bin size is not valid')
            return 0
        K = int(new_bin_size/self.bin_size)
        self.bin_size = new_bin_size
        #---- compute the spike counts with new bin size ----#
        for i, each in enumerate(self.trials[0]):
            L = int(np.floor(each['M1_spikes'].shape[0]/K))
            M1_spike_counts_new, PMd_spike_counts_new = [], []
            for j in range(L):
                temp = np.sum(each['M1_spikes'][j*K:j*K+K, :], axis = 0)
                M1_spike_counts_new.append(temp)
                temp = np.sum(each['PMd_spikes'][j*K:j*K+K, :], axis = 0)
                PMd_spike_counts_new.append(temp)
            M1_spike_counts_new, PMd_spike_counts_new = np.array(M1_spike_counts_new), np.array(PMd_spike_counts_new)
            #---- use decimate to resample the cursor positions and so on ----#
            pos_new = decimate(each['pos'], K, axis = 0)
            vel_new = decimate(each['vel'], K, axis = 0)
            acc_new = decimate(each['acc'], K, axis = 0)
            force_new = decimate(each['force'], K, axis = 0)
            #---- update the fields in the data structure ----#
            T = min(pos_new.shape[0], M1_spike_counts_new.shape[0])
            each['M1_spikes'] = M1_spike_counts_new[:T, :]
            each['PMd_spikes'] = PMd_spike_counts_new[:T, :]
            each['pos'] = pos_new[:T, :]
            each['vel'] = vel_new[:T, :]
            each['acc'] = acc_new[:T, :]
            each['force'] = force_new[:T, :]
            #---- update the indices for go cue and so on ----#
            try:
                each['idx_trial_start'] = int(np.floor(each['idx_trial_start']/K))
            except Exception:
                each['idx_trial_start'] = np.NaN 
            try:
                each['idx_target_on'] = int(np.floor(each['idx_target_on']/K))
            except Exception:
                each['idx_target_on'] = np.NaN
            try:
                each['idx_go_cue'] = int(np.floor(each['idx_go_cue']/K))
            except Exception:
                each['idx_go_cue'] = np.NaN
            try:
                each['idx_movement_on'] = int(np.floor(each['idx_movement_on']/K))
            except Exception:
                each['idx_movement_on'] = np.NaN
            try:
                each['idx_peak_speed'] = int(np.floor(each['idx_peak_speed']/K))
            except Exception:
                each['idx_peak_speed'] = np.NaN
            try:
                each['idx_trial_end'] = int(np.floor(each['idx_trial_end']/K))
            except Exception:
                each['idx_trial_end'] = np.NaN    

    def smooth_binned_spikes(self, bin_size, kernel_type, kernel_SD):
        kernel_hl = 3 * int( kernel_SD / bin_size )
        normalDistribution = stats.norm(0, kernel_SD)
        x = np.arange(-kernel_hl*bin_size, (kernel_hl+1)*bin_size, bin_size)
        kernel = normalDistribution.pdf(x)
        if kernel_type == 'gaussian':
            pass
        elif kernel_type == 'half_gaussian':
            for i in range(0, int(kernel_hl)):
                kernel[i] = 0
        #---- do the smoothing on a single trial basis ---#
        for each in self.trials[0]:
            #---- M1 array ----#
            smoothed = []
            binned_spikes = each['M1_spikes'].T.tolist()
            n_sample = np.size(binned_spikes[0])
            nm = np.convolve(kernel, np.ones((n_sample))).T[int(kernel_hl):n_sample + int(kernel_hl)] 
            for neuron in binned_spikes:
                temp1 = np.convolve(kernel, neuron)
                temp2 = temp1[int(kernel_hl):n_sample + int(kernel_hl)]/nm
                smoothed.append(temp2)
            each['M1_spikes'] = np.asarray(smoothed).T
            #---- PMd array ----#
            smoothed = []
            binned_spikes = each['PMd_spikes'].T.tolist()
            n_sample = np.size(binned_spikes[0])
            nm = np.convolve(kernel, np.ones((n_sample))).T[int(kernel_hl):n_sample + int(kernel_hl)] 
            for neuron in binned_spikes:
                temp1 = np.convolve(kernel, neuron)
                temp2 = temp1[int(kernel_hl):n_sample + int(kernel_hl)]/nm
                smoothed.append(temp2)
            each['PMd_spikes'] = np.asarray(smoothed).T


    def get_aligned_trials(self, data_type, trial_result, epoch, start_event, time_before_start, end_event, time_after_end):
        """
        A function to get trials aligned to a specific event
        The strings for data_type are as follows:
            'pos'
            'vel'
            'acc'
            'M1_spikes'
            'PMd_spikes'
        The strings for event names are as follows:
            'trial_start'
            'target_on'
            'go_cue'
            'movement_on'
            'peak_speed'
            'trial_end'
        """
        idx_before = int(np.floor(time_before_start/self.bin_size))
        idx_after = int(np.floor(time_after_end/self.bin_size))
        aligned_trials = []
        trial_id, trial_target_direction = [], []
        for i, each in enumerate(self.trials[0]):
            if (each['result'] == trial_result)&(each['epoch'] == epoch):
                start_event_idx = each['idx_'+start_event]
                end_event_idx = each['idx_'+end_event]
                idx_start = start_event_idx - idx_before
                idx_end = end_event_idx + idx_after
                if idx_start<0:
                    print('time_before_start is not reasonable for trial %d, please select a different one'%(i))
                if idx_end>each['pos'].shape[0]:
                    print('time_after_end is not reasonable for trial %d, please select a different one'%(i))
                temp = each[data_type][idx_start:idx_end+1, :]
                aligned_trials.append(temp)
                trial_id.append(each['trial_id'][0][0])
                trial_target_direction.append(int(np.round(np.rad2deg(each['target_direction'][0][0]))))
        return aligned_trials, trial_id, trial_target_direction

            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

