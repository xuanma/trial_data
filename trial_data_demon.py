from trial_data import trial_data

"""
'BL' --- baseline
'AD' --- adaptation
'WO' --- washout
"""

#---- specifying the path and the file name ----#
base_path = 'G:/Chewie_left_M1_PMD/'
file_name = 'Chewie_CO_FF_2016-09-19.mat'
#---- create a trial_data instance ----#
td = trial_data(base_path, file_name)
#---- To get M1 data in baseline experiments without trial alignment for successful trials only ----#
#---- Note, 'R' is for 'rewarded', during succssful trials the monkey will be rewarded ----#
BL_M1 = td.get_spikes('R', 'BL', 'M1')
#---- To get PMd data in baseline experiments without trial alignment for successful trials only----#
BL_PMd = td.get_spikes('R', 'BL', 'PMd')
#---- Likewise, to get PMd data in adaptation experiments without trial alignment for successful trials only----#
AD_PMd = td.get_spikes('R', 'AD', 'PMd')
#---- To get cursor position data in baseline experiments without trial alignment for successful trials only----#
BL_curs_p = td.get_cursor('R', 'BL', 'pos')
#---- Since there's learning and adaptation in these experiments, the order of the trials matters in analyses,
#---- through trial_id one can identify the early trials and late trials ----#
BL_trial_id = td.get_trial_info('R', 'BL', 'trial_id')
#---- Get the target directions in degrees for baseline, successful trials ----#
BL_trial_target_dir = td.get_trial_info('R', 'BL', 'target_direction')
#---- Get the timing information for baseline, successful trials ----#
BL_timing = td.get_trial_info('R', 'BL', 'timing') # BL_timing is a dictionary, see what's inside

#---- The original data were binned in 10 ms (0.01 s) bins, if a larger bin size is required, use the function below to bin the data ----#
new_bin_size = 0.05 # 50 ms bins
td.update_bin_data(new_bin_size)
#---- Gaussian smooth with kernel SD = smooth_size
smooth_size = 0.1
td.smooth_binned_spikes(new_bin_size, 'gaussian', smooth_size)
#---- Now the neural signals (both M1 and PMd) are re-binned with larger bin size and smoothed, plot some examples to see what happened ----#
BL_M1_ = td.get_spikes('R', 'BL', 'M1')
BL_PMD_ = td.get_spikes('R', 'BL', 'PMd')

#---- if temporal alignment of the trials is needed, please use the function below ----#
aligned_trials_M1, trial_id, trial_target_direction = td.get_aligned_trials('M1_spikes', 'R', 'BL', 'go_cue', 1, 'go_cue', 1)
aligned_trials_PMd, _, _ = td.get_aligned_trials('PMd_spikes', 'R', 'BL', 'go_cue', 1, 'go_cue', 1)
aligned_trials_cursor_pos, _, _ = td.get_aligned_trials('pos', 'R', 'BL', 'go_cue', 1, 'go_cue', 1)
#---- trial_id and trial_target_direction are also returned by calling this function



















