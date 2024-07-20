import numpy as np
import ppg_fusion_functions as helperfuncs
import scipy

# plotting
# import plotly.io as pio
# import plotly.graph_objects as go
# pio.renderers.default = 'browser'

sr = 128 # sampling rate of PPG signals
win_len = 30 # seconds to comput HR over
templ_size = 500 # how many waves to consider at most for the template
delta = 0.01 # should be float marginally greater than 0

# to improve motion artifact removal, ppg AC components can be clipped above the maximum possible PPG AC component
# this needs to be configured based on the ppg data used, this is deactivated by default. For the example data,
# an amplitude of 0.02 works well
use_clipping = False
clip_amplitude = 0.02

# ppg data array where every row represents one input trace
ppgdata = np.load("example_data/ppg.npy", allow_pickle=True)
# if the data still needs compensation for PTT offsets, you may use helperfuncs.get_av_offset(ppgpeaks1, ppgpeaks2, sr)

# ecg r peaks are optional, only used for ground truth. May be set to None if not available
ecg_rpks = np.load("example_data/ecg_r_peaks.npy", allow_pickle=True)

#%%
ppgdatafilt = [helperfuncs.filter_ppg(d, sr, clip=use_clipping, clip_ampl=clip_amplitude) for d in ppgdata]
ppgpeaks = [helperfuncs.all_ppg_peaks(d, sr) for d in ppgdatafilt]
ppgpeaks_cleaned = [helperfuncs.clean_pklst(pks, d, sr) for pks, d in zip(ppgpeaks, ppgdatafilt)]

# compute HRs on single PPG traces and ECG R peaks, if available
eval_locs = list(range(win_len*sr//2, len(ppgdata[0]), win_len*sr)) # indexes at which to compute HRs
ppghrs = [helperfuncs.hr_at_loc(pks, eval_locs, sr, win_len=win_len, min_rrs=10) for pks in ppgpeaks_cleaned]

# build PPG template, this is the only step that requires reviewing when attempting an online approach
wavetmpls = [helperfuncs.wave_template() for i in range(len(ppgdatafilt))]  # one for each body location
for wavetmpl, ppgpks, ppgfilt in zip(wavetmpls, ppgpeaks_cleaned, ppgdatafilt):
    for pk, pknp1 in zip(ppgpks, ppgpks[2::]): # wave delimiters are peaks separated by 2 IBI
        ret = wavetmpl.add_pulse(ppgfilt[pk:pknp1]) # add waves to template
    target_len = min(templ_size, wavetmpl.len//4) # rm at least 3/4 to get good quality template
    while wavetmpl.len > target_len:
        wavetmpl.rm_worst_signal(batch = (wavetmpl.len-target_len)//10+1) # remove multiples at a time to save time
    # wavetmpl.plot() # to plot wavetemplates, uncomment function and includes in headerfile

# calc scores and weights
ppgpks_scores = []
for wavetmpl, ppgpks, ppgfilt in zip(wavetmpls, ppgpeaks_cleaned, ppgdatafilt):
    peakwaves = [ppgfilt[pk:pknp1] for pk, pknp1 in zip(ppgpks, ppgpks[2:])]
    ppgpks_scores.append(wavetmpl.get_template_corr(peakwaves))

weights_res = [np.interp(list(range(0, len(sign))), xlocs[1:-1], scores) for xlocs, scores, sign in
               zip(ppgpeaks_cleaned, ppgpks_scores, ppgdatafilt)] # we do not have a score for 1st and last peak
# 1-second moving average to smoothen out kinks
weights_res = [scipy.ndimage.uniform_filter1d(weights, size=int(sr)) for weights in weights_res]
for weights in weights_res:
    weights[weights < delta] = delta
    weights[np.isnan(weights)] = delta
weights_res = np.asarray([weights ** 6 for weights in weights_res]) # now we have weights for each trace for each ppg sample

# calc resulting signal
normalized_weights = weights_res / np.sum(weights_res, axis=0)
fusion_signal = np.sum(ppgdatafilt*normalized_weights, axis=0)
fusionpeaks = helperfuncs.all_ppg_peaks(fusion_signal, sr)
fusionpeaks_cleaned = helperfuncs.clean_pklst(fusionpeaks, fusion_signal, sr)
fusionhr = helperfuncs.hr_at_loc(fusionpeaks_cleaned, eval_locs, sr, win_len=win_len, min_rrs=10)

# if there is ground truth r peaks, print results (MAE, Median AE)
if ecg_rpks is not None:
    ecghr = helperfuncs.hr_at_loc(ecg_rpks, eval_locs, sr, win_len=win_len, min_rrs=10)
    for idx, ppghr in enumerate(ppghrs):
        abserr = np.abs(ppghr-ecghr)
        print("ppg trace {} HR: Mean AE {:4.2f} Median AE {:4.2f}".format(idx, np.mean(abserr), np.median(abserr)))
    abserr = np.abs(fusionhr - ecghr)
    print("ppg fusion trace HR: Mean AE {:4.2f} Median AE {:4.2f}".format(np.mean(abserr), np.median(abserr)))


#%% plot input and output ppg traces
# fig = go.Figure()
# for pfilt in ppgdatafilt:
#     fig.add_trace(go.Scatter(y=pfilt, mode='lines', name='raw'))
# fig.add_trace(go.Scatter(y=fusion_signal, mode='lines', name='raw'))
# fig.show()
