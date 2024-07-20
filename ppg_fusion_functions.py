from scipy.signal import butter, sosfiltfilt
import numpy as np
import scipy

# plotting
# import plotly.io as pio
# import plotly.graph_objects as go
# pio.renderers.default = 'browser'

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def filter_ppg(rawppg, sr, clip=True, clip_ampl=0.02):
    '''
    Clipping can be used to minimize impact of large motion artifacts but requires a datasource-specific
    clip-amplitude which corresponds to the maximum valid AC component the signal may experience
    '''
    ppgraw = np.nan_to_num(rawppg)
    if clip:
        ppgdata2 = butter_bandpass_filter(-ppgraw, 0.6, 3.3, sr, order=1)
        ppgdata2 = np.clip(ppgdata2, -clip_ampl/2, clip_ampl/2)
        ppgdata2 = butter_bandpass_filter(ppgdata2, 0.6, 3.3, sr, order=1)
    else:
        ppgdata2 = butter_bandpass_filter(-ppgraw, 0.6, 3.3, sr, order=2)
    return scipy.stats.zscore(ppgdata2)

def find_ppg_peaks(hrdata, sr, windowsize=0.75, bpmmin=40):
    '''
    PPG peak detection, based on the heartpy library.
    Compared to heartpy, we do more sanity checks: Peaks are only added
    if they are a local maximum and offsets are only considered if they
    produce a sensible heartrate above bpmmin
    '''
    # check that the data has positive baseline for the moving average algorithm to work
    hrdata -= np.percentile(hrdata, 1)

    # moving average offset values to test
    ma_perc_list = [0, 5, 10, 15, 20, 30, 40]
    rmean = scipy.ndimage.uniform_filter1d(np.asarray(hrdata, dtype='float'), size=int(windowsize*sr))

    rrsd = []
    valid_ma = []
    all_ma = []

    for ma_perc in ma_perc_list:
        rol_mean = rmean + np.mean(rmean) * ma_perc/ 100

        peaksx = np.where((hrdata > rol_mean))[0]
        peaksy = hrdata[peaksx]
        peakedges = np.concatenate((np.array([0]),
                                    (np.where(np.diff(peaksx) > 1)[0]+1),
                                    np.array([len(peaksx)])))
        peaklist = []
        for i in range(0, len(peakedges) - 1):
            # find a peak in every period where the signal crosses its own rolling mean
            try:
                y_values = peaksy[peakedges[i]:peakedges[i + 1]].tolist()
                max_y_idx = y_values.index(max(y_values))
                if 0<max_y_idx<len(y_values)-1: # only add if it is a local maxima or minimum which is not guaranteed
                    peaklist.append(peaksx[peakedges[i] + max_y_idx])
            except:
                pass

        # if possible, we only consider mean offsets which produced sensible hrs
        if len(peaklist) > len(hrdata)*bpmmin//(60*sr):
            rr_list = (np.diff(peaklist) / sr) * 1000.0
            rrsd = np.std(rr_list)
            valid_ma.append([rrsd, peaklist, ma_perc])
        else:
            all_ma.append([rrsd, peaklist])

    if len(valid_ma) > 0: # at least one offset produced a sensible hr
        return min(valid_ma, key=lambda t: t[0])[1]
    else:
        return max(all_ma, key=lambda t: len(t[1]))[1]


def all_ppg_peaks(signal, sr, segment_size=120):
    '''
    computes PPG peaks by splitting it into segments with slight overlap to reduce
    risk of fragments. Peaks in the overlapping part are ignored
    '''
    segment_size = round(sr*segment_size) # do peak detection in 2min windows
    side_buffer = round(4*sr) # append 4 seconds on each side to avoid edge effects
    win_s = 0
    all_peaks = []
    while win_s < len(signal)-2*side_buffer:
        win_e = min(win_s+segment_size+2*side_buffer, len(signal))
        peaklist = np.asarray(find_ppg_peaks(scipy.stats.zscore(signal[win_s:win_e]), sr = sr))
        peaklist = peaklist[np.logical_and(peaklist>side_buffer, peaklist<win_e-win_s-side_buffer)]
        all_peaks = all_peaks+list(peaklist+win_s)
        win_s = win_e-2*side_buffer
    all_peaks = np.array(all_peaks, dtype='int32')
    return all_peaks


def clean_pklst(pks, data, sr, maxhr=185):
    '''
    clean ppg peak list with respect to maximum HR
    if there are too many peaks, take retain the highest, remove the others
    '''
    dellst = []
    min_smpl = int(sr / (maxhr / 60))
    rng = min_smpl//2
    for pkidx in range(len(pks)):
        pk = pks[pkidx]
        maxloc = np.argmax(data[pk-rng:pk+rng+1])
        if pk != maxloc+pk-rng:
            if maxloc == 0 or maxloc == 2*rng or maxloc+pk-rng in pks: # if there is no max within a 40smpl window remove peak
                #print("d", pk)
                dellst.append(pkidx)
            else:
                pks[pkidx] = maxloc+pk-rng
                #print("m", pk, pks[pkidx])
    pks = list(np.delete(pks, dellst))

    # remove based on max hr (too close together)
    idx = 1
    while idx < len(pks):
        prev = pks[idx] - pks[idx - 1]
        if prev < min_smpl:
            if prev < min_smpl and idx < len(pks) - 1 and pks[idx + 1] - pks[idx] < min_smpl:
                del pks[idx]
            elif pks[idx] > pks[idx - 1]:
                del pks[idx - 1]
            else:
                del pks[idx]
        else:
            idx += 1
    return np.array(pks)


def get_av_offset(pks1, pks2, sr):
    '''
    This can be used to align ppg traces based on their detected peaks.
    This may be needed to account for PTT effects in traces from different body locations
    '''
    offsets = []
    offsets_x = []
    pk2_idx = 0
    if len(pks1) <= 1 or len(pks2) <= 1:
        return 0
    for pk in pks1:
        while pk2_idx<len(pks2)-2 and abs(pks2[pk2_idx+1]-pk) < abs(pks2[pk2_idx]-pk):
            pk2_idx+=1
        offs = pks2[pk2_idx]-pk
        if abs(offs) < sr*0.15:
            offsets.append(offs)
            offsets_x.append(pk)
            pk2_idx += 1
            if pk2_idx == len(pks2):
                break

    return np.nan_to_num(scipy.stats.trim_mean(offsets, 0.2))

def quotient_filter(hbpeaks, outlier_over=5, sampling_rate=128, tol=0.8):
    '''
    Function that applies a quotient filter similar to
    "Piskorki, J., Guzik, P. (2005), Filtering Poincare plots"
    peaks and IBI are considered good, if they are part of a stretch of
    @outlier_over peaks where the min IBI is at least @tol*IBImax
    '''

    good_hbeats = []
    good_rrs = []
    good_rrs_x = []
    for i, peak in enumerate(hbpeaks[:-outlier_over-1]):
        hb_intervals = [hbpeaks[j]-hbpeaks[j-1]  for j in range(i+1, i+outlier_over)]
        hr = 60/((sum(hb_intervals))/((outlier_over-1)*sampling_rate))
        if min(hb_intervals) > max(hb_intervals)*tol and hr > 35 and hr < 185: # -> good data
            for p in hbpeaks[i+1:i+outlier_over-1]:
                if len(good_hbeats) == 0 or p > good_hbeats[-1]:
                    good_hbeats.append(p)
                    if len(good_hbeats) > 1:
                        rr = good_hbeats[-1]-good_hbeats[-2]
                        if rr<min(hb_intervals)/tol and rr>max(hb_intervals)*tol:
                            good_rrs.append(rr)
                            good_rrs_x.append(np.mean([good_hbeats[-1], good_hbeats[-2]]))
    return np.array(good_hbeats), np.array(good_rrs), np.array(good_rrs_x)


def hr_at_loc(pks, locs, sr, win_len=15, min_rrs=10):
    '''
    calculates HR based on peaks (R peaks from PPG or ECG sources)
    at a given list of locations (indices) over a window of configurable length.
    Only produces a HR if there are at least min_rrs viable IBI to work with
    '''
    _, rrs, rrxs = quotient_filter(pks, outlier_over=5, tol=0.51)
    win = win_len*sr
    hrs = []
    for loc in locs:
        msk = np.logical_and(rrxs>=loc-win/2,rrxs<loc+win/2 )
        rrs_win = rrs[msk]
        if len(rrs_win)>min_rrs: # at least min_rrs hb
            hrs.append(60*len(rrs_win)/(np.sum(rrs_win)/sr))
        else: # if there is no useable information, should be avoided, could be changed to nan values
            if len(hrs)>0:
                hrs.append(hrs[-1]) # use last measurement if no peaks
            else:
                hrs.append(75) # if at the start, use random guess of 75
    return np.asarray(hrs)


def resample_data(data, l_target):
    xvals = np.linspace(0, l_target-1, len(data)) #-1 because endpoint is included
    return np.interp(list(range(0, l_target)), xvals, data)


class wave_template:
    '''
    class to build and use wave templates
    '''
    def __init__(self):
        self.pps = []
        self.datas = []
        self.win_size = 40
        self._templ = None # buffer for performance
        self.sanity_signal = np.interp(list(range(0, self.win_size)), [0, self.win_size//3, self.win_size//2, 5*self.win_size//6,  self.win_size-1], [1, 0, 1, 0, 1])

    def preproc_data(self, data):
        '''
        resamples data to the size of the template to be independent of HR
        also standardizes the data
        '''
        d = resample_data(data, self.win_size)
        return (d-np.mean(d))/np.std(d)

    def get_template_corr(self, data):
        '''
        compute correlation of a single or multiple waves with the template
        '''
        if type(data[0]) == np.ndarray: # multiple
            data_normed = [self.preproc_data(d) for d in data]
            return np.nan_to_num(np.corrcoef(data_normed, self.template)[:-1, -1])
        else: # 1 dim
            if len(self.datas)<5:
                return np.nan
            return np.nan_to_num(np.corrcoef(self.preproc_data(data), self.template)[1][0])

    def add_pulse(self, data, do_sanity_check=True):
        '''
        add a pulse to the template (formation)
        '''
        pp = len(data)
        if pp<42*2 or pp>180*2:
            return False
        d = self.preproc_data(data)

        if do_sanity_check and (np.sum(np.isnan(d))>0 or np.corrcoef(self.sanity_signal, d)[1][0]<0.8):
            return False
        self.pps.append(pp)
        self.datas.append(d)
        self._templ = None
        return True
    @property
    def len(self):
        '''
        returns the number of waves currently contributing to the template
        '''
        return len(self.pps)

    def rm_worst_signal(self, limit_corr=None, batch=1):
        '''
        remove least fitting signals from the template to converge towards a good looking wave
        '''
        oldlen = self.len
        if self.len>15: # remove low corrs
            corrs = []
            # batching correlation calculation into 5000 at a time to limit matrix size and RAM requirements
            for i in range(0, len(self.datas), 5000):
                corrs += list(np.nan_to_num(np.corrcoef(self.datas[i:min(i + 5000, len(self.datas))], self.template)[:-1, -1]))
            worsts = np.argsort(corrs)[:batch]
            for worst in sorted(worsts, reverse=True):
                if limit_corr is None or corrs[worst]<limit_corr:
                    del self.pps[worst]
                    del self.datas[worst]
                    self._templ = None
        return self.len<oldlen # return whether something was removed


    @property
    def template(self):
        '''
        the actual template wave
        '''
        if self._templ is None:
            self._templ = np.mean(self.datas, axis=0)
        return self._templ

    # def plot(self):
    #     fig = go.Figure()
    #     for d in self.datas:
    #         fig.add_trace(go.Scatter(y=d, mode='lines'))
    #     fig.add_trace(go.Scatter(y=self.template, mode='lines', name='template', line = dict(color='firebrick', width=8)))
    #     fig.show()