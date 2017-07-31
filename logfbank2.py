# Packages we're using
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage

### Parameters ###
fft_size = 2048 # window size for the FFT
step_size = fft_size/16 # distance to slide along the window (in time)
spec_thresh = 0 # threshold for spectrograms (lower filters out more noise)
lowcut = 0 # Hz # Low cut for our butter bandpass filter
highcut = 15000 # Hz # High cut for our butter bandpass filter
samplerate = 48000
# For mels
n_mel_freq_components = 40 # number of mel frequency channels
shorten_factor = 1 # how much should we compress the x-axis (time)
start_freq = 20 # Hz # What frequency to start sampling our melS from 
end_freq = 8000 # Hz # What frequency to stop sampling our melS from 


# Most of the Spectrograms and Inversion are taken from: https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = (valid) // ss
    out = np.ndarray((nw,ws),dtype = a.dtype)

    for i in xrange(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start : stop]

    return out

def re_overlap(X, window_size, window_step):
    wave = np.zeros((X.shape[0]*window_step+window_size))
    for i in range(X.shape[0]):
        if i == X.shape[0]:
            wave[i*128:]=X[i][:]
        else:
            wave[i*128:i*128+127]=X[i][0:127]
    return wave


def stft(X, fftsize=fft_size, step=step_size, mean_normalize=False, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = np.fft.rfft
        cut = None
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2 + 1
    if mean_normalize:
        X -= X.mean()

    X = overlap(X, fftsize, step)
    
    size = fftsize
    win = scipy.hanning(size+1)[:-1]
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    phase = X/np.abs(X)
    return X,phase

def re_spectrogram(X,phase, fftsize=fft_size, step=step_size, mean_normalize=False, real=False,
         compute_onesided=True):
    X = phase * X
    if compute_onesided:
        temp = np.conjugate(X[:,-2:0:-1])
        X = np.concatenate([X, temp], axis=1)
        
    if real:
        local_ifft=np.fft.irfft
    else:
        local_ifft=np.fft.ifft
        
    
    X = np.real(local_ifft(X))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(fftsize) / (fftsize - 1))
    X = X / win[None]
    X = re_overlap(X,fftsize,step)
    return X
    

def pretty_spectrogram(d,log = False, thresh= 5, fft_size = 512, step_size = 64):
    """
    creates a spectrogram
    log: take the log of the spectrgram
    thresh: threshold minimum power for log spectrogram
    """
    specgram,phase = stft(d, fftsize=fft_size, step=step_size, real=True,
        compute_onesided=False)
    specgram = np.square(np.abs(specgram))
    if log == True: # volume normalize to max 1
        specgram = np.log10(specgram) # take log
        specgram[specgram < -thresh] = -thresh # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh # set anything less than the threshold as the threshold
    
    return specgram,phase

def mcra(specgram,log = False):
    #Initialize the parameters
    if log == True:
        specgram = np.power(10,specgram)
    length=specgram[0,:].size
    for x in range(specgram.shape[0]):
        if x == 0:
            noise_ps = specgram[x,:][:,None]
            P=specgram[x,:]
            Pmin=specgram[x,:]
            Ptmp=specgram[x,:]
            pk=np.zeros((length,1))
            ad=0.95
            a_s=0.8
            L=np.round(1000*2/20)
            delta=5
            ap=0.2
            result = noise_ps.T
        else:
            P=a_s*P+(1-a_s)*specgram[x,:]
            if np.remainder(x+1,L)==0:
                Pmin=np.minimum(Ptmp,P)
                Ptmp=P
            else:
                Pmin=np.minimum(Pmin,P)
                Ptmp=np.minimum(Ptmp,P)
            Srk=np.divide(P,Pmin)
            Ikl=np.zeros((length,1))
            ikl_index=np.nonzero(Srk>delta)
            Ikl[ikl_index]=1
            pk=ap*pk+(1-ap)*Ikl
            adk=ad+(1-ad)*pk
            noise_temp=np.multiply(adk,noise_ps)+np.multiply((1-adk),specgram[x,:][:,None])
            result=np.append(result,noise_temp.T,axis=0)
            noise_ps=noise_temp
    if log==True:
        result = np.log10(result)
    return result
            

def snr_estimation(mcra,restored_ps,noisy_ps,threshold,ay,ap,amax,amin,beta):
    one = np.zeros((noisy_ps/mcra).shape)+1

    post_snr_averaging=np.maximum([noisy_ps/mcra],one)
    post_snr_averaging=post_snr_averaging.reshape(post_snr_averaging.shape[1],post_snr_averaging.shape[2])
    for n in range(post_snr_averaging.shape[0]):
        if not n==0:
            post_snr_averaging[n,:]=ay*post_snr_averaging[n-1,:]+(1-ay)*post_snr_averaging[n,:]
    post_snr_averaging[post_snr_averaging<threshold]=0
    post_snr_averaging[post_snr_averaging>=threshold]=1
    for n in range(post_snr_averaging.shape[0]):
        if not n==0:
            post_snr_averaging[n,:]=ap*post_snr_averaging[n-1,:]+(1-ap)*post_snr_averaging[n,:]
    smoothing=amin+(1-post_snr_averaging)*(amax-amin)
    post_snr=np.maximum([noisy_ps/mcra],one)
    post_snr=post_snr.reshape(post_snr.shape[1],post_snr.shape[2])
    priori_snr=(1-smoothing)*(beta*(restored_ps/mcra)+(1-beta)*(post_snr-1))
            
    for n in range(post_snr_averaging.shape[0]):
        if not n==0:
            priori_snr[n,:]=smoothing[n,:]*priori_snr[n-1,:]+priori_snr[n,:]
    return priori_snr

def wiener_filter(priori_snr):
    return priori_snr/(1+priori_snr)

def power_estimate(mcra,restored_ps,noisy_ps,threshold,log,noisy_input,pcra,ay,ap,amax,amin,beta):
    if log:
        mcra = np.power(10,mcra)
        restored_ps = np.power(10,restored_ps)
        noisy_ps = np.power(10,noisy_ps)
    if pcra:
        priori_snr = snr_estimation(mcra,restored_ps,noisy_ps,threshold,ay,ap,amax,amin,beta)
    else:
        priori_snr = restored_ps/(restored_ps+mcra)
    w_filter=wiener_filter(priori_snr)
    if noisy_input:
        clean_ps = noisy_ps * w_filter
    else:
        clean_ps = restored_ps * w_filter
    return clean_ps

def istft(X, phase, overlap=2):
    X = np.sqrt(X)
    X = phase * X
    fftsize=(X.shape[1]-1)*2
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop) 
    for n,i in enumerate(range(0, len(x)-fftsize, hop)): 
        x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x


            
# Also mostly modified or taken from https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
def invert_pretty_spectrogram(X_s, log = False, fft_size = 512,compute_onesided=False, step_size = 512/4, n_iter = 10):
    
    if log == True:
        X_s = np.power(10, X_s)
    X_s = np.sqrt(X_s)
    if compute_onesided:
        X_s = np.concatenate([X_s, X_s[:,-2:0:-1]], axis=1)
    X_t = iterate_invert_spectrogram(X_s, fft_size, step_size, n_iter=n_iter)
    return X_t

def iterate_invert_spectrogram(X_s, fftsize, step, n_iter=10, verbose=False):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    X_best = copy.deepcopy(X_s)
    for i in range(n_iter):
        if verbose:
            print("Runnning iter %i" % i)
        if i == 0:
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=True)
        else:
            # Calculate offset was False in the MATLAB version
            # but in mine it massively improves the result
            # Possible bug in my impl?
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=False)
        est,phase= stft(X_t, fftsize=fftsize, step=step, compute_onesided=False)
        X_best = X_s * phase[:len(X_s)]
    X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                             set_zero_phase=False)
    return np.real(X_t)

def invert_spectrogram(X_s, step, calculate_offset=True, set_zero_phase=True):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    size = int(X_s.shape[1] // 2)
    wave = np.zeros((X_s.shape[0] * step + size))
    # Getting overflow warnings with 32 bit...
    wave = wave.astype('float64')
    total_windowing_sum = np.zeros((X_s.shape[0] * step + size))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))

    est_start = int(size // 2) - 1
    est_end = est_start + size
    for i in range(X_s.shape[0]):
        wave_start = int(step * i)
        wave_end = wave_start + size
        if set_zero_phase:
            spectral_slice = X_s[i].real + 0j
        else:
            # already complex
            spectral_slice = X_s[i]

        # Don't need fftshift due to different impl.
        wave_est = np.real(np.fft.ifft(spectral_slice))[::-1]
        if calculate_offset and i > 0:
            offset_size = size - step
            if offset_size <= 0:
                offset_size = step
            offset = xcorr_offset(wave[wave_start:wave_start + offset_size],
                                  wave_est[est_start:est_start + offset_size])
        else:
            offset = 0
        wave[wave_start:wave_end] += win * wave_est[est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += win
    wave = np.real(wave) / (total_windowing_sum + 1E-6)
    return wave

def xcorr_offset(x1, x2):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2 + 1
    corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    offset = corrs.argmax() - len(x1)
    return offset

# def freq_to_mel(f):
#     return 2595.*np.log10(1+(f/700.))
# def mel_to_freq(m):
#     return 700.0*(10.0**(m/2595.0)-1.0)

# def create_mel_filter(fft_size, n_freq_components = 64, start_freq = 300, end_freq = 8000):
#     """
#     Creates a filter to convolve with the spectrogram to get out mels

#     """
#     spec_size = fft_size/2
#     start_mel = freq_to_mel(start_freq)
#     end_mel = freq_to_mel(end_freq)
#     plt_spacing = []
#     # find our central channels from the spectrogram
#     for i in range(10000):
#         y = np.linspace(start_mel, end_mel, num=i, endpoint=False)
#         logp = mel_to_freq(y)
#         logp = logp/(rate/2/spec_size)
#         true_spacing = [int(i)-1 for i in np.ceil(logp)] 
#         plt_spacing_mel = np.unique(true_spacing)
#         if len(plt_spacing_mel) == n_freq_components:
#             break
#     plt_spacing = plt_spacing_mel
#     if plt_spacing_mel[-1] == spec_size:
#         plt_spacing_mel[-1] = plt_spacing_mel[-1]-1
#     # make the filter
#     mel_filter = np.zeros((int(spec_size),n_freq_components))
#     # Create Filter
#     for i in range(len(plt_spacing)):  
#         if i > 0:
#             if plt_spacing[i-1] < plt_spacing[i] - 1:
#                 # the first half of the window should start with zero
#                 mel_filter[plt_spacing[i-1]:plt_spacing[i], i] = np.arange(0,1,1./(plt_spacing[i]-plt_spacing[i-1]))
#         if i < n_freq_components-1:
#             if plt_spacing[i+1] > plt_spacing[i]+1:
#                 mel_filter[plt_spacing[i]:plt_spacing[i+1], i] = np.arange(0,1,1./(plt_spacing[i+1]-plt_spacing[i]))[::-1]
#         elif plt_spacing[i] < spec_size:
#             mel_filter[plt_spacing[i]:int(mel_to_freq(end_mel)/(rate/2/spec_size)), i] =  \
#                 np.arange(0,1,1./(int(mel_to_freq(end_mel)/(rate/2/spec_size))-plt_spacing[i]))[::-1]
#         mel_filter[plt_spacing[i], i] = 1
#     # Normalize filter
#     mel_filter = mel_filter / mel_filter.sum(axis=0)
#     # Create and normalize inversion filter
#     mel_inversion_filter = np.transpose(mel_filter) / np.transpose(mel_filter).sum(axis=0)
#     mel_inversion_filter[np.isnan(mel_inversion_filter)] = 0 # for when a row has a sum of 0

#     return mel_filter, mel_inversion_filter

def make_mel(spectrogram, mel_filter, shorten_factor = 1, log=False):
    mel_spec =np.transpose(mel_filter).dot(np.transpose(spectrogram))
    if log:
        return np.log10(mel_spec)
    else:
        return mel_spec


def mel_to_spectrogram(mel_spec, mel_inversion_filter, spec_thresh, shorten_factor,log=False):
    """
    takes in an mel spectrogram and returns a normal spectrogram for inversion 
    """
    if log == True:
        mel_spec = np.power(10, mel_spec)
    mel_spec = (mel_spec+spec_thresh)
    uncompressed_spec = np.transpose(np.transpose(mel_spec).dot(mel_inversion_filter))
    uncompressed_spec = scipy.ndimage.zoom(uncompressed_spec, [1,shorten_factor])
    uncompressed_spec = uncompressed_spec -4
    return uncompressed_spec

# From https://github.com/jameslyons/python_speech_features

def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)
    
def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)
'''
def re_powspec(spectrogram,NFFT = fft_size):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """    
    return np.sqrt(NFFT*spectrogram)

def powspec(spectrogram,NFFT = fft_size):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """    
    return 1.0/NFFT * np.square(spectrogram)
'''
def get_filterbanks(nfilt=n_mel_freq_components,nfft=fft_size,samplerate=samplerate,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def create_mel_filter(fft_size, n_freq_components = n_mel_freq_components, start_freq = start_freq, end_freq = end_freq, samplerate=samplerate):
    """
    Creates a filter to convolve with the spectrogram to get out mels

    """
    mel_inversion_filter = get_filterbanks(nfilt=n_freq_components, 
                                           nfft=fft_size, samplerate=samplerate, 
                                           lowfreq=start_freq, highfreq=end_freq)
    # Normalize filter
    mel_filter = mel_inversion_filter.T / mel_inversion_filter.sum(axis=1)

    return mel_filter, mel_inversion_filter


def fbank(data, fft_size = 2048,
        step_size = fft_size/16,
        spec_thresh = 0, 
        lowcut = 0, 
        highcut = 15000, 
        samplerate = 48000,
        n_mel_freq_components = 40, 
        shorten_factor = 1,
        start_freq = 20,
        end_freq = 8000,
        log = False
        ):
    wav_spectrogram,phase = pretty_spectrogram(data, fft_size = fft_size, 
                                   step_size = step_size, log = False, thresh = spec_thresh)

    mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq,
                                                        samplerate=samplerate)

    mel_spec = np.transpose(make_mel(wav_spectrogram, mel_filter, shorten_factor = shorten_factor, log = log))
    mel_spec[mel_spec<0]=0
    return mel_spec

def invfbank(mel_spec, fft_size = 2048,
        step_size = fft_size/16,
        spec_thresh = 0, 
        lowcut = 0, 
        highcut = 15000, 
        samplerate = 48000,
        n_mel_freq_components = 40, 
        shorten_factor = 1,
        start_freq = 20,
        end_freq = 8000,
        log = False
        ):
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq,
                                                        samplerate=samplerate)
    re_mel_spec = mel_to_spectrogram(mel_spec, mel_inversion_filter, spec_thresh, shorten_factor,log=log)
    recovered_audio_orig = invert_pretty_spectrogram(re_mel_spec.T, fft_size = fft_size,
                                         step_size = step_size, log = False, n_iter = 10)
    return recovered_audio_orig



def power(data, fft_size = 2048,
        step_size = fft_size/16,
        spec_thresh = 0, 
        lowcut = 0, 
        highcut = 15000, 
        samplerate = 48000,
        noise = False,
        log = False
        ):
    data = data[:,0].astype('float64')
    wav_spectrogram,phase = pretty_spectrogram(data, fft_size = fft_size, 
                                   step_size = step_size, log = log, thresh = spec_thresh)
    if noise:
        noise_ps = mcra(wav_spectrogram,log=log)
        return wav_spectrogram,noise_ps,phase
    return wav_spectrogram,phase

def invpower(data, fft_size = 2048,
        step_size = fft_size/16,
        spec_thresh = 0, 
        lowcut = 0, 
        highcut = 15000, 
        samplerate = 48000,
        n_mel_freq_components = 40, 
        shorten_factor = 1,
        start_freq = 20,
        end_freq = 8000,
        log = False
        ):
    recovered_audio_orig = invert_pretty_spectrogram(data.T, fft_size = fft_size,
                                         step_size = step_size, log = log, n_iter = 300)
    return recovered_audio_orig