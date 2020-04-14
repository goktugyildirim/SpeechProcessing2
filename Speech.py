#!/usr/bin/env python
# coding: utf-8

# In[150]:


from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
from playsound import playsound
import wavio

class SpeechProcessing(object):
    
    def __init__(self):   
        import numpy as np
    
    def window(self, x, fs, window_dur_in_second, frame_shift):
        import numpy as np
        sound_length = len(x)
        window_length = int(fs*window_dur_in_second)
        shift_length = int(fs*frame_shift)
        num_window = int((sound_length-window_length)/shift_length + 1)

        print("Sound length", sound_length)
        print("Sampling Rate:", fs)
        print("Window length", window_length)
        print("Shift length:",shift_length)
        print("Num window: ",num_window)

        windowed_data = []

        for i in range(int(num_window)):
            window = [0.54-0.46*np.cos((2*3.14*i)/(window_length-1)) for i in range(window_length)]#applying Hamming window
            frame = x[(i*shift_length):(i*shift_length)+window_length]*window
            windowed_data.append(frame)

        return windowed_data
    
    
    def shortTimeZeroCrossing(self, windowed_data, frame_shift, fs):
        
        shift_length = int(fs*frame_shift)
        
        import numpy as np
        zc_vector = []
        
        for frame in (windowed_data):
            sum = 0
            for i in range(len(frame)-1):
                first_element = frame[i]
                second_element = frame[i+1]
                element = np.abs(np.sign(first_element)-np.sign(second_element))
                sum = sum + element
            zc = (shift_length*sum)/(len(frame)*2)
            zc_vector.append(zc)#her window için tek bir zc değeri
    
        zc_vector = np.array(zc_vector).reshape((len(windowed_data),1))
        #zc_vector = zc_vector/max(zc_vector)
        #zc_vector = 10*np.log10(zc_vector)
        
        return zc_vector
    
    
    def shortTimeEnergy(self, windowed_data):
        import numpy as np
        energy_Vector = []
        sum = 0
        for frame in (windowed_data):
            sum = 0
            for i in range(len(frame)):
                sum = sum + frame[i]*frame[i]
            energy_Vector.append(sum)
        
        energy_Vector = np.array(energy_Vector).reshape((len(windowed_data),1))
        energy_Vector = energy_Vector/max(energy_Vector)
        energy_Vector = 10*np.log10(energy_Vector)
        
        return energy_Vector
    
    
    def endPointDetector(self, shortTimeEnergy, shortTimeZeroCrossing, window_dur_in_second, frame_shift, fs):
        import numpy as np
        #100ms to frame count:
        frame_count = int(0.1/window_dur_in_second)
        energy = shortTimeEnergy[:frame_count]
        zc = shortTimeZeroCrossing[:frame_count]
        
        av_energy = np.mean(energy)
        av_zc = np.mean(zc)
        std_energy = np.std(energy)
        std_zc = np.std(zc)
        
        IF = 35
        IZCT = max(IF, av_zc + 3*std_zc)
        ITU = -15 #constant in range of -10 to -20dB
        ITR = max(ITU-10, av_energy + 3*std_zc)
        
         #Find B1**********************************
        B1 = 0
        for i in range(len(shortTimeEnergy)-15):
            amplitude = shortTimeEnergy[i]
            frame = shortTimeEnergy[i:i+15]
            
            if amplitude >= ITR:
                if np.mean(frame)>ITU:
                    B1 = i
                    break    
        #****************************************** 
        #Find E1**********************************
        E1 = 0
        for i in range(len(shortTimeEnergy)-1,15,-1):
            amplitude = shortTimeEnergy[i]
            frame = shortTimeEnergy[i-15:i]
            
            if amplitude >= ITR:
                if np.mean(frame)>ITU:
                    E1 = i
                    break 
         #******************************************   
        #Find B2**********************************
        B2 = B1
        counter = 0
        for i in range(B1,B1-25,-1):
            amplitude = shortTimeZeroCrossing[i]
            if amplitude > IZCT:
                counter += 1
         
        if counter>4:
            for i in range(len(shortTimeZeroCrossing)):
                amplitude = shortTimeZeroCrossing[i]
                if amplitude > IZCT:
                    B2 = i
                    break
        #******************************************   
        #Find E2**********************************
        E2 = E1
        counter = 0
        for i in range(E1,E1+25,1):
            amplitude = shortTimeZeroCrossing[i]
            if amplitude > IZCT:
                counter += 1
        
        if counter>4:
            for i in range(len(shortTimeZeroCrossing)-1,E1,-1):
                amplitude = shortTimeZeroCrossing[i]
                if amplitude > IZCT:
                    E2 = i
                    break
        #******************************************
        
        time_id_B1 = (frame_shift)*B1
        sample_id_B1 = int(time_id_B1*fs)
        time_id_E1 = (frame_shift)*E1
        sample_id_E1 = time_id_E1*fs
        time_id_B2 = (frame_shift)*B2
        sample_id_B2 = int(time_id_B2*fs)
        time_id_E2 = (frame_shift)*E2
        sample_id_E2 = int(time_id_E2*fs)
        
        print("ITR: ",ITR,"|ITU: ",ITU,"|IZCT: ",IZCT)
        print("Frame ID B1:",B1, "|Frame ID E1:", E1, "|Frame ID B2:",B2,"|Frame ID E2:",E2)
        print("Start id: ",sample_id_B2," End id:", sample_id_E2)
        
        return sample_id_B2, sample_id_E2
    
    
    
    def STFT(self, data, N, fs, window_dur_in_second, frame_shift, plot):
        
        windowed_data = self.window(data,fs, window_dur_in_second , frame_shift = frame_shift )
        
        print("FFT N: ",N)
        import numpy as np
        
        STFT = []
        dft_frame = []
        
        for i,frame in enumerate(windowed_data):            
            
            dft_frame = np.fft.fft(frame,N)
            STFT.append(dft_frame)
            
            if plot==True:
                f = np.arange(0,fs/2,fs/N)
                plt.plot(f,20*np.log10(np.abs(dft_frame[:int(len(dft_frame)/2)])))
                plt.grid(True)
                plt.title("FFT of Frame: {}".format(i))
                plt.xlabel("Freqency(Hz)")
                plt.figure()
                
    
        STFT = np.array(STFT)
        return STFT
    
    def plotSTFT(self, STFT, N ,fs):
        import numpy as np
        from scipy import signal
        import matplotlib.pyplot as plt
        
        STFT = np.transpose(STFT)
        STFT = STFT[:int(N/2)][:]
        f = np.arange(0,fs/2,fs/N)
        f = f.reshape((f.shape[0],1))
        t = np.arange(0,STFT.shape[1],1)
        
        plt.pcolormesh(t, f, 20*np.log10(np.abs(STFT)))
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Frames')
        plt.show()
        
        
        
    def cepstrum(self, data, N, fs, window_dur_in_second, frame_shift, lp_liftering, plot, cutoff):
        windowed_data = self.window(data,fs, window_dur_in_second , frame_shift = frame_shift )
        
        import numpy as np
        
        cepstrum = []
                
        for i,frame in enumerate(windowed_data):     
            
            dft_frame = np.fft.fft(frame,N)
            
            ceps_frame = np.real(np.fft.ifft(np.log10(np.abs(np.fft.fft(frame,N))))).reshape((N,1)).ravel()
            #print(ceps_frame.shape)
            
            if lp_liftering == True:
                ones = np.ones((1,cutoff))
                zeros = np.zeros((1,(1024-(2*cutoff))))
                lif = np.concatenate((ones,zeros,ones),axis=1).ravel()
                ceps_frame = ceps_frame*lif
            
            cepstrum.append(ceps_frame)
            
            if plot==True:
                ceps_frame = ceps_frame/max(ceps_frame)
                f = np.arange(0,N,1)
                plt.plot(ceps_frame)
                
                scale_factor = 0.05
                ymin, ymax = plt.ylim()
                plt.ylim(ymin * scale_factor, ymax * scale_factor)
                
                plt.grid(True)
                plt.title("Cepstrum of Frame: {}: ".format(i))
                plt.xlabel("Quefrency (Sample Id)")
                plt.figure()
                
                
                

                

                

                
                
                
                f = np.arange(0,fs/2,fs/N)
                plt.plot(f,20*np.log10(np.abs(dft_frame[:int(len(dft_frame)/2)])))
                plt.grid(True)
                plt.title("FFT of Frame: {} Before Cepstrum Operation: ".format(i))
                plt.xlabel("Freqency(Hz)")
                plt.figure()
                
                
                dft_frame = np.fft.fft(ceps_frame,N)
                f = np.arange(0,fs/2,fs/N)
                plt.plot(f,20*np.log10(np.abs(dft_frame[:int(len(dft_frame)/2)])))
                plt.grid(True)
                plt.title("FFT of Frame: {} After Cepstrum Operation".format(i))
                plt.xlabel("Freqency(Hz)")
                plt.figure()
                
        return cepstrum
        


# ### Read Sound "Ödevin uzun olmamasına üzüldüm"

# In[151]:


#Read Sound
fs, data = wavfile.read('3.wav')


# ### Wideband Spectogram with SciPy Module

# In[152]:


from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

window_length = 160
shift_length = window_length/2
f, t, Zxx = signal.stft(data,fs,window="hamming",nfft=1024,nperseg =window_length,noverlap=shift_length,padded = True)
plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Frames')
plt.show()


# ### Narrowband Spectogram with SciPy Module

# In[153]:


from scipy import signal
import matplotlib.pyplot as plt
window_length = 1024
shift_length = window_length/2
f, t, Zxx = signal.stft(data,fs,window="hamming",nfft=1024,nperseg =window_length,noverlap=shift_length,padded = True)
plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Frames')
plt.show()


# ### Wideband Spectogram with My STFT Function and My Plotting Function

# In[154]:


#Short Time Speech Processing
speech = SpeechProcessing()
wideband_STFT = speech.STFT(data, 1024, fs, window_dur_in_second = 0.01, frame_shift = 0.005, plot = False)
speech.plotSTFT(wideband_STFT, 1024 ,fs)


# ### Narrowband Spectogram with My STFT Function and My Plotting Function

# In[155]:


speech = SpeechProcessing()
narrowband_STFT = speech.STFT(data, 1024, fs, window_dur_in_second = 0.064, frame_shift = 0.032, plot = False)
speech.plotSTFT(narrowband_STFT, 1024 ,fs)


# ### PART 3 # 10000-30000:Voiced sound # 65000-66000:Unvioced sound

# In[156]:


#Read Sound
fs, data = wavfile.read('letters.wav')
speech = SpeechProcessing()
voiced_sound = data[10000:30000]
unvoiced_sound = data[65000:66000]
windowed_sound = speech.window(data, fs, window_dur_in_second = 0.03, frame_shift = 0.01)
plt.plot(data/max(data))


# ### Plotting Waveform of All Frames

# In[157]:


for frame_id, frame in enumerate(windowed_sound):
    frame_shift = 0.01
    sample_id = int((frame_shift)*frame_id*fs)
    plt.plot(frame/max(frame))
    plt.title("Frame id:{} Sample id:{}".format(frame_id,sample_id))
    plt.figure()


# ### Plotting DFT of Voiced Frames

# In[158]:


speech = SpeechProcessing()
voiced_sound = data[10000:30000]
STFT = speech.STFT(voiced_sound, 1024, fs, window_dur_in_second = 0.03, frame_shift = 0.01, plot = True)


# ### Plotting Cepstrum of Voiced Frames

# In[159]:


fs, data = wavfile.read('letters.wav')
speech = SpeechProcessing()
cepstrum = speech.cepstrum(voiced_sound, 1024, fs, window_dur_in_second = 0.03, frame_shift = 0.01, lp_liftering = True, plot = True, cutoff=25)


# ### Plotting Cepstrum of Unvoiced Frames

# In[160]:


fs, data = wavfile.read('letters.wav')
speech = SpeechProcessing()
cepstrum = speech.cepstrum(unvoiced_sound, 1024, fs, window_dur_in_second = 0.03, frame_shift = 0.01, lp_liftering = True, plot = True, cutoff = 50)


# In[ ]:




