import librosa
import numpy as np
from numpy import linalg

y,sr =librosa.load("./Thunder.wav")

x = len(y)#83711
z = 1
index = 0
while z < x:
    z = z*2
    index = index + 1
print(z)
print(2**index)

y_zeroes = np.pad(y,(0, 2*z-x), 'constant', constant_values=(0,0))

w_zeroes = np.pad(y,(0, 2*z-x), 'constant', constant_values=(0,0))

#s1 = librosa.stft(y_zeroes)
s1 = np.fft.fft(y_zeroes, norm="ortho")
#s2 = librosa.stft(w_zeroes)
s2 = np.fft.fft(w_zeroes, norm="ortho")

s2_conj = np.conjugate(s2)

m = np.multiply(s1, s2_conj)

#o = librosa.istft(m)
o = np.fft.ifft(m, norm="ortho")

print(np.amax(o))
print(np.argmax(o))

o_norm = o/linalg.norm(o)
print(np.amax(o_norm))
print(np.argmax(o_norm))
