import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

data = pd.read_csv("scope_2_e2.csv")
time = [float(x) for count,x in enumerate(data['x-axis']) if count != 0]
volts_1 = [abs(float(x))+0.09 for count,x in enumerate(data['1']) if count != 0]
volts_2 = [float(x) for count,x in enumerate(data['2']) if count != 0]

sos = scipy.signal.butter(1, 0.01, 'lowpass', output='sos')
filtered_1 = scipy.signal.sosfilt(sos, volts_1)
filtered_2 = scipy.signal.sosfilt(sos, volts_2)
plt.plot(time[320:], volts_1[320:], label='Original signal')
plt.plot(time[320:], volts_2[320:], label='Reflected signal')
plt.plot(time[320:], filtered_1[320:], color='black', label='filtered signals')
plt.plot(time[320:], filtered_2[320:], color='black')
plt.xlabel('Time [seconds]', fontsize=18)
plt.ylabel('Volts [V]', fontsize=18)
plt.legend()
plt.show()
