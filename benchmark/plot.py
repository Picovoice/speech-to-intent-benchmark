import matplotlib.pyplot as plt

df_cafe = [0.67, 0.69, 0.74, 0.76, 0.78, 0.79, 0.84]
df_kitchen = [0.68, 0.72, 0.74, 0.77, 0.78, 0.79, 0.79]

pv_cafe = [0.85, 0.92, 0.92, 0.96, 0.96, 0.97, 0.97]
pv_kitchen = [0.9, 0.93, 0.96, 0.97, 0.97, 0.98, 0.98]

snr = [6, 9, 12, 15, 18, 21, 24]

plt.plot(snr, df_cafe, color='r', marker='o', label='Dialogflow (Cafe)')
plt.plot(snr, df_kitchen, color='b', marker='o', label='Dialogflow (Kitchen)')
plt.plot(snr, pv_cafe, color='r', linestyle='--', marker='o', label='Picovoice (Cafe)')
plt.plot(snr, pv_kitchen, color='b', linestyle='--', marker='o', label='Picovoice (Kitchen)')
plt.xlim(6, 24)
plt.xlabel('SNR dB')
plt.ylim(0.6, 1)
plt.ylabel('Accuracy (Command Acceptance Probability)')
plt.xticks([6, 9, 12, 15, 18, 21, 24])
plt.legend()
plt.grid()
plt.show()
