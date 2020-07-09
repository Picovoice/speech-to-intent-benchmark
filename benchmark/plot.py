import matplotlib.pyplot as plt
import numpy as np

aws_cafe = np.array([0.70, 0.80, 0.85, 0.86, 0.87, 0.87, 0.87])
aws_kitchen = np.array([0.79, 0.83, 0.85, 0.86, 0.87, 0.87, 0.87])
aws = (aws_cafe + aws_kitchen) / 2.

df_cafe = np.array([0.63, 0.70, 0.75, 0.78, 0.79, 0.80, 0.81])
df_kitchen = np.array([0.69, 0.73, 0.77, 0.80, 0.81, 0.81, 0.83])
df = (df_cafe + df_kitchen) / 2.

ibm_cafe = np.array([0.51, 0.76, 0.85, 0.93, 0.95, 0.96, 0.97])
ibm_kitchen = np.array([0.69, 0.82, 0.91, 0.93, 0.94, 0.96, 0.96])
ibm = (ibm_cafe + ibm_kitchen) / 2.

pv_cafe = np.array([0.90, 0.94, 0.97, 0.97, 0.97, 0.98, 0.98])
pv_kitchen = np.array([0.96, 0.97, 0.98, 0.98, 0.99, 0.99, 0.99])
pv = (pv_cafe + pv_kitchen) / 2.

snr = [6, 9, 12, 15, 18, 21, 24]

plt.plot(snr, aws, color='g', marker='^', label='Amazon Lex')
plt.plot(snr, df, color='r', marker='x', label='Google Dialogflow')
plt.plot(snr, ibm, color='k', marker='s', label='IBM Watson')
plt.plot(snr, pv, color='b', marker='o', label='Picovoice Rhino')
plt.xlim(6, 24)
plt.xlabel('SNR dB')
plt.ylim(0.6, 1)
plt.ylabel('Accuracy (Command Acceptance Probability)')
plt.xticks([6, 9, 12, 15, 18, 21, 24])
plt.legend()
plt.grid()
plt.show()
