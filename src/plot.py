import matplotlib.pyplot as plt
import numpy as np

AMAZON_LEX_CAFE = np.array([0.71, 0.82, 0.84, 0.87, 0.87, 0.87, 0.87])
AMAZON_LEX_KITCHEN = np.array([0.80, 0.84, 0.85, 0.87, 0.87, 0.87, 0.88])
AMAZON_LEX = (AMAZON_LEX_CAFE + AMAZON_LEX_KITCHEN) / 2.

GOOGLE_DIALOGFLOW_CAFE = np.array([0.63, 0.70, 0.76, 0.79, 0.82, 0.83, 0.82])
GOOGLE_DIALOGFLOW_KITCHEN = np.array([0.70, 0.78, 0.78, 0.80, 0.81, 0.82, 0.83])
GOOGLE_DIALOGFLOW = (GOOGLE_DIALOGFLOW_CAFE + GOOGLE_DIALOGFLOW_KITCHEN) / 2.

IBM_WATSON_CAFE = np.array([0.50, 0.76, 0.88, 0.92, 0.94, 0.95, 0.96])
IBM_WATSON_KITCHEN = np.array([0.72, 0.84, 0.89, 0.93, 0.95, 0.95, 0.96])
IBM_WATSON = (IBM_WATSON_CAFE + IBM_WATSON_KITCHEN) / 2.

MICROSOFT_LUIS_CAFE = np.array([0.83, 0.85, 0.89, 0.90, 0.91, 0.93, 0.93])
MICROSOFT_LUIS_KITCHEN = np.array([0.86, 0.89, 0.90, 0.91, 0.92, 0.93, 0.93])
MICROSOFT_LUIS = (MICROSOFT_LUIS_CAFE + MICROSOFT_LUIS_KITCHEN) / 2.

PICOVOICE_RHINO_CAFE = np.array([0.92, 0.96, 0.97, 0.98, 0.99, 0.99, 0.99])
PICOVOICE_RHINO_KITCHEN = np.array([0.95, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99])
PICOVOICE_RHINO = (PICOVOICE_RHINO_CAFE + PICOVOICE_RHINO_KITCHEN) / 2.

SNR_dB = [6, 9, 12, 15, 18, 21, 24]


def plot_detailed():
    plt.plot(SNR_dB, AMAZON_LEX, color='g', marker='^', label='Amazon Lex')
    plt.plot(SNR_dB, GOOGLE_DIALOGFLOW, color='r', marker='x', label='Google Dialogflow')
    plt.plot(SNR_dB, IBM_WATSON, color='k', marker='s', label='IBM Watson')
    plt.plot(SNR_dB, MICROSOFT_LUIS, color='m', marker='d', label='Microsoft LUIS')
    plt.plot(SNR_dB, PICOVOICE_RHINO, color='b', marker='o', label='Picovoice Rhino')
    plt.xlim(6, 24)
    plt.xlabel('SNR dB')
    plt.ylim(0.6, 1)
    plt.ylabel('Accuracy (Command Acceptance Probability)')
    plt.xticks([6, 9, 12, 15, 18, 21, 24])
    plt.legend()
    plt.title("Accuracy of NLU Engines")
    plt.grid()
    plt.show()


PV_COLOR = (55 / 255, 125 / 255, 255 / 255)
COLOR = (100 / 255, 100 / 255, 100 / 255)


def plot():
    fig, ax = plt.subplots()

    for spine in plt.gca().spines.values():
        if spine.spine_type != 'bottom':
            spine.set_visible(False)

    command_acceptance_rates = [
        GOOGLE_DIALOGFLOW.mean() * 100,
        AMAZON_LEX.mean() * 100,
        IBM_WATSON.mean() * 100,
        MICROSOFT_LUIS.mean() * 100,
        PICOVOICE_RHINO.mean() * 100
    ]

    ax.bar(np.arange(1, 5), command_acceptance_rates[:-1], 0.4, color=COLOR)
    ax.bar([5], [command_acceptance_rates[-1]], 0.4, color=PV_COLOR)

    for i in np.arange(4):
        ax.text(i + 1 - 0.2, int(command_acceptance_rates[i]) + 2, '%.1f%%' % command_acceptance_rates[i], color=COLOR)
    ax.text(5 - 0.2, int(command_acceptance_rates[4]) + 2, '%.1f%%' % command_acceptance_rates[4], color=PV_COLOR)

    plt.xticks(
        np.arange(1, 6),
        ['Google\nDialogflow', 'Amazon\nLex', 'IBM\nWatson', 'Microsoft\nLUIS', 'Picovoice\nRhino'])
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.title("Command Acceptance Rate\n(Averaged across various noisy environments)")
    plt.show()


if __name__ == '__main__':
    plot_detailed()

    plot()
