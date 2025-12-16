DATA_PATH = '/Users/hxh/PycharmProjects/wavelet/wavelet/Dataset/'

DATA_NAMES = ['AbnormalHeartbeat', 'Computers', 'Earthquakes',
              'ECG5000', 'PowerCons', 'StarLightCurves',
              'WormsTwoClass', 'ECG5000', 'EOGVerticalSignal',
              'Wafer','ProximalPhalanxOutlineCorrect','ECG200']

DUAL_FEATURES_MODELS = ["DualFeatureMLP", "DualFeatureMRAClassifier"]

DATA_INFO = {
    DATA_NAMES[0]: {
        'SEQUENCE_LENGTH': 18530,
        'NUM_CLASSES': 2,
    },
    DATA_NAMES[1]: {
        'SEQUENCE_LENGTH': 720,
        'NUM_CLASSES': 2,
    },
    DATA_NAMES[2]: {
        'SEQUENCE_LENGTH': 512,
        'NUM_CLASSES': 2,
    },
    DATA_NAMES[3]: {
        'SEQUENCE_LENGTH': 140,
        'NUM_CLASSES': 5,
    },
    DATA_NAMES[4]: {
        'SEQUENCE_LENGTH': 144,
        'NUM_CLASSES': 2,
    },
    DATA_NAMES[5]: {
        'SEQUENCE_LENGTH': 1024,
        'NUM_CLASSES': 3,
    },
    DATA_NAMES[6]: {
        'SEQUENCE_LENGTH': 900,
        'NUM_CLASSES': 2,
    },
    DATA_NAMES[7]: {
        'SEQUENCE_LENGTH': 140,
        'NUM_CLASSES': 5,
    },
    DATA_NAMES[8]: {
        'SEQUENCE_LENGTH': 1250,
        'NUM_CLASSES': 12,
    },
    DATA_NAMES[9]: {
        'SEQUENCE_LENGTH': 152,
        'NUM_CLASSES': 2,
    },
    DATA_NAMES[10]: {
        'SEQUENCE_LENGTH': 80,
        'NUM_CLASSES': 2,
    },
    DATA_NAMES[11]: {
        'SEQUENCE_LENGTH': 96,
        'NUM_CLASSES': 2,
    }
}