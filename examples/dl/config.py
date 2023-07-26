WINDOW_SIZE = 8192
TRAINING_STEP = 4096
TESTING_STEP = 8192
RANDOM_SEED = 46

EPOCH = 1
PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

NUM_PROC_WORKERS_DATA = 1

def get_dict():
    return {
        "WINDOW_SIZE": WINDOW_SIZE,
        "TRAINING_STEP": TRAINING_STEP,
        "TESTING_STEP": TESTING_STEP,
        "RANDOM_SEED": RANDOM_SEED,
        "EPOCH": EPOCH,
        "PATIENCE": PATIENCE,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_PROC_WORKERS": NUM_PROC_WORKERS_DATA
    }