BLANK_INIT = True

#================== DATASET
DATA_FILE = '../SymCheck/simple_context/data/75dis_100w_data'
LABEL_FILE = '../SymCheck/simple_context/data/75dis_100w_label'
INIT_FILE = '../SymCheck/simple_context/data/75dis_100w_init'

DATA_VAL_FILE = '../SymCheck/simple_context/data/75dis_small_test_noise_data'
LABEL_VAL_FILE = '../SymCheck/simple_context/data/75dis_small_test_noise_label'
INIT_VAL_FILE = '../SymCheck/simple_context/data/75dis_small_test_noise_init'


CLASSES = 75
FEATURE_DIM = 244
CONTEXT_DIM = 10
STATE_DIM = FEATURE_DIM + CONTEXT_DIM
ACTION_DIM = FEATURE_DIM + CLASSES
HIDDEN_DIM = [2048, 1024, 1024]
 
#================== RL
FEATURE_FACTOR   =   0.005
REWARD_CORRECT   =   0
REWARD_INCORRECT =  -1

#================== TRAINING
AGENTS = 1000
TRAINING_EPOCHS = 2000000
EPOCH_STEPS = 1

EPSILON_START  = 1.00
EPSILON_END    = 0.1
EPSILON_EPOCHS = 100000	 	# epsilon will fall to EPSILON_END after EPSILON_EPOCHS
EPSILON_UPDATE_EPOCHS = 100  # update epsilon every x epochs

#================== LOG
LOG_PERF_EPOCHS = 100

#================== NN
BATCH_SIZE =    100000
POOL_SIZE  =   2000000

OPT_LR = 5.0e-5
OPT_ALPHA = 0.95
OPT_MAX_NORM = 1.0

# LR scheduling => lower LR by LR_SC_FACTOR every LR_SC_EPOCHS epochs
LR_SC_FACTOR =  0.5
LR_SC_EPOCHS = 10000
LR_SC_MIN = 1.0e-7

TARGET_RHO = 0.01

#================== AUX
SAVE_EPOCHS = 1000
MAX_MASK_CONST = 1.e6

SEED = 1126


#================== PLOT
PLOT_ENV = 'ddqn'