
# Some constants
WINDOW_HEIGHT = 588
GRID_SIZE = WINDOW_HEIGHT
SIDEPANEL_WIDTH = 250
SIDEPANEL_OFFSET = GRID_SIZE
WINDOW_WIDTH = GRID_SIZE + SIDEPANEL_WIDTH
NCELLS = 28
IMG_SIZE = NCELLS * NCELLS
CELL_SIZE = GRID_SIZE / NCELLS

N_EXAMPLES = 100

# Colors
BLACK       = (  0,   0,   0)
GRAY        = ( 50,  50,  50)
LIGHT_GRAY  = (100, 100, 100)
WHITE       = (255, 255, 255)
RED         = (255,   0,   0)
GREEN       = (  0, 255,   0)
BLUE        = (  0,   0, 255)


# Custom event types
EVENT_UNDEFINED_ON_CLICK = 1
EVENT_CLEAR_GRID = 2
EVENT_LOAD_EXAMPLE = 3


# Cell types
C_EMPTY = 0
C_FULL = 1


# Paths
DATA_DIR = "../../datasets/"
TRAIN_IMAGES = '''train-images-idx3-ubyte.gz'''
TRAIN_LABELS = '''train-labels-idx1-ubyte.gz'''
TEST_IMAGES = '''t10k-images-idx3-ubyte.gz'''
TEST_LABELS = '''t10k-labels-idx1-ubyte.gz'''

MODEL_DIR = "../models/"
MLPMODEL = "MLPModel.pkl"

