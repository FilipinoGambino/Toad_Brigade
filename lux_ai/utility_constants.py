import getpass

# from .lux.constants import Constants
# from .lux.game_constants import GAME_CONSTANTS

USER = getpass.getuser()
LOCAL_EVAL = USER in ['isaiah']

# Shorthand constants

# Derived constants
DAY_LEN = 30
CYCLE_LENGTH = 50
MAX_FACTORIES = 2
MAP_SIZE = (48, 48)
