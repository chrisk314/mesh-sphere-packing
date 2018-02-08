
import logging

TOL = 1.e-10                        # Floating point tolerance
ONE_THIRD = 0.3333333333333333      # One third: nothing more, nothing less.
GROWTH_LIMIT = 5                    # Limits area of boundary triangles

LOG_FORMAT = '%(asctime)s %(name)-12s : %(levelname)-8s  %(message)s'
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
