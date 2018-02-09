
import logging

TOL = 1.e-10                        # Floating point tolerance
ONE_THIRD = 0.3333333333333333      # One third: nothing more, nothing less.
GROWTH_LIMIT = 5                    # Limits area of boundary triangles

LOG_FORMAT = '%(name)-8s  %(levelname)-8s %(asctime)s.%(msecs)03d  :    %(message)s'
TIME_STAMP_FORMAT = '%d-%m-%y %H:%M:%S'
logger = logging.getLogger('MSP-Build')
handler = logging.StreamHandler()
formatter = logging.Formatter(LOG_FORMAT, TIME_STAMP_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
