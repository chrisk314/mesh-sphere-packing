import logging

TOL = 1.e-10                        # Floating point tolerance.
ONE_THIRD = 0.3333333333333333      # One third: nothing more, nothing less.
GROWTH_LIMIT = 5                    # Limits area of boundary triangles.
OVERLAP_TRIM_FACTOR = 0.995         # Reduces sphere radii to avoid overlaps.

LOG_FORMAT = '%(name)-8s  %(levelname)-8s %(asctime)s.%(msecs)03d  :    %(message)s'
TIME_STAMP_FORMAT = '%d-%m-%y %H:%M:%S'
formatter = logging.Formatter(LOG_FORMAT, TIME_STAMP_FORMAT)

logger = logging.getLogger('MSP-Build')
stream_h = logging.StreamHandler()
file_h = logging.FileHandler('msp.log', mode='w')
for handler in (stream_h, file_h):
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
