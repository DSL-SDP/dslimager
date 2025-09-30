"""Example Map-making task.

Inheritance diagram
-------------------

.. inheritance-diagram:: MapMaking
   :parts: 2

"""

from caput import mpiutil
from dslpipe.pipeline.pipeline import OneAndOne
from dslpipe.utils.logging import get_logger

logger = get_logger(__name__)


class MapMaking(OneAndOne):
    """Example Map-making task.

    """

    params_init = {
                    'algorithm': 'algorithm1', # algorithm used for this analysis
                  }

    prefix = 'mm_'

    def process(self, input):

        algorithm = self.params['algorithm']

        if mpiutil.rank0:
            logger.info(f'Executing map-making task with algorithm: {algorithm}')

        return super().process(input)

    def read_input(self):
        """Override to implement reading inputs from disk."""

        return self.params['input_files']
