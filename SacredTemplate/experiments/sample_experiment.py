import os
import random
import sys

#import pandas as pd

sys.path.append("../src")
from utils.experiment import Bunch, make_experiment, make_experiment_tempfile


if __name__ == '__main__':
    experiment = make_experiment()


    @experiment.config
    def config():
        params = dict(
            steps=20,
            range=(10, 15)
        )


    @experiment.automain
    def main(params, _run):
        """
        Sample experiment that generates random numbers in a range for a specified number of steps
        """
        # Convert dict to bunch to access parameters in dot notation
        #print(params) #{'steps': 20, 'range': [10, 15]}
        params = Bunch(params)
        #print(params) #<utils.experiment.Bunch object at 0x0000014E04DD67D0>
        low, high = params.range
        #print(params.range) #[10, 15]
        #print(params.steps) #20

        print('First we print something that will appear in the log!')

        for step in range(params.steps):
            randnum = low + random.random() * (high - low)
            # Log scalar wil log a single number. The string is the metrics name
            _run.log_scalar('random_metric', randnum)
            _run.log_scalar('random_metric_squared', randnum**2)

        # We can save files either in the Directory or database using the make_experiment_tempfile method
        with make_experiment_tempfile('test.txt', _run, mode='w') as f:
            f.write('Hello World!')

        # If we open a resource (for example a data file) with open_resource, this will also be logged by sacred
        # And that file is saved in the database or log directory
        with _run.open_resource('../omniboard/docker-compose.yml', mode='r') as f:
            # We do nothing for now
            pass

        print('Done!')