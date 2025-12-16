import pandas as pd
import gutenTAG.api as gt
import matplotlib.pyplot as plt

from gutenTAG import GutenTAG, TrainingType, LABEL_COLUMN_NAME

gutentag = GutenTAG()
gutentag.load_config_yaml('config.yaml')



datasets = gutentag.generate(return_timeseries=True)
import pandas as pd

data = pd.DataFrame(datasets[0])