from Dataset.sythetic_data import get_gutenTAG_loaders
from gutenTAG import GutenTAG
from train_eval import *
import neptune
import os 
from model import *

run = neptune.init_run(
    project="casestudy",
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    mode="debug"
)

gutentag = GutenTAG()
gutentag.load_config_yaml('Dataset/sythetic_data/config.yaml')
datasets = gutentag.generate(return_timeseries=True)
seq_length = 50
batch_size = 256
train_ratio = 0.8
epoch = 10
lr = 1e-3
train_loaders, test_loaders = get_gutenTAG_loaders(datasets, seq_length, batch_size, train_ratio, stride=20)

model_params = {
    'filter_length': 4,
    'levels': 3,
    'signal_length': seq_length,
    'num_classes': 2,
    'hidden_dim': 64,
    'num_wavelets': 4,
    'init_type': 'random',
}

training_params = {
    'model_class':MultiWaveletClassifier,
    'model_params':model_params,
    'train_path':TRAIN_FILE,
    'test_path':TEST_FILE,
    'trainer':train_epoch,
    'custom_dataloader':(),
    'evaluator':evaluate_model,
    'num_epochs':epoch,
    'learning_rate':lr,
    'batch_size':batch_size,
    'run':run,
    'verbose':True,
}

best_acc_ls = []

for idx in range(len(train_loaders)):

    run = neptune.init_run(
        project="casestudy",
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        mode="debug"
    )

    training_params['custom_dataloader'] =(train_loaders[idx],test_loaders[idx])
    print(len(train_loaders[idx]))
    _, best_acc = main_train_and_test_generic(**training_params)
    best_acc_ls.append(best_acc)
    run.stop()

print(best_acc_ls)