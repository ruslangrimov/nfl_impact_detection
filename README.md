Below you can find a outline of how to reproduce my solution for the "NFL 1st and Future - Impact Detection" competition.

# ARCHIVE CONTENTS
lib	: core source code

lib/scripts: 	python scripts to run data preparation, training, submitting etc. You needn't run them manually

lib/scripts/configs:	various configuration files

models	: trained models

checkpoints	: provisional files saved during training

scripts	: shell scripts to run data preparation, training, submitting. This scripts run corresponding python scripts with proper parameters. See `entry_points.md` for more info.

submissions	: this is the place where the script `./scripts/predict.sh` saves submissions

logs	: logs directory

tmp	: temporary directory

directory_structure.txt	:	contains directory structure of the project

entry_points.md	: entry points to run data preparation, training, submitting

ModelSummary.odt	: summary of my solution. It is also available on [kaggle forum](https://www.kaggle.com/c/nfl-impact-detection/discussion/209235)

README.md	: 

requirements.txt	: requriments for pip

settings.json	: main configuration file with paths to essential directories

# HARDWARE: (The following specs were used to create the original solution)

Ubuntu 18.04.5 LTS

Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz, 32 GB memory

2 x NVIDIA 1080 Ti


# SOFTWARE (python packages are detailed separately in `requirements.txt`):

Python 3.8.5

CUDA 10.2

cuddn 7.6.5_0

nvidia drivers v440.118.02

If you use docker image like [pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch/) you may need to install this software:

`./scripts/install_other_soft.sh`

After installation of packages from requirements.txt **you need to install** [SlowFast](https://github.com/facebookresearch/SlowFast) library running script:
`./scripts/install_slowfast.sh` from the top directory.

# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)

```bash
mkdir -p /mnt/SSDData/AF/
cd /mnt/SSDData/AF/
kaggle competitions download -c nfl-impact-detection
unzip nfl-impact-detection.zip
```

# DATA PROCESSING
Add path to the unzipped files' directory into `./settings.json` as `RAW_DATA_DIR` 

Run `./scripts/prepare_data.sh`. It takes half an hour to process the data and needs 60GB of free disk space in the directory `TRAIN_PREPARED_DATA_PATH`.

# MODEL BUILD
1. very fast prediction
    - there is no very fast prediction in my solution
2. ordinary prediction
    - expect this to run for 4-6 hours for 15 pairs of videos
    - uses binary model files
3. retrain models
    - expect this to run about tho days
    - trains all models from scratch
    - follow this with (2) to produce entire solution from scratch

shell command to run each build is below:

1. None

2. ordinary prediction (uses pairs of raw videos from `TEST_DATA_PATH` and overwrites predictions in `./submissions` directory) 

```bash
./scripts/predict.sh
```

3. retrain models (overwrites models in `./checkpoints` and `./models` directory) 

```bash
./scripts/train_models.sh
```
