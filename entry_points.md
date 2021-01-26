All these shell scripts should be run from the top directory of the project

### Data preparation
This script:

Read training data from `RAW_DATA_DIR` (specified in `SETTINGS.json`)

Prepared data (it takes half an hour in average)

Saves processed data to `TRAIN_PREPARED_DATA_PATH` (needs 60GB of free disk space)

```bash
./scripts/prepare_data.sh
```

### Model training

Trains models using files from `TRAIN_PREPARED_DATA_PATH`

Saves provisional models to `CHECKPOINT_DIR` (needs ~10 GB of disk space)

Saves trained final models to `MODEL_DIR` (needs ~1GB of disk space)

```bash
./scripts/train_models.sh
```

### Making submission
Reads pairs of raw videos from `TEST_DATA_PATH`. Video file's name should be of this pattern `{game_id}_{Endzone|Sideline}.mp4`. And each video should have its counterpart from the other view.

Loads models from `MODEL_DIR` 

Saves predictions to `SUBMISSION_DIR`

```bash
./scripts/predict.sh
```