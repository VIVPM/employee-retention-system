import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from apps.core.logger import logging
from apps.core.file_operation import FileOperation
from apps.tuning.model_tuner import ModelTuner
from apps.ingestion.load_validate import LoadValidate
from apps.preprocess.preprocessor import Preprocessor


class TrainModel:
    """
    *****************************************************************************
    *
    * filename:       TrainModel.py
    * version:        2.0
    * author:         VPM
    * creation date:  11-MAR-2026
    *
    * change history:
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * VIVEK           11-MAR-2026    1.0      initial creation
    * VIVEK           18-MAR-2026    2.0      removed clustering, added scaler,
    *                                         single best model with class_weight=balanced
    *
    *
    * description:    Class to training the models
    *
    ****************************************************************************
    """

    def __init__(self, run_id, data_path):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = logging.getLogger('TrainModel')
        self.loadValidate = LoadValidate(self.run_id, self.data_path, 'training')
        self.preProcess = Preprocessor(self.run_id, self.data_path, 'training')
        self.modelTuner = ModelTuner(self.run_id, self.data_path, 'training')
        self.fileOperation = FileOperation(self.run_id, self.data_path, 'training')

    def training_model(self):
        try:
            self.logger.info('Training started (run_id: %s)' % str(self.run_id))

            # Load, validate and transform
            self.loadValidate.validate_trainset()

            # Preprocessing
            self.X, self.y = self.preProcess.preprocess_trainset()

            # Save column names for prediction alignment
            columns = {"data_columns": [col for col in self.X.columns]}
            with open('apps/database/columns.json', 'w') as f:
                f.write(json.dumps(columns))

            # Train-test split (80/20, stratified)
            x_train, x_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
            )
            self.logger.info('Train: %d samples, Test: %d samples' % (len(x_train), len(x_test)))

            # Scale features
            scaler = MinMaxScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            # Train RandomForest with best params
            best_model, metrics = self.modelTuner.train_best_model(
                x_train_scaled, y_train, x_test_scaled, y_test
            )

            # Save model, scaler, and results
            os.makedirs('apps/models', exist_ok=True)
            self.fileOperation.save_model(best_model, 'best_model')
            self.fileOperation.save_model(scaler, 'scaler')

            results_df = pd.DataFrame([metrics])
            results_df.to_csv('apps/models/results.csv', index=False)
            self.logger.info('Results saved to results.csv')

            # Upload to Hugging Face
            try:
                from apps.core.hf_uploader import HFUploader
                uploader = HFUploader(logger=self.logger)
                if uploader.upload_models():
                    self.logger.info('Models uploaded to Hugging Face Hub')
                    import shutil
                    shutil.rmtree('apps/models', ignore_errors=True)
            except Exception as e:
                self.logger.exception('HuggingFace upload failed: %s' % str(e))

            self.logger.info('Training complete')
            self.logger.info('TRAINING_COMPLETE')
        except Exception:
            self.logger.exception('Training failed')
            raise Exception
