import pandas as pd
from apps.core.logger import logging
from apps.ingestion.load_validate import LoadValidate
from apps.preprocess.preprocessor import Preprocessor
from apps.core.file_operation import FileOperation


class PredictModel:
    """
    *****************************************************************************
    *
    * filename:       PredictModel.py
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
    *                                         single model prediction
    *
    *
    * description:    Class to prediction the result
    *
    ****************************************************************************
    """

    def __init__(self, run_id, data_path, base_path=None):
        self.run_id = run_id
        self.data_path = data_path
        self.base_path = base_path
        self.logger = logging.getLogger('PredictModel')
        self.loadValidate = LoadValidate(self.run_id, self.data_path, 'prediction')
        self.preProcess = Preprocessor(self.run_id, self.data_path, 'prediction')
        self.fileOperation = FileOperation(self.run_id, self.data_path, 'prediction')

    def batch_predict_from_model(self):
        try:
            self.logger.info('Batch prediction started (run_id: %s)' % str(self.run_id))

            self.loadValidate.validate_predictset()
            self.X = self.preProcess.preprocess_predictset()

            # Load scaler and model
            scaler = self.fileOperation.load_model('scaler', base_path=self.base_path)
            model = self.fileOperation.load_model('best_model', base_path=self.base_path)

            # Scale and predict
            emp_ids = self.X['empid']
            features = self.X.drop(['empid'], axis=1)
            features_scaled = scaler.transform(features)
            y_predicted = model.predict(features_scaled)

            # Save results
            import os
            result = pd.DataFrame({"EmpId": emp_ids, "Prediction": y_predicted})
            file_path = self.data_path + '_results/' + 'Predictions.csv'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            result.to_csv(file_path, header=True, index=False)

            self.logger.info('Batch prediction complete: %d predictions' % len(y_predicted))
        except Exception:
            self.logger.exception('Batch prediction failed')
            raise Exception

    def single_predict_from_model(self, data):
        try:
            self.X = self.preProcess.preprocess_predict(data)

            # Load scaler and model
            scaler = self.fileOperation.load_model('scaler', base_path=self.base_path)
            model = self.fileOperation.load_model('best_model', base_path=self.base_path)

            # Scale and predict
            features = self.X.drop(['empid'], axis=1)
            features_scaled = scaler.transform(features)
            y_predicted = model.predict(features_scaled)

            self.logger.info('Single prediction: %d' % y_predicted[0])
            return int(y_predicted[0])
        except Exception:
            self.logger.exception('Single prediction failed')
            raise Exception
