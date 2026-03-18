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
        """
        * method: batch_predict_from_model
        * description: method to predict results for batch data
        * return: none
        *
        * who             when           version  change (include bug# if apply)
        * ----------      -----------    -------  ------------------------------
        * VIVEK           11-MAR-2026    1.0      initial creation
        * VIVEK           18-MAR-2026    2.0      removed clustering, added scaler
        *
        * Parameters
        *   none:
        """
        try:
            self.logger.info('Start of Prediction')
            self.logger.info('run_id:' + str(self.run_id))

            # Validations and transformation
            self.loadValidate.validate_predictset()

            # Preprocessing activities
            self.X = self.preProcess.preprocess_predictset()

            # Load scaler and model
            scaler = self.fileOperation.load_model('scaler', base_path=self.base_path)
            model = self.fileOperation.load_model('best_model', base_path=self.base_path)

            # Store empid before scaling
            emp_ids = self.X['empid']
            features = self.X.drop(['empid'], axis=1)

            # Scale features
            features_scaled = scaler.transform(features)

            # Predict
            y_predicted = model.predict(features_scaled)

            # Save results
            import os
            result = pd.DataFrame({"EmpId": emp_ids, "Prediction": y_predicted})
            file_path = self.data_path + '_results/' + 'Predictions.csv'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            result.to_csv(file_path, header=True, index=False)

            self.logger.info('End of Prediction')
        except Exception:
            self.logger.exception('Unsuccessful End of Prediction')
            raise Exception

    def single_predict_from_model(self, data):
        """
        * method: single_predict_from_model
        * description: method to predict result for single employee
        * return: prediction (0 or 1)
        *
        * who             when           version  change (include bug# if apply)
        * ----------      -----------    -------  ------------------------------
        * VIVEK           11-MAR-2026    1.0      initial creation
        * VIVEK           18-MAR-2026    2.0      removed clustering, added scaler
        *
        * Parameters
        *   data:
        """
        try:
            self.logger.info('Start of Prediction')
            self.logger.info('run_id:' + str(self.run_id))

            # Preprocessing activities
            self.X = self.preProcess.preprocess_predict(data)

            # Load scaler and model
            scaler = self.fileOperation.load_model('scaler', base_path=self.base_path)
            model = self.fileOperation.load_model('best_model', base_path=self.base_path)

            # Scale features (drop empid for prediction)
            features = self.X.drop(['empid'], axis=1)
            features_scaled = scaler.transform(features)

            # Predict
            y_predicted = model.predict(features_scaled)

            self.logger.info('Output : ' + str(y_predicted))
            self.logger.info('End of Prediction')
            return int(y_predicted[0])
        except Exception:
            self.logger.exception('Unsuccessful End of Prediction')
            raise Exception
