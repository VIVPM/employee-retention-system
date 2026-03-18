import joblib
import os
import shutil
from apps.core.logger import logging


class FileOperation:
    """
    *****************************************************************************
    *
    * file_name:       FileOperation.py
    * version:        2.0
    * author:         VPM
    * creation date:  11-MAR-2026
    *
    * change history:
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * VIVEK           11-MAR-2026    1.0      initial creation
    * VIVEK           18-MAR-2026    2.0      cleaned up logging
    *
    *
    * description:    Class for file operation
    *
    ****************************************************************************
    """

    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = logging.getLogger('FileOperation')

    def save_model(self, model, file_name):
        try:
            path = os.path.join('apps/models/', file_name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

            joblib.dump(model, os.path.join(path, file_name + '.joblib'))
            self.logger.info('Saved: %s' % file_name)
            return 'success'
        except Exception as e:
            self.logger.exception('Failed to save model %s: %s' % (file_name, e))
            raise Exception()

    def load_model(self, file_name, base_path=None):
        try:
            if not base_path:
                raise ValueError("base_path must be provided for model loading")

            model_path = os.path.join(base_path, file_name, file_name + '.joblib')
            self.logger.info('Loaded: %s' % file_name)
            return joblib.load(model_path)
        except Exception as e:
            self.logger.exception('Failed to load model %s: %s' % (file_name, e))
            raise Exception()
