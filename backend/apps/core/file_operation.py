import pickle
import os
import shutil
from apps.core.logger import logging

class FileOperation:
    """
    *****************************************************************************
    *
    * file_name:       FileOperation.py
    * version:        1.0
    * author:         VPM
    * creation date:  11-MAR-2026
    *
    * change history:
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * VIVEK           11-MAR-2026    1.0      initial creation
    *
    *
    * description:    Class for file operation
    *
    ****************************************************************************
    """

    def __init__(self,run_id,data_path,mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = logging.getLogger('FileOperation')

    def save_model(self,model,file_name):
        """
        * method: save_model
        * description: method to save the model file
        * return: File gets saved
        *
        * who             when           version  change (include bug# if apply)
        * ----------      -----------    -------  ------------------------------
        * VIVEK           11-MAR-2026    1.0      initial creation
        *
        * Parameters
        *   model:
        *   file_name:
        """
        try:
            self.logger.info('Start of Save Models')
            # Models are saved during training to apps/models before upload
            path = os.path.join('apps/models/',file_name) 
            if os.path.isdir(path): 
                shutil.rmtree(path) # Just remove the specific model folder
            os.makedirs(path, exist_ok=True)
            
            with open(os.path.join(path, file_name + '.sav'), 'wb') as f:
                pickle.dump(model, f) 
            self.logger.info('Model File '+file_name+' saved to apps/models/')
            self.logger.info('End of Save Models')
            return 'success'
        except Exception as e:
            self.logger.exception('Exception raised while Save Models: %s' % e)
            raise Exception()

    def load_model(self, file_name, base_path=None):
        """
        * method: load_model
        * description: method to load the model file
        * return: Model object
        *
        * who             when           version  change (include bug# if apply)
        * ----------      -----------    -------  ------------------------------
        * VIVEK           11-MAR-2026    1.0      initial creation
        *
        * Parameters
        *   file_name:
        *   base_path: (optional) The directory to load models from
        """
        try:
            self.logger.info('Start of Load Model')
            # If no base_path is provided, we fail explicitly to avoid 'ghost' local loads
            if not base_path:
                raise ValueError("base_path must be provided for model loading (Hub-centric)")
            
            load_dir = base_path
            model_path = os.path.join(load_dir, file_name, file_name + '.sav')
            
            with open(model_path, 'rb') as f:
                self.logger.info('Model File ' + file_name + ' loaded from ' + load_dir)
                self.logger.info('End of Load Model')
                return pickle.load(f)
        except Exception as e:
            self.logger.exception('Exception raised while Loading Model: %s' % e)
            raise Exception()

    def correct_model(self, cluster_number, base_path=None):
        try:
            self.logger.info('Start of finding correct model')
            if not base_path:
                raise ValueError("base_path must be provided for model selection (Hub-centric)")
                
            self.cluster_number = cluster_number
            self.folder_name = base_path
            self.list_of_model_files = []
            self.list_of_files = os.listdir(self.folder_name)
            for self.file in self.list_of_files:
                try:
                    if (self.file.index(str( self.cluster_number))!=-1):
                        self.model_name=self.file
                except:
                    continue
            self.model_name=self.model_name.split('.')[0]
            self.logger.info('End of finding correct model from ' + base_path)
            return self.model_name
        except Exception as e:
            self.logger.info('Exception raised while finding correct model' + str(e))
            raise Exception()