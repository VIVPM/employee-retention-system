from apps.core.logger import logging
import  json
from sklearn.model_selection import train_test_split
from apps.core.file_operation import FileOperation
from apps.tuning.model_tuner import ModelTuner
from apps.ingestion.load_validate import LoadValidate
from apps.preprocess.preprocessor import Preprocessor
from apps.tuning.cluster import KMeansCluster


class TrainModel:
    """
    *****************************************************************************
    *
    * filename:       TrainModel.py
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
    * description:    Class to training the models
    *
    ****************************************************************************
    """

    def __init__(self,run_id,data_path):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = logging.getLogger('TrainModel')
        self.loadValidate = LoadValidate(self.run_id, self.data_path,'training')
        self.preProcess = Preprocessor(self.run_id, self.data_path,'training')
        self.modelTuner = ModelTuner(self.run_id, self.data_path, 'training')
        self.fileOperation = FileOperation(self.run_id, self.data_path, 'training')
        self.cluster = KMeansCluster(self.run_id, self.data_path)

    def training_model(self):
        """
        * method: trainingModel
        * description: method to training the model
        * return: none
        *
        * who             when           version  change (include bug# if apply)
        * ----------      -----------    -------  ------------------------------
        * VIVEK           11-MAR-2026    1.0      initial creation
        *
        * Parameters
        *   none:
        """
        try:
            self.logger.info('Start of Training')
            self.logger.info('Run_id:' + str(self.run_id))
            #Load, validations and transformation
            self.loadValidate.validate_trainset()
            #preprocessing activities
            self.X, self.y = self.preProcess.preprocess_trainset()
            columns = {"data_columns":[col for col in self.X.columns]}
            with open('apps/database/columns.json','w') as f:
                f.write(json.dumps(columns))
            #create clusters
            number_of_clusters = self.cluster.elbow_plot(self.X)
            # Divide the data into clusters
            self.X= self.cluster.create_clusters(self.X, number_of_clusters)
            # create a new column in the dataset consisting of the corresponding cluster assignments.
            self.X['Labels'] = self.y
            # getting the unique clusters from our data set
            list_of_clusters = self.X['Cluster'].unique()
            
            results_list = [] # to store metrics for Hugging Face

            # parsing all the clusters and look for the best ML algorithm to fit on individual cluster
            for i in list_of_clusters:
                cluster_data=self.X[self.X['Cluster']==i] # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
                cluster_label= cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                try:
                    x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=0.2, random_state=0, stratify=cluster_label)
                except ValueError:
                    self.logger.info('Stratified split failed for cluster %s (too few samples in a class), falling back to regular split' % str(i))
                    x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=0.2, random_state=0)
                #getting the best model for each of the clusters
                best_model_name, best_model, all_results = self.modelTuner.get_best_model(x_train, y_train, x_test, y_test)

                #saving the best model to the directory.
                save_model = self.fileOperation.save_model(best_model, best_model_name+str(i))
                
                # Append ALL model results (RandomForest + DT + LR + SVM) for this cluster
                for r in all_results:
                    results_list.append({
                        'Cluster': i,
                        'Model Name': r['model_name'],
                        'Best Score Recall': r['score'],
                        'Best Score AUC': r.get('auc', 'N/A'),
                        'Best Parameters': str(r['params']),
                        'Selected': 'Yes' if r['model_name'] == best_model_name else 'No'
                    })

            import pandas as pd
            import os
            # Save the results to results.csv
            results_df = pd.DataFrame(results_list)
            os.makedirs('apps/models', exist_ok=True)
            results_df.to_csv('apps/models/results.csv', index=False)
            self.logger.info('Saved results.csv successfully with tuning metrics.')

            # Upload models to Hugging Face
            try:
                from apps.core.hf_uploader import HFUploader
                self.logger.info('Starting Hugging Face Upload...')
                uploader = HFUploader(logger=self.logger)
                if uploader.upload_models():
                    self.logger.info('Purging local apps/models after successful Hub upload...')
                    import shutil
                    shutil.rmtree('apps/models', ignore_errors=True)
            except Exception as e:
                self.logger.exception(f'Failed to upload to Hugging Face Hub: {str(e)}')

            self.logger.info('End of Training')
            self.logger.info('TRAINING_COMPLETE')
        except Exception:
            self.logger.exception('Unsuccessful End of Training')
            raise Exception