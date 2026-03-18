import json
from os import listdir
import shutil
import pandas as pd
import os
from apps.core.logger import logging


class LoadValidate:
    """
    *****************************************************************************
    *
    * filename:       LoadValidate.py
    * version:        2.0
    * author:         VPM
    * creation date:  11-MAR-2026
    *
    * change history:
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * VIVEK           11-MAR-2026    1.0      initial creation
    * VIVEK           18-MAR-2026    2.0      removed SQLite database layer,
    *                                         removed archiving, direct CSV pipeline
    *
    *
    * description:    Class to load, validate and transform the data
    *
    ****************************************************************************
    """

    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = logging.getLogger('LoadValidate')

    def values_from_schema(self, schema_file):
        try:
            with open('apps/database/' + schema_file + '.json', 'r') as f:
                dic = json.load(f)
            column_names = dic['ColName']
            number_of_columns = dic['NumberofColumns']
            return column_names, number_of_columns
        except Exception as e:
            self.logger.exception('Failed to read schema: %s' % e)
            raise e

    def validate_column_length(self, number_of_columns):
        try:
            for file in listdir(self.data_path):
                csv = pd.read_csv(self.data_path + '/' + file)
                if csv.shape[1] != number_of_columns:
                    shutil.move(self.data_path + '/' + file, self.data_path + '_rejects')
                    self.logger.info("Rejected %s: expected %d columns, got %d" % (file, number_of_columns, csv.shape[1]))
        except Exception as e:
            self.logger.exception('Column length validation failed: %s' % e)
            raise e

    def validate_missing_values(self):
        try:
            for file in listdir(self.data_path):
                csv = pd.read_csv(self.data_path + '/' + file)
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        shutil.move(self.data_path + '/' + file, self.data_path + '_rejects')
                        self.logger.info("Rejected %s: column '%s' has all missing values" % (file, columns))
                        break
        except Exception as e:
            self.logger.exception('Missing values validation failed: %s' % e)
            raise e

    def merge_csv_to_inputfile(self):
        try:
            validation_dir = self.data_path + '_validation/'
            os.makedirs(validation_dir, exist_ok=True)

            all_dfs = []
            for file in listdir(self.data_path):
                df = pd.read_csv(self.data_path + '/' + file)
                all_dfs.append(df)

            merged_df = pd.concat(all_dfs, ignore_index=True)
            merged_df.to_csv(validation_dir + 'InputFile.csv', index=False)
            self.logger.info('Loaded %d rows from %d file(s)' % (len(merged_df), len(all_dfs)))
        except Exception as e:
            self.logger.exception('CSV merge failed: %s' % e)
            raise e

    def validate_trainset(self):
        try:
            self.logger.info('Validating training data...')
            column_names, number_of_columns = self.values_from_schema('schema_train')
            self.validate_column_length(number_of_columns)
            self.validate_missing_values()
            self.merge_csv_to_inputfile()
            self.logger.info('Training data validated successfully')
        except Exception:
            self.logger.exception('Training data validation failed')
            raise Exception

    def validate_predictset(self):
        try:
            self.logger.info('Validating prediction data...')
            column_names, number_of_columns = self.values_from_schema('schema_predict')
            self.validate_column_length(number_of_columns)
            self.validate_missing_values()
            self.merge_csv_to_inputfile()
            self.logger.info('Prediction data validated successfully')
        except Exception:
            self.logger.exception('Prediction data validation failed')
            raise Exception
