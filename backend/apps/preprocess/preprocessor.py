import pandas as pd
import numpy as np
import json
from sklearn.impute import KNNImputer
from apps.core.logger import logging


class Preprocessor:
    """
    *****************************************************************************
    *
    * filename:       Preprocessor.py
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
    * description:    Class to pre-process training and predict dataset
    *
    ****************************************************************************
    """

    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = logging.getLogger('Preprocessor')

    def get_data(self):
        try:
            self.data = pd.read_csv(self.data_path + '_validation/InputFile.csv')
            self.logger.info('Loaded dataset: %d rows, %d columns' % (self.data.shape[0], self.data.shape[1]))
            return self.data
        except Exception as e:
            self.logger.exception('Failed to read dataset: %s' % e)
            raise Exception()

    def drop_columns(self, data, columns):
        try:
            self.useful_data = data.drop(labels=columns, axis=1)
            return self.useful_data
        except Exception as e:
            self.logger.exception('Failed to drop columns %s: %s' % (columns, e))
            raise Exception()

    def is_null_present(self, data):
        self.null_present = False
        try:
            self.null_counts = data.isna().sum()
            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break
            if self.null_present:
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv(self.data_path + '_validation/' + 'null_values.csv')
                self.logger.info('Missing values found: %d columns affected' % self.null_counts.astype(bool).sum())
            return self.null_present
        except Exception as e:
            self.logger.exception('Failed to check missing values: %s' % e)
            raise Exception()

    def impute_missing_values(self, data):
        try:
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
            self.new_array = imputer.fit_transform(data)
            self.new_data = pd.DataFrame(data=self.new_array, columns=data.columns)
            self.logger.info('Imputed missing values with KNNImputer')
            return self.new_data
        except Exception as e:
            self.logger.exception('Failed to impute missing values: %s' % e)
            raise Exception()

    def feature_encoding(self, data):
        try:
            self.new_data = data.select_dtypes(include=['object']).copy()
            for col in self.new_data.columns:
                self.new_data = pd.get_dummies(self.new_data, columns=[col], prefix=[col], drop_first=True)

            # Pandas 3.x returns bool dtype from get_dummies; cast to int for KNNImputer compatibility
            bool_cols = self.new_data.select_dtypes(include='bool').columns
            self.new_data[bool_cols] = self.new_data[bool_cols].astype(int)

            return self.new_data
        except Exception as e:
            self.logger.exception('Failed to encode features: %s' % e)
            raise Exception()

    def split_features_label(self, data, label_name):
        try:
            self.X = data.drop(labels=label_name, axis=1)
            self.y = data[label_name]
            return self.X, self.y
        except Exception as e:
            self.logger.exception('Failed to split features and label: %s' % e)
            raise Exception()

    def final_predictset(self, data):
        try:
            with open('apps/database/columns.json', 'r') as f:
                data_columns = json.load(f)['data_columns']
            df = pd.DataFrame(data=None, columns=data_columns)
            df_new = pd.concat([df, data], ignore_index=True, sort=False)
            data_new = df_new.fillna(0)
            # Cast all to numeric so sklearn doesn't reject them
            for col in data_new.columns:
                data_new[col] = pd.to_numeric(data_new[col], errors='coerce').fillna(0)
            return data_new
        except Exception as e:
            self.logger.exception('Failed to build final predictset: %s' % e)
            raise e

    def preprocess_trainset(self):
        try:
            self.logger.info('Preprocessing training data...')
            data = self.get_data()
            data = self.drop_columns(data, ['empid'])
            cat_df = self.feature_encoding(data)
            data = pd.concat([data, cat_df], axis=1)
            data = self.drop_columns(data, ['salary'])
            is_null_present = self.is_null_present(data)
            if is_null_present:
                data = self.impute_missing_values(data)
            self.X, self.y = self.split_features_label(data, label_name='left')
            self.logger.info('Preprocessing complete: %d features, %d samples' % (self.X.shape[1], self.X.shape[0]))
            return self.X, self.y
        except Exception:
            self.logger.exception('Preprocessing failed')
            raise Exception

    def preprocess_predictset(self):
        try:
            self.logger.info('Preprocessing prediction data...')
            data = self.get_data()
            cat_df = self.feature_encoding(data)
            data = pd.concat([data, cat_df], axis=1)
            data = self.drop_columns(data, ['salary'])
            is_null_present = self.is_null_present(data)
            if is_null_present:
                data = self.impute_missing_values(data)
            data = self.final_predictset(data)
            self.logger.info('Preprocessing complete: %d rows' % len(data))
            return data
        except Exception:
            self.logger.exception('Preprocessing failed')
            raise Exception

    def preprocess_predict(self, data):
        try:
            # Manual encoding for single row (get_dummies drops the only category with drop_first=True)
            salary_val = data['salary'].iloc[0]
            data = data.copy()
            data['salary_low'] = 1 if salary_val == 'low' else 0
            data['salary_medium'] = 1 if salary_val == 'medium' else 0
            data = self.drop_columns(data, ['salary'])
            is_null_present = self.is_null_present(data)
            if is_null_present:
                data = self.impute_missing_values(data)
            data = self.final_predictset(data)
            return data
        except Exception:
            self.logger.exception('Single prediction preprocessing failed')
            raise Exception
