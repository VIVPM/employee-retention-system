from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics  import roc_auc_score,accuracy_score
from apps.core.logger import logging

class ModelTuner:
    """
    *****************************************************************************
    *
    * filename:       model_tuner.py
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
    * description:    Class to tune and select best model
    *
    ****************************************************************************
    """
    def __init__(self,run_id,data_path,mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = logging.getLogger('ModelTuner')
        self.rfc = RandomForestClassifier()
        self.dt = DecisionTreeClassifier()
        self.lr = LogisticRegression()
        self.sv = SVC(probability=True)

    def best_params_randomforest(self,train_x,train_y):
        """
        * method: best_params_randomforest
        * description: method to get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Parameters are fixed to optimize training speed.
        * return: The model with the best parameters
        *
        * who             when           version  change (include bug# if apply)
        * ----------      -----------    -------  ------------------------------
        * VIVEK           11-MAR-2026    1.1      fixed parameters for speed
        *
        * Parameters
        *   train_x:
        *   train_y:
        """
        try:
            self.logger.info('Start of building randomforest with fixed params...')
            # Using fixed parameters provided by user for optimization
            self.criterion = 'gini'
            self.max_depth = 3
            self.max_features = 'sqrt'
            self.n_estimators = 10

            #creating a new model with the best parameters
            self.rfc = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features,
                                              class_weight='balanced')
            # training the new model
            self.rfc.fit(train_x, train_y)
            self.rf_best_params = {'criterion': self.criterion, 'max_depth': self.max_depth, 'max_features': self.max_features, 'n_estimators': self.n_estimators}
            self.logger.info('Random Forest used fixed params: ' + str(self.rf_best_params))
            self.logger.info('End of building randomforest with fixed params...')

            return self.rfc
        except Exception as e:
            self.logger.exception('Exception raised while building randomforest:' + str(e))
            raise Exception()

    def best_params_decisiontree(self,train_x,train_y):
        try:
            self.logger.info('Start of finding best params for Decision Tree algo...')
            self.param_grid_dt = {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': range(2, 4, 1),
                'max_features': ['sqrt', 'log2']
            }
            self.grid = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), self.param_grid_dt, cv=5, scoring='roc_auc')
            self.grid.fit(train_x, train_y)

            self.criterion = self.grid.best_params_['criterion']
            self.splitter = self.grid.best_params_['splitter']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']

            self.dt = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter,
                                             max_depth=self.max_depth, max_features=self.max_features,
                                             class_weight='balanced')
            self.dt.fit(train_x, train_y)
            self.dt_best_params = self.grid.best_params_
            self.logger.info('Decision Tree best params: ' + str(self.grid.best_params_))
            return self.dt
        except Exception as e:
            self.logger.exception('Exception raised while finding best params for Decision Tree algo:' + str(e))
            raise Exception()

    def best_params_logistic_regression(self,train_x,train_y):
        try:
            self.logger.info('Start of finding best params for Logistic Regression algo...')
            self.param_grid_lr = {
                'C': [0.1, 0.5, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
            self.grid = GridSearchCV(LogisticRegression(class_weight='balanced'), self.param_grid_lr, cv=5, scoring='roc_auc')
            self.grid.fit(train_x, train_y)

            self.C = self.grid.best_params_['C']
            self.penalty = self.grid.best_params_['penalty']
            self.solver = self.grid.best_params_['solver']

            self.lr = LogisticRegression(C=self.C, penalty=self.penalty, solver=self.solver, class_weight='balanced')
            self.lr.fit(train_x, train_y)
            self.lr_best_params = self.grid.best_params_
            self.logger.info('Logistic Regression best params: ' + str(self.grid.best_params_))
            return self.lr
        except Exception as e:
            self.logger.exception('Exception raised while finding best params for Logistic Regression algo:' + str(e))
            raise Exception()

    def best_params_svm(self,train_x,train_y):
        try:
            self.logger.info('Start of finding best params for SVM algo...')
            self.param_grid_svm = {
                'C': [0.1, 0.5, 1, 10],
                'kernel': ['linear'],
                'gamma': ['scale', 'auto']
            }
            self.grid = GridSearchCV(SVC(probability=True, class_weight='balanced'), self.param_grid_svm, cv=5, scoring='roc_auc')
            self.grid.fit(train_x, train_y)

            self.C = self.grid.best_params_['C']
            self.kernel = self.grid.best_params_['kernel']
            self.gamma = self.grid.best_params_['gamma']

            self.sv = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=True, class_weight='balanced')
            self.sv.fit(train_x, train_y)
            self.sv_best_params = self.grid.best_params_
            self.logger.info('SVM best params: ' + str(self.grid.best_params_))
            return self.sv
        except Exception as e:
            self.logger.exception('Exception raised while finding best params for SVM algo:' + str(e))
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
        * method: get_best_model
        * description: method to get best model (Optimized: Random Forest only)
        * return: none
        *
        * who             when           version  change (include bug# if apply)
        * ----------      -----------    -------  ------------------------------
        * VIVEK           11-MAR-2026    1.1      optimized for speed (RF only)
        *
        * Parameters
        *   train_x:
        *   train_y:
        *   test_x:
        *   test_y:
        """
        try:
            self.logger.info('Start of finding best model (Optimized)...')

            # We bypass searching and directly use the expert parameters for Random Forest
            self.random_forest = self.best_params_randomforest(train_x, train_y)
            self.rf_score = roc_auc_score(test_y, self.random_forest.predict_proba(test_x)[:, 1])
            self.logger.info('AUC for Random Forest: ' + str(self.rf_score))

            self.logger.info('End of model selection (Optimization enabled: RF only).')
            
            best_model_name = 'RandomForest'
            all_results = [
                {'model_name': 'RandomForest', 'score': self.rf_score, 'params': self.rf_best_params}
            ]
            
            return best_model_name, self.random_forest, all_results

        except Exception as e:
            self.logger.exception('Exception raised while finding best model:' + str(e))
            raise Exception()