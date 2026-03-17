from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics  import roc_auc_score,accuracy_score,recall_score
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
        try:
            self.logger.info('Start of finding best params for Random Forest algo...')
            self.param_grid_rf = {
                'n_estimators': [10, 50, 100],
                'criterion': ['gini', 'entropy'],
                'max_depth': range(2, 4, 1),
                'max_features': ['sqrt', 'log2']
            }
            self.grid = GridSearchCV(RandomForestClassifier(class_weight='balanced'), self.param_grid_rf, cv=5, scoring='recall')
            self.grid.fit(train_x, train_y)

            self.rfc = RandomForestClassifier(
                n_estimators=self.grid.best_params_['n_estimators'],
                criterion=self.grid.best_params_['criterion'],
                max_depth=self.grid.best_params_['max_depth'],
                max_features=self.grid.best_params_['max_features'],
                class_weight='balanced'
            )
            self.rfc.fit(train_x, train_y)
            self.rf_best_params = self.grid.best_params_
            self.logger.info('Random Forest best params: ' + str(self.grid.best_params_))
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
            self.grid = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), self.param_grid_dt, cv=5, scoring='recall')
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
            self.grid = GridSearchCV(LogisticRegression(class_weight='balanced'), self.param_grid_lr, cv=5, scoring='recall')
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
            self.grid = GridSearchCV(SVC(probability=True, class_weight='balanced'), self.param_grid_svm, cv=5, scoring='recall')
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
        try:
            self.logger.info('Start of finding best model using Recall...')

            # Train all 4 models with GridSearchCV (scoring=recall)
            self.random_forest = self.best_params_randomforest(train_x, train_y)
            self.rf_recall = recall_score(test_y, self.random_forest.predict(test_x))
            self.rf_auc = roc_auc_score(test_y, self.random_forest.predict_proba(test_x)[:, 1])
            self.logger.info('Random Forest - Recall: ' + str(self.rf_recall) + ' | AUC: ' + str(self.rf_auc))

            self.decision_tree = self.best_params_decisiontree(train_x, train_y)
            self.dt_recall = recall_score(test_y, self.decision_tree.predict(test_x))
            self.dt_auc = roc_auc_score(test_y, self.decision_tree.predict_proba(test_x)[:, 1])
            self.logger.info('Decision Tree - Recall: ' + str(self.dt_recall) + ' | AUC: ' + str(self.dt_auc))

            self.logistic_reg = self.best_params_logistic_regression(train_x, train_y)
            self.lr_recall = recall_score(test_y, self.logistic_reg.predict(test_x))
            self.lr_auc = roc_auc_score(test_y, self.logistic_reg.predict_proba(test_x)[:, 1])
            self.logger.info('Logistic Regression - Recall: ' + str(self.lr_recall) + ' | AUC: ' + str(self.lr_auc))

            self.svm = self.best_params_svm(train_x, train_y)
            self.svm_recall = recall_score(test_y, self.svm.predict(test_x))
            self.svm_auc = roc_auc_score(test_y, self.svm.predict_proba(test_x)[:, 1])
            self.logger.info('SVM - Recall: ' + str(self.svm_recall) + ' | AUC: ' + str(self.svm_auc))

            # Pick best model by recall
            models = {
                'RandomForest': (self.random_forest, self.rf_recall, self.rf_auc, self.rf_best_params),
                'DecisionTree': (self.decision_tree, self.dt_recall, self.dt_auc, self.dt_best_params),
                'LogisticRegression': (self.logistic_reg, self.lr_recall, self.lr_auc, self.lr_best_params),
                'SVM': (self.svm, self.svm_recall, self.svm_auc, self.sv_best_params),
            }

            best_model_name = max(models, key=lambda k: models[k][1])
            best_model = models[best_model_name][0]
            self.logger.info('Best model selected: ' + best_model_name + ' with Recall: ' + str(models[best_model_name][1]))

            all_results = [
                {'model_name': name, 'score': info[1], 'auc': info[2], 'params': info[3]}
                for name, info in models.items()
            ]

            return best_model_name, best_model, all_results

        except Exception as e:
            self.logger.exception('Exception raised while finding best model:' + str(e))
            raise Exception()