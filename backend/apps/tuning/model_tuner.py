from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score,
    precision_score, f1_score, classification_report
)
from apps.core.logger import logging


class ModelTuner:
    """
    *****************************************************************************
    *
    * filename:       model_tuner.py
    * version:        2.0
    * author:         VPM
    * creation date:  11-MAR-2026
    *
    * change history:
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * VIVEK           11-MAR-2026    1.0      initial creation
    * VIVEK           18-MAR-2026    2.0      removed GridSearchCV and clustering,
    *                                         use best params with class_weight=balanced
    *
    *
    * description:    Class to train RandomForest with best params
    *
    ****************************************************************************
    """
    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = logging.getLogger('ModelTuner')

    def train_best_model(self, train_x, train_y, test_x, test_y):
        """
        Train RandomForestClassifier with best params from notebook
        (class_weight='balanced' instead of SMOTETomek).
        Returns: model, metrics_dict
        """
        try:
            self.logger.info('Training RandomForest (class_weight=balanced)...')

            model = RandomForestClassifier(
                n_estimators=150,
                criterion='gini',
                max_depth=None,
                max_features='log2',
                class_weight='balanced',
                random_state=42
            )
            model.fit(train_x, train_y)

            # Evaluate on test set
            y_pred = model.predict(test_x)
            y_proba = model.predict_proba(test_x)[:, 1]

            metrics = {
                'Model': 'RandomForestClassifier',
                'Accuracy': round(accuracy_score(test_y, y_pred), 4),
                'Recall': round(recall_score(test_y, y_pred), 4),
                'Precision': round(precision_score(test_y, y_pred), 4),
                'F1_Score': round(f1_score(test_y, y_pred), 4),
                'AUC_ROC': round(roc_auc_score(test_y, y_proba), 4),
                'Parameters': str({
                    'n_estimators': 150,
                    'criterion': 'gini',
                    'max_depth': None,
                    'max_features': 'log2',
                    'class_weight': 'balanced',
                    'random_state': 42
                })
            }

            self.logger.info('Accuracy: %s | Recall: %s | F1: %s | AUC-ROC: %s' % (
                metrics['Accuracy'], metrics['Recall'], metrics['F1_Score'], metrics['AUC_ROC']
            ))

            return model, metrics

        except Exception as e:
            self.logger.exception('Model training failed: %s' % str(e))
            raise Exception()
