# Random forest parameters
rf_params = {}

rf_params['test2'] = {'bootstrap': True, 'min_samples_leaf': 12,
                      'min_samples_split': 5, 'criterion': 'entropy',
                      'max_features': 0.6, 'max_depth': 17,
                      'class_weight': None, 'n_estimators': 10000}

rf_params['test4'] = {'bootstrap': True, 'min_samples_leaf': 8,
                      'min_samples_split': 9, 'criterion': 'gini',
                      'max_features': 1.0, 'max_depth': 19,
                      'class_weight': None, 'n_estimators': 10000}

# Gradient Boosting parameters
gb_params = {}

gb_params['test3'] = {'learning_rate': 0.02, 'min_samples_leaf': 18,
                     'min_samples_split': 22,
                     'max_features': 0.4, 'max_depth': 29,
                     'class_weight': None, 'n_estimators': 10000}

gb_params['test4'] = {'learning_rate': 0.1, 'min_samples_leaf': 15,
                     'subsample':0.8,'min_samples_split': 2,
                     'max_features': 0.1, 'max_depth': 10,
                     'n_estimators': 30000}

gb_params['test5'] = {'learning_rate': 0.02, 'min_samples_leaf': 5,
                     'subsample':1.0,'min_samples_split': 19,
                     'max_features': 0.4, 'max_depth': 15,
                     'n_estimators': 30000}
