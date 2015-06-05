# Random forest parameters
rf_params = {}

rf_params['test2'] = {'bootstrap': True, 'min_samples_leaf': 12,
                      'min_samples_split': 5, 'criterion': 'entropy',
                      'max_features': 0.6, 'max_depth': 17,
                      'class_weight': None, 'n_estimators': 10000}
