from keras.wrappers.scikit_learn import KerasRegressor

class Custom_KerasRegressor(KerasRegressor):
    def fit(self, x, y, val_1, val_2, **fit_params):
        if val_1 is not None and val_2 is not None:
            super(Custom_KerasRegressor, self).fit(x, y, validation_data=[val_1, val_2], **fit_params)
        else:
            super(Custom_KerasRegressor, self).fit(x, y, **fit_params)