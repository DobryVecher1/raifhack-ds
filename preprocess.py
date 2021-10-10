import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from category_encoders import TargetEncoder
from copy import copy
from metrics import deviation_metric

class NotFitException(Exception):
    pass


class GroupNanReplacer:
    
    def __init__(self, cat_features: list, groupby_name: str):
        self.regions_group = {}
        self.cat_features = cat_features
        self.groupby_name = groupby_name
    
    def fit(self, df, target=None, **kwargs):
        # get columns of df which have nan values
        nan_columns = [col for col in df if any(df[col].isna())]
        
        # search region groups for each nan columns
        for col in nan_columns:
            agg_func = pd.Series.mode if col in self.cat_features else pd.Series.mean
            group = df.groupby(self.groupby_name)[col].agg(lambda x: agg_func(x)).fillna(agg_func(df[col]))
            self.regions_group[col] = group
        return self
    
    def transform(self, df, target=None, **kwargs):
        if self.regions_group == {}:
            raise NotFitException('You must fit the replacer before transforming!')
            
        for col, group in self.regions_group.items():
            df[col] = df[col].fillna('None')
            
            for i in range(len(df)):
                if df[col].iloc[i] == 'None':
                    group_value = df[self.groupby_name].iloc[i]
                    fillna_value = self.regions_group[col].loc[group_value]
                    
                    if type(fillna_value) is np.ndarray:
                        if len(fillna_value) == 0:
                            fillna_value = df[col].mode() if col in self.cat_features else float(df[col].mean())
                        else:
                            fillna_value = fillna_value[0]
                    if type(fillna_value) is pd.Series:
                        fillna_value = fillna_value.values
                        if len(fillna_value) == 0:
                            fillna_value = df[col].mode() if col in self.cat_features else float(df[col].mean())
                        else:
                            fillna_value = fillna_value[0]
                    df[col].iloc[i] = fillna_value
            if col not in self.cat_features:
                df[col] = df[col].astype('float')
        return df
    
    def fit_transform(self, df, target=None, **kwargs):
        self.fit(df, target, **kwargs)
        return self.transform(df, target, **kwargs)


class Processing:
    
    def __init__(self, model):
        self.gnr = None
        self.encoder = None
        self.model = model
        self.best_features = []
        self.first_iteration_model = None
        self.ohe = None
        self.nn_model = None
        
    def kmean_feature(self, data):
        km = KMeans(
            n_clusters=10, init='random',
            n_init=300, max_iter=3000,
            tol=0.01, random_state=0)
        km_feature = km.fit_predict(data[['floor', 'total_square']])
        return km_feature
    
    def ohe_transform(self, df, features=[], mode='train'):
        if mode == 'train':
            self.ohe = OneHotEncoder(handle_unknown='ignore')
            enc_df = pd.DataFrame(self.ohe.fit_transform(df[features]).toarray())
        else:
            enc_df = pd.DataFrame(self.ohe.transform(df[features]).toarray())
        df = df.join(enc_df)
        df = df.drop(features, 1)
        return df
    
    def generate_feature_importance(self, data):
        features_importance = dict(zip(data.columns, self.model.feature_importances_))
        features_importance = {feature: importance for feature, importance in sorted(features_importance.items(), key=lambda item: item[1])}
        features = [feature for feature, importance in features_importance.items() if importance > 50]
        return features
        
    def transform_target(self, target, reverse=False):
        return np.expm1(target) if reverse else np.log1p(target)
        
    def fit(self, data: pd.DataFrame, target_col: str, price_type=1, validation=False, **fit_params):
        df, target = self._process(data, target_col=target_col, price_type=price_type, mode='train')
        
        if validation:
            X_train, X_test, y_train, y_test = train_test_split(df, target, shuffle=False, test_size=0.15)
            self.model = self.model.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)], **fit_params)
        else:
            self.model = self.model.fit(df, target, **fit_params)
            
        return self, df
    
    def predict(self, data, target_col=None, price_type=1) -> np.ndarray:
        df, _ = self._process(data, target_col=target_col, price_type=price_type, mode='test')
        prediction = self.model.predict(df)
        return self.transform_target(prediction, reverse=True) * 0.94, df
    
    def evaluate(self, data, target_col: str, price_type=1, n_splits=5, shuffle=False, validation=False, **fit_params):
        best_models_result = 10
        kf = KFold(n_splits=n_splits, shuffle=shuffle)
        
        kf_means = {
            'train': [],
            'test': []
        }
        
        df_train, df_target = self._process(data, target_col=target_col)
        for train_index, test_index in kf.split(df_train):

            X_train, X_test = df_train.iloc[train_index], df_train.iloc[test_index]
            y_train, y_test = df_target.iloc[train_index], df_target.iloc[test_index]
            
            if validation:
                X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, shuffle=False, test_size=0.15)
                self.model = self.model.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)], **fit_params)
            else:
                self.model = self.model.fit(X_train, y_train, **fit_params)
            
            train_predictions = self.model.predict(X_train)
            test_predictions = self.model.predict(X_test)
            
            y_train = self.transform_target(y_train, reverse=True)
            y_test = self.transform_target(y_test, reverse=True)
            train_predictions = self.transform_target(train_predictions, reverse=True)
            test_predictions = self.transform_target(test_predictions, reverse=True)

            train_metric = deviation_metric(y_train, train_predictions)
            test_metric = deviation_metric(y_test, test_predictions)

            kf_means['train'].append(train_metric)
            kf_means['test'].append(test_metric)

        return kf_means
    
    def _process(self, data, target_col=None, price_type=1, mode='train', init_nan=True):
        df = data[data.price_type == price_type].drop('price_type', price_type)
        
        if target_col:
            target = df[target_col]
            df = df.drop(target_col, 1)
            
            target = self.transform_target(target)
        else:
            target = None
            
        cat_features = [col for col in df if df[col].dtype == 'object']
        
        if mode == 'train' or init_nan:
            self.gnr = GroupNanReplacer(cat_features, 'region')
            df = self.gnr.fit_transform(df)
        else:
            df = self.gnr.transform(df)
            
        df = self.ohe_transform(df, ['city', 'street', 'region'], mode=mode)
        cat_features = [col for col in df if df[col].dtype == 'object']
                
        if mode == 'train':
            self.encoder = TargetEncoder(cols=cat_features)
            df = self.encoder.fit_transform(df, target)
        else:
            df = self.encoder.transform(df)
            
        km_feature = self.kmean_feature(df)
        df['km_feature'] = km_feature
        
        if mode != 'train':
            df = df[self.best_features]
            
        if mode == 'train':
            self.model = self.model.fit(df, target)
            features = self.generate_feature_importance(df)
            self.best_features = features
            df = df[self.best_features]
            
            self.first_iteration_model = copy(self.model)
            self.first_iteration_model = self.first_iteration_model.fit(df, target)
        
        predictions = self.first_iteration_model.predict(df) * 0.94
        df['first_predictions'] = predictions
        
        return df, target