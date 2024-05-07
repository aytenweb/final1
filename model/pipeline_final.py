import pandas as pd
import numpy as np
import dill as dill
from sklearn.pipeline import Pipeline
from datetime import datetime
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

columns_to_replace = ['utm_campaign', 'utm_adcontent', 'utm_keyword', 'utm_source', 'utm_medium', 'device_brand',
                      'geo_city', 'geo_country', 'device_model', 'device_os']
categorical_features = [
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
    'device_category', 'device_os', 'device_brand', 'device_model',
    'device_screen_resolution', 'geo_country', 'geo_city',
    'device_browser_first', 'resolution_category', 'traffic_type'
]


def convert_columns(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col] = df_copy[col].fillna('unknown')
    return df_copy


def safe_convert_to_int(df):
    import pandas as pd
    import numpy as np
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=[np.number]).columns:
        if df_copy[col].isnull().any():
            df_copy[col] = df_copy[col].fillna(0)
        try:
            df_copy[col] = df_copy[col].astype(int)
        except ValueError as e:
            df_copy[col] = df_copy[col].astype(float)
    return df_copy


def browser_first(df):
    df_copy = df.copy()
    if 'device_browser' in df_copy.columns and df_copy['device_browser'].notna().any():
        df_copy['device_browser_first'] = df_copy['device_browser'].apply(lambda x: x.split()[0] if isinstance(x, str) and x else None)
    else:
        df_copy['device_browser_first'] = None
    return df_copy


def fill_model(df):
    df_copy = df.copy()
    df_copy['device_os'] = df_copy['device_os'].str.strip().replace('(not set)', 'Unknown')
    most_common_os = df_copy['device_model'].mode()[0]
    df_copy['device_model'] = df_copy['device_model'].fillna(most_common_os)
    return df_copy


def traffic(df):
    df_copy = df.copy()
    df_copy['traffic_type'] = df_copy['utm_medium'].where(df_copy['utm_medium'].isin(['organic']), 'inorganic')
    return df_copy


def resolution(df):
    import pandas as pd
    df_copy = df.copy()
    df_copy['device_screen_resolution'].replace({'(not set)': '0x0', 'not set': '0x0'}, inplace=True)
    resolution = df_copy['device_screen_resolution'].str.split('x', expand=True)
    resolution.columns = ['height', 'width']
    resolution['height'] = resolution['height'].astype(int)
    resolution['width'] = resolution['width'].astype(int)
    df_copy['height'] = resolution['height']
    df_copy['width'] = resolution['width']
    df_copy['resolution_sum'] = df_copy['height'] + df_copy['width']
    bins = [0, 1000, 2000, 3000, float('inf')]
    labels = ['low', 'medium', 'big', 'extra']
    df_copy['resolution_category'] = pd.cut(df_copy['resolution_sum'], bins=bins, labels=labels)
    df_copy['resolution_category'] = df_copy['resolution_category'].astype(str).fillna('other')
    df_copy[['height', 'width']] = df_copy['device_screen_resolution'].str.split('x', expand=True).astype(int)
    return df_copy


def transforms(df, columns_to_replace):
    import pandas as pd
    import numpy as np
    if df is None:
        raise ValueError("Переданный DataFrame не должен быть None")
    df_copy = df.copy()
    df_copy[columns_to_replace] = pd.DataFrame.replace(df_copy[columns_to_replace], ['', np.nan], 'other')
    df_copy[columns_to_replace] = df_copy[columns_to_replace].replace(r'\(other\)', 'other', regex=True)
    df_copy[columns_to_replace] = df_copy[columns_to_replace].replace(r'\(not set\)', 'other', regex=True)
    df_copy['utm_medium'] = df_copy['utm_medium'].replace({'referral': 'organic', 'organic': 'organic', '\(none\)': 'organic'}, regex=True)
    return df_copy

def dat(df):
    import pandas as pd
    df_copy = df.copy()
    df_copy['visit_datetime'] = pd.to_datetime(df_copy['visit_date'] + ' ' + df_copy['visit_time'],format='%Y-%m-%d %H:%M:%S')
    df_copy['day'] = df_copy['visit_datetime'].dt.day
    df_copy['weekday'] = df_copy['visit_datetime'].dt.weekday
    df_copy['month'] = df_copy['visit_datetime'].dt.month
    df_copy['hour'] = df_copy['visit_datetime'].dt.hour
    df_copy['visit_date_timestamp'] = (pd.to_datetime(df_copy['visit_date']).astype('int64') // 10 ** 9).astype('int32')
    base_datetime = pd.to_datetime('1970-01-01 00:00:00')
    df_copy['visit_datetime_timestamp'] = (df_copy['visit_datetime'] - base_datetime).dt.total_seconds()
    base_date = pd.to_datetime('1970-01-01')
    df_copy['visit_date_timestamp'] = (pd.to_datetime(df_copy['visit_date']) - base_date).dt.days
    base_time = pd.to_datetime('00:00:00')
    df_copy['visit_time'] = pd.to_datetime(df_copy['visit_time'], format='%H:%M:%S')
    return df_copy


def drop_columns(df, columns_to_drop):
    df_copy = df.copy()
    return df_copy.drop(columns_to_drop, axis=1)


def scale_numeric_features(df, numeric_features=None):
    from sklearn.preprocessing import StandardScaler
    df_copy = df.copy()
    scaler = StandardScaler()
    if numeric_features is None:
        numeric_features = df_copy.select_dtypes(include=['number']).columns
    df_copy.loc[:, numeric_features] = scaler.fit_transform(df_copy[numeric_features])
    return df_copy


def main():
    import pandas as pd
    dfs = pd.read_csv('data/ga_hits.csv', low_memory=False)
    desired_event_actions = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                             'sub_open_dialog_click', 'sub_custom_question_submit_click',
                             'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                             'sub_car_request_submit_click']
    dfs['flag'] = 0
    dfs.loc[dfs['event_action'].isin(desired_event_actions), 'flag'] = 1
    grouped_dfs = dfs.groupby('session_id')['flag'].max().reset_index()
    dfse = pd.read_csv('data/ga_sessions.csv', low_memory=False)
    df = dfse.merge(grouped_dfs, on='session_id', how='inner')
    df.to_csv('data/merged_data_final.csv', index=False)

    columns_to_replace = ['utm_campaign', 'utm_adcontent', 'utm_keyword', 'utm_source','utm_medium', 'device_brand','geo_city', 'geo_country','device_model','device_os']

    categorical_features = [
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
        'device_category', 'device_os', 'device_brand', 'device_model',
        'device_screen_resolution', 'geo_country', 'geo_city',
        'device_browser_first', 'resolution_category', 'traffic_type'
    ]

    columns_to_drop = ['session_id', 'visit_date','visit_time','visit_datetime','device_browser']

    numeric_features = ['visit_number', 'month', 'day', 'weekday', 'hour', 'width', 'height', 'resolution_sum',
                        'visit_date_timestamp']

    if df is None or df.empty:
        raise ValueError("DataFrame 'df' is empty or None")

    preprocessor = Pipeline(steps=[
        ('convert_columns', FunctionTransformer(convert_columns)),
        ('transforms', FunctionTransformer(transforms, kw_args={'columns_to_replace': columns_to_replace})),
        ('resolution', FunctionTransformer(resolution)),
        ('browser_first', FunctionTransformer(browser_first)),
        ('traffic', FunctionTransformer(traffic)),
        ('fill_model', FunctionTransformer(fill_model)),
        ('dates', FunctionTransformer(dat)),
        ('scale_numeric', FunctionTransformer(scale_numeric_features, kw_args={'numeric_features': numeric_features})),
        ('drop_columns', FunctionTransformer(drop_columns, kw_args={'columns_to_drop': columns_to_drop})),
        ('safe_convert_to_int', FunctionTransformer(safe_convert_to_int))

    ])

    processed_df = preprocessor.fit_transform(df)
    processed_df.to_csv('data/transformed_final.csv', index=False)

    X_train, X_test, y_train, y_test = train_test_split(processed_df.drop(['flag'], axis=1), processed_df['flag'],
                                                        test_size=0.2, random_state=42)

    categorical_features_indices = [X_train.columns.get_loc(c) for c in categorical_features if c in X_train.columns]

    for col in categorical_features_indices:
        X_train.iloc[:, col] = X_train.iloc[:, col].astype(str)
        X_test.iloc[:, col] = X_test.iloc[:, col].astype(str)

    class_weights = [1, 15]

    classifier = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        random_state=42,
        verbose=100,
        cat_features=categorical_features_indices,
        class_weights=class_weights
    )

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    acc_mean = cv_scores.mean()
    cv_scores_std = cv_scores.std()
    #print("Средняя точность:", acc_mean)
    #print("Стандартное отклонение:", cv_scores_std)

    print(f'model: {type(classifier).__name__}, roc_auc: {roc_auc:.4f}, accuracy: {accuracy:.4f}, acc_mean: {acc_mean:.4f}, cv_scores.std: {cv_scores_std:.4f}')
    with open('car_pipe_final.pkl', 'wb') as file:
        dill.dump({
            'model': classifier,
            'preprocessor': preprocessor,
            'columns': X_train.columns.tolist(),
            'categorical_features_indices': categorical_features_indices,
                'model_params': {
                'iterations': 1000,
                'depth': 6,
                'learning_rate': 0.1,
                'loss_function': 'Logloss',
                'class_weights': class_weights,
                'random_state': 42
                },
                'metadata': {
                'name': 'Final Work : Car price prediction model',
                'author': 'Aiten_A',
                'version': 101,
                'date': datetime.now(),
                'accuracy': accuracy,
                'roc-auc': roc_auc,
                'acc_mean': acc_mean,
                'cv_scores_std': cv_scores_std
            }
        }, file)

    with open('X_final.pkl', 'wb') as f:
        dill.dump((X_train, y_train, X_test, y_test), f)


if __name__ == '__main__':
    main()
