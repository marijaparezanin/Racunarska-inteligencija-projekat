import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from machine_learning.consts import DIABETES_PREDICTION_DATASET, DIABETES_PREDICTION_TARGET, DIABETES_PREDICTION_TARGET_TYPE, DEFAULT_NUMERIC_IMPUTER_STRATEGY, DEFAULT_CATEGORY_IMPUTER_STRATEGY

# Nominals get auto mapped with OneHotEncoder
def map_numeric_age_to_group(age):
    if age <= 10:
        return 0
    elif age <= 17:
        return 1
    elif age <= 24:
        return 2
    elif age <= 29:
        return 3
    elif age <= 34:
        return 4
    elif age <= 39:
        return 5
    elif age <= 44:
        return 6
    elif age <= 49:
        return 7
    elif age <= 54:
        return 8
    elif age <= 59:
        return 9
    elif age <= 64:
        return 10
    elif age <= 69:
        return 11
    elif age <= 74:
        return 12
    elif age <= 79:
        return 13
    else:
        return 14


def load_diabetes_dataset(split='train'):
    dataset_path =DIABETES_PREDICTION_DATASET + split + ".parquet"
    df = pd.read_parquet(dataset_path)
    return df

def preprocess_data(df, target=DIABETES_PREDICTION_TARGET, target_type=DIABETES_PREDICTION_TARGET_TYPE):
    #df.head(20).to_csv("beginner.csv", index=False)

    #Transform the words into their numerical equivalents (only for ordinals)
    df['age'] = df['age'].apply(map_numeric_age_to_group)


    # Drop columns not useful for training
    drop_cols = [target, 'ID']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target]

    # Manually define column types
    ordinal_cols = ['age']
    numeric_cols = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    nominal_cols = ['diabetes', 'gender', 'hypertension', 'heart_disease', 'smoking_history']


    # If i choose to change the target
    if target_type == 'ordinal':
        if target in ordinal_cols:
            ordinal_cols.remove(target)
    elif target_type == 'nominal':
        if target in nominal_cols:
            nominal_cols.remove(target)
    else:
        if target in numeric_cols:
            numeric_cols.remove(target)


    # Imputers
    num_imputer = SimpleImputer(strategy=DEFAULT_NUMERIC_IMPUTER_STRATEGY)
    cat_imputer = SimpleImputer(strategy=DEFAULT_CATEGORY_IMPUTER_STRATEGY)

    # Pipelines
    numeric_pipeline = Pipeline([
        ('imputer', num_imputer),
        ('scaler', StandardScaler())
    ])
    ordinal_pipeline = Pipeline([
        ('imputer', cat_imputer)
    ])
    nominal_pipeline = Pipeline([
        ('imputer', cat_imputer),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop="if_binary")) # drop means that if theres only two values i can have 1 colymn instead of 2
    ])

    transformers = [
        ('num', numeric_pipeline, numeric_cols),
        ('ord', ordinal_pipeline, ordinal_cols)
    ]
    if nominal_cols:
        transformers.append(('nom', nominal_pipeline, nominal_cols))

    preprocessor = ColumnTransformer(transformers)

    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    #X_df.head(20).to_csv("output.csv", index=False)

    return X_processed, y, preprocessor
