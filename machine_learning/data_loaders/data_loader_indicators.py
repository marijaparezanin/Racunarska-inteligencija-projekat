import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from machine_learning.consts import DIABETES_INDICATORS_DATASET, DIABETES_INDICATORS_TARGET, DIABETES_INDICATORS_TARGET_TYPE, DEFAULT_NUMERIC_IMPUTER_STRATEGY, DEFAULT_CATEGORY_IMPUTER_STRATEGY

ordinal_mappings = {
    'Age': {
        '0 to 10': 0,
        '11 to 27': 1,
        '18 to 24': 2,
        '25 to 29': 3,
        '30 to 34': 4,
        '35 to 39': 5,
        '40 to 44': 6,
        '45 to 49': 7,
        '50 to 54': 8,
        '55 to 59': 9,
        '60 to 64': 10,
        '65 to 69': 11,
        '70 to 74': 12,
        '75 to 79': 13,
        '80 or older': 14
    },
    'GenHlth': {
        'Poor': 1,
        'Fair': 2,
        'Good': 3,
        'Very good': 4,
        'Excellent': 5
    },
    'Education': {
        'Never Attended School': 1,
        'Elementary': 2,
        'High School': 3,
        'Some College Degree': 4,
        "Advanced Degree": 5,
        "6": 6
    }
}

def load_diabetes_dataset(split='train'):
    dataset_path = DIABETES_INDICATORS_DATASET + split + ".parquet"
    df = pd.read_parquet(dataset_path)
    return df

def conditional_map(df, column, mapping):
    if df[column].dtype == object or not pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].map(mapping)

def preprocess_data(df, target=DIABETES_INDICATORS_TARGET, target_type=DIABETES_INDICATORS_TARGET_TYPE):
    #df.head(20).to_csv("beginner.csv", index=False)

    conditional_map(df, 'Age', ordinal_mappings['Age'])
    conditional_map(df, 'GenHlth', ordinal_mappings['GenHlth'])
    conditional_map(df, 'Education', ordinal_mappings['Education'])
    conditional_map(df, 'Diabetes_binary', {'Diabetic': 1, 'Non-Diabetic': 0})


    # Drop columns not useful for training
    drop_cols = [target, 'ID']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target]

    # Manually define column types
    ordinal_cols = ['PhysHlth', 'Age', 'GenHlth', 'MentHlth', 'Income']
    numeric_cols = ['BMI']
    nominal_cols = ['Diabetes_binary', 'HighBP', "HighChol", 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'Fruits', 'Veggies', 'HvyAlcoholConsump','AnyHealthcare', 'NoDocbcCost', 'PhysActivity', 'DiffWalk', 'Sex']

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
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop="if_binary"))
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
