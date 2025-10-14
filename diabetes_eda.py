# diabetes_eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from machine_learning.data_loaders.data_loader_indicators import load_diabetes_dataset, preprocess_data

sns.set(style="whitegrid")

# ------------------------------
# 1. Load dataset
# ------------------------------
df = load_diabetes_dataset()

# ------------------------------
# 2. Basic overview
# ------------------------------
print("Dataset shape:", df.shape)
print("\nTarget value counts:\n", df['Diabetes_binary'].value_counts())

numeric_cols = ['BMI', 'PhysHlth', 'MentHlth', 'Age', 'Income', 'PhysActivity']
categorical_cols = ['Sex','HighBP','HighChol','CholCheck','Smoker','Stroke','HeartDiseaseorAttack',
                    'Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk']

# Descriptive stats for numeric
print("\nNumeric attributes stats:\n", df[numeric_cols].describe())

# Counts for categorical
for col in categorical_cols:
    print(f"\nCounts for {col}:\n", df[col].value_counts())

# ------------------------------
# 3. MLP Autoencoder for outlier detection
# ------------------------------
# Preprocess numeric data
scaler = StandardScaler()
X_num = scaler.fit_transform(df[numeric_cols])

# Simple autoencoder using MLP
mlp_autoencoder = MLPRegressor(hidden_layer_sizes=(32, 16, 32),
                               activation='relu',
                               max_iter=500,
                               random_state=42)

# Fit autoencoder: input = output
mlp_autoencoder.fit(X_num, X_num)
X_pred = mlp_autoencoder.predict(X_num)

# Compute reconstruction error
reconstruction_error = np.mean((X_num - X_pred)**2, axis=1)
df['reconstruction_error'] = reconstruction_error

# Flag outliers as top 5% highest reconstruction error
threshold = np.percentile(reconstruction_error, 95)
df['outlier_mlp'] = (reconstruction_error > threshold).astype(int)

print("\nOutliers detected by MLP Autoencoder:")
print(df[df['outlier_mlp'] == 1][numeric_cols + ['reconstruction_error']])

# ------------------------------
# 4. Missing value imputation using MLP (numeric only)
# ------------------------------
# Identify numeric columns with missing values
num_missing_cols = df[numeric_cols].columns[df[numeric_cols].isna().any()].tolist()
if num_missing_cols:
    print("\nImputing missing numeric values using MLP...")
    for col in num_missing_cols:
        other_cols = [c for c in numeric_cols if c != col]
        # train MLP to predict missing column
        not_null = df[df[col].notnull()]
        X_train = not_null[other_cols].values
        y_train = not_null[col].values

        mlp_imputer = MLPRegressor(hidden_layer_sizes=(32,16),
                                   activation='relu',
                                   max_iter=500,
                                   random_state=42)
        mlp_imputer.fit(X_train, y_train)

        # Predict missing values
        null_mask = df[col].isnull()
        X_missing = df.loc[null_mask, other_cols].values
        df.loc[null_mask, col] = mlp_imputer.predict(X_missing)

# ------------------------------
# 5. Continue with original EDA
# ------------------------------

# Numeric histograms and boxplots
for col in numeric_cols:
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Histogram of {col}')

    plt.subplot(1,2,2)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')

    plt.show()

# Categorical bar charts and pie charts
for col in categorical_cols:
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    sns.countplot(x=df[col])
    plt.title(f'Bar plot of {col}')

    plt.subplot(1,2,2)
    df[col].value_counts().plot.pie(autopct='%1.1f%%')
    plt.ylabel('')
    plt.title(f'Pie chart of {col}')

    plt.show()

# Heatmap of correlations between numeric attributes
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols + ['Diabetes_binary']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation heatmap")
plt.show()

# ------------------------------
# Continue with outlier analysis, correlation, pairplots etc. as before
# ------------------------------
# ------------------------------
# 6. Outlier detection (original + MLP)
# ------------------------------
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col} classical outliers:\n", outliers[[col,'Diabetes_binary']])

# Categorical rare classes
for col in categorical_cols:
    counts = df[col].value_counts(normalize=True)
    rare = counts[counts < 0.05]
    if not rare.empty:
        print(f"Rare categories in {col}:\n", rare)

# ------------------------------
# 7. Correlation analysis
# ------------------------------
# Pearson and Spearman correlation for numeric
for col in numeric_cols:
    pearson_corr = pearsonr(df[col], df['Diabetes_binary'])[0]
    spearman_corr = spearmanr(df[col], df['Diabetes_binary'])[0]
    print(f"{col}: Pearson={pearson_corr:.3f}, Spearman={spearman_corr:.3f}")

# Chi-square for categorical vs target
for col in categorical_cols:
    contingency = pd.crosstab(df[col], df['Diabetes_binary'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"{col}: Chi2={chi2:.2f}, p-value={p:.3f}")

# ------------------------------
# 8. Visualization for research questions
# ------------------------------
# Age group vs BMI scatter + boxplot
plt.figure(figsize=(12,5))
sns.scatterplot(x='Age', y='BMI', hue='Diabetes_binary', data=df)
plt.title("Age vs BMI colored by Diabetes")
plt.show()

sns.boxplot(x='Age', y='BMI', hue='Diabetes_binary', data=df)
plt.title("Boxplot of Age vs BMI by Diabetes")
plt.show()

# Smoking, HighBP, HeartDisease vs Diabetes
for col in ['Smoker','HighBP','HeartDiseaseorAttack']:
    plt.figure(figsize=(12,5))
    sns.countplot(x=col, hue='Diabetes_binary', data=df)
    plt.title(f"Diabetes by {col}")
    plt.show()

# Distribution of numeric values by target
for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='Diabetes_binary', y=col, data=df)
    sns.violinplot(x='Diabetes_binary', y=col, data=df, alpha=0.3)
    plt.title(f"{col} distribution by Diabetes")
    plt.show()

# Pairplot for selected attributes
selected_cols = numeric_cols[:4] + ['Diabetes_binary']
sns.pairplot(df[selected_cols], hue='Diabetes_binary')
plt.show()

# ------------------------------
# 9. Summary including MLP outliers
# ------------------------------
print("EDA completed. Key insights:")
print("- Identified outliers in numeric attributes using classical IQR method.")
print("- Identified outliers using MLP Autoencoder (reconstruction error).")
print("- Detected rare categories in categorical attributes.")
print("- Missing numeric values imputed using MLPRegressor.")
print("- Calculated Pearson and Spearman correlations for numeric attributes.")
print("- Chi-square tests for categorical attributes vs target.")
print("- Visualizations prepared for research questions and feature selection.")
print(f"Number of MLP outliers: {df['outlier_mlp'].sum()} ({df['outlier_mlp'].mean()*100:.2f}%)")
