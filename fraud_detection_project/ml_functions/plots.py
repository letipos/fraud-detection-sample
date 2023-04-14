
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot(df):
    # convert categorical variables to numerical labels
    cat_features = ['gender', 'merchant_category', 'merchant_country', 'merchant_city']
    for feature in cat_features:
        df[feature] = df[feature].astype('category').cat.codes

    # create a correlation matrix
    corr = df.corr()

    # create a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='Blues', annot=True, fmt='.2f')
    plt.title('Correlation Matrix')

    # create a pairplot of the numerical features
    num_features = ['age', 'income', 'credit_score', 'transaction_amount']
    plt.figure(figsize=(10, 8))
    sns.pairplot(df[num_features], diag_kind='kde')
    plt.suptitle('Pairplot of Numerical Features', y=1.02)

    # create a countplot of the categorical features
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    for i, feature in enumerate(cat_features):
        row = i // 2
        col = i % 2
        sns.countplot(x=feature, data=df, ax=axs[row][col])
        axs[row][col].set_title(f'Countplot of {feature}')
        axs[row][col].set_xlabel('')
        axs[row][col].set_ylabel('Count')
        axs[row][col].tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
