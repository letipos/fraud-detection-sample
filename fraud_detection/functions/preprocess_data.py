
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    # separate numerical and categorical features
    num_features = ['age', 'income', 'credit_score', 'transaction_amount', 'transaction_time']
    cat_features = ['gender', 'merchant_category', 'merchant_country', 'merchant_city']

    # perform data cleaning and feature engineering
    # ...

    # preprocess numerical features
    num_transformer = StandardScaler()
    num_transformed = num_transformer.fit_transform(df[num_features])

    # preprocess categorical features
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    cat_transformed = cat_transformer.fit_transform(df[cat_features])

    # combine numerical and categorical features
    transformer = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    X = transformer.fit_transform(df)
    y = df['label']
    return X, y
