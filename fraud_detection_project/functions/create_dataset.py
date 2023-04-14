
import pandas as pd
import numpy as np

def create_dataset():
    # create a dataset with multiple features
    n_samples = 10000
    n_fraud_samples = 100
    n_normal_samples = n_samples - n_fraud_samples

    features = {
        'age': np.random.randint(18, 70, size=n_samples),
        'gender': np.random.choice(['M', 'F'], size=n_samples),
        'income': np.random.randint(1000, 100000, size=n_samples),
        'credit_score': np.random.randint(300, 900, size=n_samples),
        'transaction_amount': np.random.randint(1, 10000, size=n_samples),
        'merchant_category': np.random.choice(['Grocery', 'Gas', 'Restaurants', 'Entertainment'], size=n_samples),
        'merchant_country': np.random.choice(['USA', 'Canada', 'Mexico'], size=n_samples),
        'merchant_city': np.random.choice(['New York', 'Los Angeles', 'Toronto', 'Vancouver', 'Mexico City'], size=n_samples),
        'transaction_time': np.random.randint(0, 24, size=n_samples),
    }

    fraud_indices = np.random.choice(n_samples, size=n_fraud_samples, replace=False)
    labels = np.zeros(n_samples)
    labels[fraud_indices] = 1

    df = pd.DataFrame(features)
    df['label'] = labels
    
    # save the dataset as a csv file
    df.to_csv('fraud_detection_data.csv', index=False)

    return(df)
