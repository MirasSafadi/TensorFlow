import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import csv

def generate_synthetic_test_dataset(X_train, Y_train, sample_size=100, seed=42):
    np.random.seed(seed)

    # Determine age range from training data
    age_min = int(X_train.min())
    age_max = int(X_train.max())

    # Generate random ages
    X_test = np.random.randint(age_min, age_max + 1, size=sample_size).reshape(-1, 1)

    # Compute salary label distribution
    unique, counts = np.unique(Y_train, return_counts=True)
    salary_distribution = dict(zip(unique, counts / len(Y_train)))

    # Generate salary labels based on distribution
    Y_test = np.random.choice(
        list(salary_distribution.keys()),
        size=sample_size,
        p=list(salary_distribution.values())
    )

    return X_test, Y_test



def extract_training_data_numpy_only(csv_path):
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header row

        age_index = header.index('age')
        salary_index = header.index('salary')

        ages = []
        salaries = []

        for row in reader:
            ages.append(float(row[age_index]))
            salaries.append(float(row[salary_index]))

    X = np.array(ages)  # 2D array for features
    Y = np.array(salaries)             # 1D array for labels
    return X, Y

model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(1,)),
    layers.Dense(15, activation='tanh'),
    layers.Dense(20, activation='tanh'),
    layers.Dense(15, activation='tanh'),
    layers.Dense(5, activation='relu'),
    layers.Dense(1, activation='sigmoid')

])       

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Example dataset (random values for demonstration)
X_train, Y_train = extract_training_data_numpy_only('adult11_modified.csv')
# X_train = np.random.rand(100, 5)
# y_train = np.random.randint(0, 2, 100)
print(X_train.dtype)
print(Y_train.dtype)

# Train the model
model.fit(X_train, Y_train, epochs=15, batch_size=5)


# Example test data
X_test, Y_test = generate_synthetic_test_dataset(X_train, Y_train)
# test_data = np.random.rand(10, 5)
# test_labels = np.random.randint(0, 2, 10)

loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy:.2f}")