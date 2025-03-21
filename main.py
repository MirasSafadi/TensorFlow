import tensorflow as tf
import numpy as np

# Creating a NumPy array
data = np.array([[1, 2], [3, 4], [5, 6]])

# Creating a dataset
dataset = tf.data.Dataset.from_tensor_slices(data)

for element in dataset:
    print(element.numpy())


# Load CSV file
dataset = tf.data.experimental.make_csv_dataset(
    "Titanic-Dataset.csv",
    batch_size=2,
    label_name="PassengerId",
    num_epochs=1
)
dataset = tf.data.Dataset.range(10)
dataset = dataset.shuffle(5)

# Iterating through dataset
for batch in dataset.take(1):
    print(batch)
