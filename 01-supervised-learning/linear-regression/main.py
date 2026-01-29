
import numpy as np
import pandas as pd

# # machine learning
# import keras
# import ml_edu.experiment
# import ml_edu.results

# # data visualization
import plotly.express as px
import matplotlib.pyplot as plt

# chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
# training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

# print('Read dataset completed successfully.')
# print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
# training_df.head(200)

# View dataset statistics
# print(training_df.describe(include='all'))

# What is the maximum fare?
# max_fare = training_df['FARE'].max()
# print("What is the maximum fare? Answer: ${fare:.2f}".format(fare = max_fare))

# # What is the mean distance across all trips?
# mean_distance = training_df['TRIP_MILES'].mean()
# print("What is the mean distance across all trips? 	Answer: {mean:.4f} miles".format(mean = mean_distance))

# # How many cab companies are in the dataset?
# num_unique_companies =  training_df['COMPANY'].nunique()
# print("How many cab companies are in the dataset? 		Answer: {number}".format(number = num_unique_companies))


# # What is the most frequent payment type?
# most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
# print("What is the most frequent payment type? 		Answer: {type}".format(type = most_freq_payment_type))


# # Are any features missing data?
# missing_values = training_df.isnull().sum().sum()
# print("Are any features missing data? 	Answer:", "No" if missing_values == 0 else "Yes")

# View correlation matrix
# print(training_df.corr(numeric_only = True))
# px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

m = len(x_train)
print(f"Number of training examples is: {m}")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()