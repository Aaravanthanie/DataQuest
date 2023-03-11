import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataSet = pd.read_csv("DataQuest Dataset - train_data.csv")

# print(dataSet.head())


# X = dataSet[['LeadTime', 'AvgRoomPrice', 'SpecialRequests']]
# y = dataSet['BookingStatus']

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier()
# knn.fit(X_train,y_train)


# dataSet_prediction = knn.predict([[10, 95.00, 0]])

# if dataSet_prediction == "Canceled":
#     print("I Guess... Canceled")
# elif dataSet_prediction == "Not_Canceled":
#     print("I Guess... a Not Canceled")

# print("Score on Testing Data: ", knn.score(X_test, y_test))

# import pandas as pd
# import matplotlib.pyplot as plt

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'LeadTime')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'ArrivalYear')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'ArrivalMonth')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'ArrivalDate')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'NumWeekendNights')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'NumWeekNights')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'MealPlan')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'Parking')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'RoomType')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'NumAdults')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'NumChildren')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'MarketSegment')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'RepeatedGuest')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'NumPrevCancellations')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'NumPreviousNonCancelled')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'AvgRoomPrice')

# plt.show()

# dataSet.plot(kind = 'scatter', x = 'BookingStatus', y = 'SpecialRequests')

# plt.show()

from pandas import *

dataFile = read_csv('DataQuest Dataset - train_data.csv')
month = dataFile['ArrivalMonth'].tolist()
adults = dataFile['NumAdults'].tolist()
booking = dataFile['BookingStatus'].tolist()

cancelledMonths = []
cancelledAdults = []

for i in range(len(adults)):
    if booking[i] == "Canceled":
        cancelledMonths.append(month[i])
        cancelledAdults.append(adults[i])

a1 = []
m1 = []

a2 = []
m2 = []

a3 = []
m3 = []

a4 = []
m4 = []

for i in range(len(cancelledAdults)):
    if cancelledAdults[i] == "0":
        a1.append(cancelledAdults[i])
        m1.append(cancelledMonths[i])
    elif cancelledAdults[i] == "1":
        a2.append(cancelledAdults[i])
        m2.append(cancelledMonths[i])
    elif cancelledAdults[i] == "2":
        a3.append(cancelledAdults[i])
        m3.append(cancelledMonths[i])
    elif cancelledAdults[i] == "3":
        a4.append(cancelledAdults[i])
        m4.append(cancelledMonths[i])
    


print("ca: " + str(len(cancelledAdults)))
print("a0: " + str(len(a1)))
print("m0: " + str(len(m1)))
print("a1: " + str(len(a2)))
print("m1: " + str(len(m2)))
print("a2: " + str(len(a3)))
print("m2: " + str(len(m3)))
print("a3: " + str(len(a4)))
print("m3: " + str(len(m4)))

import matplotlib.pyplot as plt

# plt.scatter(cancelledAdults,)
# plt.xlabel('Adults')
# plt.ylabel('Booking')
# plt.title('Scatter Plot')
# plt.show()

# text = dataFile.read()

# type(dataFile)

# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs

# n_samples = 200
# blob_centers = ([1, 1], [3, 4])
# data, labels = make_blobs(n_samples=n_samples, 
#                           centers=blob_centers, 
#                           cluster_std=0.5,
#                           random_state=0
#                           )


# colours = ('green', 'red')
# fig, ax = plt.subplots()

# for n_class in range(len(blob_centers)):
#     ax.scatter(data[labels==n_class][:, 0], 
#                data[labels==n_class][:, 1], 
#                c=colours[n_class], 
#                s=30, 
#                label=str(n_class))

# plt.show()
