#%%
#Fawad Javed Fateh 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import data 
import sklearn.cluster as cluster

#importing and cleaning the dataset
dataSet=pd.read_csv('Iris.csv',usecols=[i for i in range(5)])
dataSet.head()
dataSet=dataSet.iloc[:,[1,2,3,4]].values

#applying the k-means algorithm to calcualte the withinClusterSums of 10 iterations of inreasing k-values
k=range(1,11)
withinClusterSum=[]
for k in k:
    k_Means=cluster.KMeans(n_clusters=k,init="k-means++")
    k_Means=k_Means.fit(dataSet)
    itterWss=k_Means.inertia_
    withinClusterSum.append(itterWss)

#plotting the acculumated withinCLusterSum values along with the k values ranging from 1 to 10 to observe an elbow point
plt.scatter(range(1,11),withinClusterSum,color="red")
plt.plot(range(1,11),withinClusterSum,color="blue")
plt.title("Using The Elbow Method")
plt.xlabel('K-Values')
plt.ylabel('WCSS')
plt.show()
print("After a careful glance of the above visualization, it can be observed that there is no signifcant decrease in the WSS after k=3. Hence this is the optimum value of k")

#Using the k-value determined by the elbow method to retrain the dataset acc to the value
k_Means=cluster.KMeans(n_clusters=3,init='k-means++')
k_Means=k_Means.fit_predict(dataSet)

#Now visualing the data in form of optimum clusters
plt.scatter(dataSet[k_Means == 0, 0], dataSet[k_Means == 0, 1], s = 100, color = 'orange', label = 'Iris-setosa')
plt.scatter(dataSet[k_Means == 1, 0], dataSet[k_Means == 1, 1], s = 100, color = 'yellow', label = 'Iris-versicolour')
plt.scatter(dataSet[k_Means == 2, 0], dataSet[k_Means == 2, 1],s = 100, color = 'green', label = 'Iris-virginica')
plt.legend()
plt.title("Clustered Data")
plt.show()

print("The dataset has been classified into 3 distinct clusters")

# %%
