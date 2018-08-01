
# coding: utf-8

# In[6]:


import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# In[7]:


from sklearn.model_selection import train_test_split
iris=datasets.load_iris()
X=iris.data
Y=iris.target

x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.3,stratify=Y)
#in 3 khat e zir aksaran tekrar mishe 
KNN=KNeighborsClassifier(n_neighbors=3,p=2)
KNN.fit(x_train,y_train)
KNN.score(x_test,y_test)


# In[8]:


neighbors=np.arange(1,30) #araye az 1 ta 30 por shode


# In[13]:


test_ac=np.empty(len(neighbors)) #yek araye khali be size e neighbors
train_ac=np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn=KNeighborsClassifier(n_neighbors=k,p=2)
    knn.fit(x_train, y_train)
    test_ac[i]=knn.score(x_test,y_test)
    train_ac[i]=knn.score(x_train,y_train)
plt.plot(neighbors,test_ac,label="test")
plt.plot(neighbors,train_ac,label="train")
plt.legend()

plt.show()
#k fold behtare!!!miangin migire behtarin k ro entekhab mikone :miangine deghat ha ro migire:)


# In[ ]:


#task = read about overfit and underfit


# In[15]:


from sklearn.tree import DecisionTreeClassifier 
#monaseb baraye dade haye ba meghyase kam
DT=DecisionTreeClassifier()
DT.fit(x_train,y_train)
DT.score(x_test,y_test)

