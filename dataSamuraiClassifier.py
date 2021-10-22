#For .csv manipulation
import pandas as pd
#For guessing gender
import gender_guesser.detector as gender
import numpy as np

#Only going to use variables from signups and leads
signups = pd.read_csv('signups.csv')
leads = pd.read_csv('leads.csv')

#Build classifier database
c_db = leads

#Drop phone column
c_db = c_db.drop(['Phone Number'],axis=1)

#Add signup colums
c_db['Signup'] = -1
c_db['Signup'] = np.where(c_db['Name'].isin(signups.Lead),1,0)

#Now want to guess gender from name
g = gender.Detector()
#Only keep the first name
c_db.Name = c_db.Name.apply(lambda x: x.split()[0])

#Now add gender and apply gender-guesser to fill new column
c_db['Gender'] = 'blah'
c_db['Gender'] = c_db['Name'].apply(g.get_gender)
#Replace all mostly_male with male, mostly_female with female, and unknown with andy (androgynous)
conditions = [c_db['Gender'] == 'mostly_male',
              c_db['Gender'] == 'male',
              c_db['Gender'] == 'mostly_female',
              c_db['Gender'] == 'female',
              c_db['Gender'] == 'unknown',
              c_db['Gender'] == 'andy']
choices = ['male','male','female','female','andy','andy']
c_db['Gender'] = np.select(conditions,choices)

#Now can drop name
c_db = c_db.drop(['Name'],axis=1)

c_db = c_db.sort_values(by=['Signup'])
c_db = c_db.tail(1536)
#from sklearn.utils import shuffle
#c_db = shuffle(c_db)

#Encode categorical variables and drop excess 
#to avoid the dummy variable trap
c_db = pd.get_dummies(c_db)
c_db = c_db.drop(['Region_wales'],axis=1)
c_db = c_db.drop(['Sector_wholesale'],axis=1)
c_db = c_db.drop(['Gender_andy'],axis=1)
#Uncomment these to remove /add
c_db = c_db.drop(['Gender_male'],axis=1)
c_db = c_db.drop(['Gender_female'],axis=1)

#Swap signup to end
cols = list(c_db.columns)
#a, b = cols.index('Gender_male'), cols.index('Signup')
a, b = cols.index('Sector_retail'), cols.index('Signup')
cols[b], cols[a] = cols[a], cols[b]
c_db = c_db[cols]

#Split into training variables, X, and results, Y
X = c_db.iloc[:,:-1].values
Y = c_db.iloc[:,-1].values

#Scale the training variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

##For the test-train split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

#Scale the training variables
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initialise NN
classifier = Sequential()
#Initial Layer
classifier.add(Dense(activation="relu",input_dim=16,units=6,kernel_initializer="uniform"))
#Hidden Layer
classifier.add(Dense(activation="relu",units=6,kernel_initializer="uniform"))
#Output Layer
classifier.add(Dense(activation="sigmoid",units=1,kernel_initializer="uniform"))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'] )

classifier.fit(X_train,Y_train,batch_size = 15,epochs = 100)

print('Test size is:',len(X_test))
Y_pred = classifier.predict(X_test)
summedProb = np.sum(Y_pred)
signupRate = summedProb / 154.0
expectedSignups = 1000*signupRate

print('Signed up rate predicted to be',signupRate)
print('Expected sign ups for the next 1000 (by extrapolating)',expectedSignups)
