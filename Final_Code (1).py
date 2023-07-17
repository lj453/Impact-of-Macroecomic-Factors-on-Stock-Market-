#!/usr/bin/env python
# coding: utf-8

# **1.Exploratory Data Analysis of Apple**

# In[1]:


#Loading the System
# all imports and env variables
import pandas as pd
import numpy as np
pd.core.common.is_list_like = pd.api.types.is_list_like
import datetime
import pandas_datareader.data as web
from pandas.plotting import autocorrelation_plot, scatter_matrix
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(8, 5))
import seaborn as sns


# In[ ]:


#Install yfinance inorder to get stock data of other companies
pip install yfinance


# In[2]:


#Mentioning the start and end date for the comparison of stock prices of major tech companies
start = datetime.datetime(2016, 7, 12)
end = datetime.datetime(2022, 11, 16)


# In[3]:


#Importing yahoofinance
import yfinance as yf
from pandas_datareader import data as pdr

# 1) Using pandas datareader and Yahoo Finance
yf.pdr_override()

apple = pdr.get_data_yahoo('AAPL', start = start,end=end)


# In[4]:


# download multipe stocks into a single dataframe:

all_stocks_list = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
all_stocks = yf.download(all_stocks_list, start = start,end=end)


# In[5]:


all_stocks


# In[6]:


stocks_adj = pd.DataFrame(all_stocks.iloc[:,:4])
stocks_adj.head()


# In[7]:


#making sure no null values
stocks_adj.isnull().values.any()


# In[8]:


#resampling to the last day of a business month
stocks_adj = stocks_adj.resample('BM').last()
stocks_adj


# In[9]:


#remove the additional index, so that it's easier to index
stocks_adj.columns = stocks_adj.columns.droplevel(0)


# In[10]:


stocks_adj


# In[11]:


#plot the Apple stock price data to see trends
colors = ['b']
ax = stocks_adj['AAPL'].plot(linewidth=2, fontsize=12, figsize = (16,5),color = colors);
plt.title('Stocks prices changes of Apple (Adj Close)', fontsize = 15)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Price in US', fontsize = 15)


# In[12]:


#Ploting the data to see the trends of Major tech comanies
colors = ['b', 'r', 'g', 'y']
ax = stocks_adj.plot(linewidth=2, fontsize=12, figsize = (16,5),color = colors);
ax.legend(['Apple','Google','Microsoft','Tesla']);
plt.title('Stocks prices changes', fontsize = 15)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Price in US', fontsize = 15)


# In[13]:


stocks_volume = pd.DataFrame(all_stocks.iloc[:,20:24])
stocks_volume.head()


# In[14]:


#plot the data to see trends of volume 
colors = ['b', 'r', 'g', 'y']
ax = stocks_volume.plot(linewidth=2, fontsize=12, figsize = (16,5),color = colors);
ax.legend(['Apple','Google','Microsoft','Tesla']);
plt.title('Stocks prices changes', fontsize = 15)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Volume', fontsize = 15)


# In[15]:


#Potting the correaltion heatmap
sns.heatmap(stocks_adj.corr(),annot=True)


# In[17]:


sns.pairplot(data=stocks_adj)


# **2.Macroeconomic Factors on Stock Price**

# In[19]:


#Read the Datafile 
df=pd.read_csv('Apple_M.csv')
df


# In[20]:


df=df.iloc[:51,:]
df=pd.DataFrame(df)
df


# In[21]:


df.dtypes


# In[22]:


#
df['DATE']=pd.to_datetime(df['DATE'])
df['UNRATE']=df['UNRATE'].astype(float)


# In[23]:


df.dtypes


# In[24]:


#making sure no null values
df.isnull().values.any()


# In[25]:


#PLot the GDP Progression
plt.plot(df['DATE'],df['GDP'])
plt.title('GDP ', fontsize = 15)
plt.xlabel('DATE', fontsize = 12)
plt.ylabel('GDP', fontsize = 12)


# In[26]:


#Plot the Unemployment rate over the years
plt.plot(df['DATE'],df['UNRATE'])
plt.title('Unemployment Rate', fontsize = 15)
plt.xlabel('DATE', fontsize = 12)
plt.ylabel('UNRATE', fontsize = 12)


# In[27]:


plt.plot(df['DATE'],df['CPI'])
plt.title('Consumer Price Index', fontsize = 15)
plt.xlabel('DATE', fontsize = 12)
plt.ylabel('CPI', fontsize = 12)


# In[28]:


#Pearson Co-realtion
pearsoncorr=df.corr(method='pearson')
pearsoncorr


# In[29]:


sns.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)


# In[30]:


#Plotting pairplot
sns.pairplot(data=df)


# In[31]:


import seaborn as sns


# In[32]:


#plotting regression

fig=plt.figure(figsize=(10,7))
sns.regplot(x=df['Adj close'],y=df['GDP'],color='blue',marker='+')

plt.title("Stock Price Vs GDP",size =18)
plt.xlabel('Stock Price(Adj close)',size=14)
plt.ylabel('GDP',size=14)


# In[33]:


#plotting regression

fig=plt.figure(figsize=(10,7))
sns.regplot(x=df['Adj close'],y=df['UNRATE'],color='green',marker='+')

plt.title("Stock Price Vs Unemployment Rate ",size =18)
plt.xlabel('Stock Price(Adj close)',size=14)
plt.ylabel('UnRate',size=14)


# In[34]:


#plotting regression

fig=plt.figure(figsize=(10,7))
sns.regplot(x=df['Adj close'],y=df['CPI'],color='orange',marker='+')

plt.title("Stock Price Vs CPI ",size =18)
plt.xlabel('Stock Price(Adj close)',size=14)
plt.ylabel('CPI',size=14)


# **Linear Regression**

# In[35]:


#Loading the System 
import numpy as np
from sklearn.linear_model import LinearRegression


# In[36]:


#Creating x and y variable fpr the model 
x=np.array(df['Adj close']).reshape((-1,1))
y=np.array(df['GDP'])


# In[37]:


#reate a linear regression model and fit it using the existing data.
#Create an instance of the class LinearRegression, which will represent the regression model:
model = LinearRegression()
#This statement creates the variable model as an instance of LinearRegression


# In[38]:


#With .fit(), you calculate the optimal values of the weights 𝑏₀ and 𝑏₁, using the existing input and output, x and y, as the arguments. In other words, .fit() fits the model. 
model=LinearRegression().fit(x,y)


# In[39]:


r_sq=model.score(x,y)
print(f"coeficient of determination:{r_sq}")


# In[40]:


print(f"intercept: {model.intercept_}")


# In[41]:


print(f"slope: {model.coef_}")


# **3.Technical Analysis**

# In[18]:


import time
import numpy as np
import datetime
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA

import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[72]:


pip install xgboost


# In[43]:


import warnings 
warnings.filterwarnings("ignore")


# In[45]:


def parser(x):
    return datetime.datetime.strptime(x,'%m/%d/%Y')


# In[46]:


import pandas as pd 
data=pd.read_csv("AAPL.csv", header=[0], parse_dates=[0], date_parser=parser)


# In[47]:


data.head(5)


# In[48]:


print('There are {} number of days in the dataset.'.format(data.shape[0]))


# In[49]:


#Feature Generation
def get_technical_indicators(data_t): #function to generate feature technical indicators
    
    
    # Create 7 and 21 days Moving Average
    data_t['ma7'] = data_t['Adj Close'].rolling(window = 7).mean()
    data_t['ma21'] = data_t['Adj Close'].rolling(window = 21).mean()
    
    #Create MACD
    data_t['26ema'] = data_t['Adj Close'].ewm(span=26).mean()
    data_t['12ema'] = data_t['Adj Close'].ewm(span=12).mean()
    data_t['MACD'] = (data_t['12ema']-data_t['26ema'])
    
    #Create Bollinger Bands
    data_t['20sd'] = data_t['Adj Close'].rolling(window = 20).std()
    data_t['upper_band'] = (data_t['Adj Close'].rolling(window = 20).mean()) + (data_t['20sd']*2)
    data_t['lower_band'] = (data_t['Adj Close'].rolling(window = 20).mean()) - (data_t['20sd']*2)
    
    
    #Create Exponential moving average
    data_t['ema'] = data_t['Adj Close'].ewm(com=0.5).mean()
    
    #Create Momentum
    data_t['momentum'] = (data_t['Adj Close']/100)-1
    
    
    
    return data_t


# In[50]:


data_TI= get_technical_indicators(data)


# In[51]:


data_TI.head()


# In[52]:


def plot_technical_indicators(data_t, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = data_t.shape[0]
    xmacd_ = shape_0-last_days
    
    data_t= data_t.iloc[-last_days:, :]
    x_ = range(3, data_t.shape[0])
    x_ =list(data_t.index)
    
    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(data_t['ma7'],label='MA 7', color='orange',linestyle='--')
    plt.plot(data_t['Adj Close'],label='Adj Closing Price', color='b')
    plt.plot(data_t['ma21'],label='MA 21', color='r',linestyle='--')
    plt.plot(data_t['upper_band'],label='Upper Band', color='c')
    plt.plot(data_t['lower_band'],label='Lower Band', color='c')
    plt.fill_between(x_, data_t['lower_band'], data_t['upper_band'], alpha=0.35)
    plt.title('Technical indicators for Apple - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD & Momentum')
    plt.plot(data_t['MACD'],label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(data_t['momentum'],label='Momentum', color='b',linestyle='-')

    plt.legend(loc=(0.85,0.7))
    plt.show()


# In[53]:


plot_technical_indicators(data_TI,500)


# In[50]:


data_FT = data[['Date', 'Adj Close']]
close_fft = np.fft.fft(np.asarray(data_FT['Adj Close'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))


# In[51]:


#Compare perfomance of the Fourier trasforms with different number of components on the same plot. 
plt.figure(figsize=(14,7), dpi=100)
fft_list = np.asarray(fft_df['fft'].tolist())
for num_ in [3, 6, 9,105]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
    plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
plt.plot(data_FT['Adj Close'],  label='Real')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 3: Apple (Adj close) stock prices & Fourier transforms')
plt.legend()
plt.show()


# In[52]:


def get_fourier(data):
    data_FT = data[['Date', 'Adj Close']]
    close_fft = np.fft.fft(np.asarray(data_FT['Adj Close'].tolist()))
    close_fft = np.fft.ifft(close_fft)
    close_fft
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_list_m10= np.copy(fft_list); fft_list_m10[100:-100]=0
    data['Fourier'] = pd.DataFrame(fft_list_m10).apply(lambda x: np.abs(x))
    #dataset['absolute'] = dataset['Fourier'].apply(lambda x: np.abs(x))
    return data


# In[53]:


data_TI=get_fourier(data)


# In[54]:


data_TI.head()


# In[55]:


from collections import deque
items = deque(np.asarray(fft_df['absolute'].tolist()))
items.rotate(int(np.floor(len(fft_df)/2)))
plt.figure(figsize=(10, 7), dpi=80)
plt.stem(items)
plt.title('Components of Fourier transforms')
plt.show()


# **LSTM**

# In[16]:


import pandas as pd
import numpy as nd 


# In[9]:


df=pd.read_csv('AAPL.csv')


# In[10]:


df.head()


# In[14]:


df1=df.reset_index()['Adj Close']


# In[15]:


df1.head()


# In[20]:


##LSTM are sensitive to the scale of the data,so we apply MInMax scaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[22]:


df1.shape


# In[25]:


df1


# In[27]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[28]:


training_size,test_size


# In[29]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[30]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[31]:


print(X_train.shape),print(y_train.shape)


# In[32]:


print(X_test.shape),print(ytest.shape)


# In[33]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[34]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[35]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[36]:


model.summary()


# In[37]:


model.summary()


# In[38]:


#Model fitting number of epochs=100
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[39]:


import tensorflow as tf


# In[40]:


###prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[41]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[42]:


### Calculate RMSE performance metrics(for the training dataset)
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[43]:


### Calculate RMSE for testing dataset
math.sqrt(mean_squared_error(ytest,test_predict))


# In[68]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1),color = 'blue',label = 'Actual Closing Prices')
plt.plot(trainPredictPlot,color ='orange',label = 'Training predicitons')
plt.plot(testPredictPlot,color ='green',label = 'Predicted Stock price')
plt.xlabel('Days')
plt.ylabel('Stock price')
plt.legend(loc='best')
plt.show()


# Blue color : Actual value
# Green Color : Predicted stock price 
# orange color : Training Predicitons 

# In[46]:


len(test_data)


# In[47]:


x_input=test_data[1035:].reshape(1,-1)
x_input.shape


# In[49]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input


# In[50]:


# demonstrate prediction for next 30 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[51]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[52]:


len(df1)


# In[70]:


plt.plot(day_new,scaler.inverse_transform(df1[3141:]),color = 'blue',label = 'Actual Closing Prices')
plt.plot(day_pred,scaler.inverse_transform(lst_output),color = 'orange',label = 'Forecasted price')
plt.legend(loc='best')


# In[54]:


#Extending the prediction to the actual time line 
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[55]:


df3=scaler.inverse_transform(df3).tolist()


# In[56]:


plt.plot(df3)

