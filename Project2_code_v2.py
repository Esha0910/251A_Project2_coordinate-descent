#!/usr/bin/env python
# coding: utf-8

# ## Importing the necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import math
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[2]:


# important variables

num_epochs = 10000
random_state = 42

height=600
width=1600
renderer="png"
test_size = 0.33
training_losses_df = pd.DataFrame(index=np.arange(1, num_epochs + 1), columns=[f'Random Co-ordinate Descent', f'Greedy Co-ordinate Descent'])


# ### Load and analyse the data

# In[3]:


wine_data = load_wine()


# In[4]:


print(wine_data.DESCR)


# In[5]:


x = wine_data.data
y = wine_data.target
print(f'Shape of features: {x.shape}')
print(f'Shape of target: {y.shape}')


# In[6]:


mask = (y==0) | (y==1)
x_binary = x[mask]
y_binary = y[mask]
print(f'Shape of features for binary classification: {x_binary.shape}')
print(f'Shape of target for binary classification: {y_binary.shape}')


# ### Normalize the data

# In[7]:


scaler = StandardScaler()
scaler.fit(x_binary)
x_binary = scaler.transform(x_binary)


# In[8]:


def evaluate(y_test, y_pred):
    accuracy = 100*accuracy_score(y_test, y_pred)
    print(f'Accuracy of the model: {accuracy}\n -----------------------------')
    print(f'Classification report: \n {classification_report(y_test, y_pred)}\n -----------------------------')
    print(f'Confusion matrix: \n {confusion_matrix(y_test, y_pred)}\n -----------------------------')    


# ### Training loss for binary data

# In[9]:


lr_model = LogisticRegression(random_state=random_state, max_iter=num_epochs)
lr_model.fit(x_binary, y_binary)
y_pred_prob = lr_model.predict_proba(x_binary)


# In[10]:


gd_loss = log_loss(y_binary, y_pred_prob)
print(f'Log Loss for Gradient Descent: {gd_loss}')


# In[11]:


lr_model.predict(x_binary)


# In[12]:


lr_model.predict_proba(x_binary)


# In[13]:


lr_model.predict(x_binary).shape


# In[14]:


lr_model.predict_proba(x_binary).shape


# In[15]:


def sigmoid_func(x):
    return (1.0 / (1+(np.exp(-1 * x))))


# In[16]:


learning_rate = 0.05
coefficients = np.random.rand(x_binary.shape[1], 1)

for current_epoch in np.arange(1, num_epochs+1):
#     print(f'{current_epoch}')

    #calulating predicted probabilities
    y_binary_prob = sigmoid_func(np.dot(x_binary, coefficients))
#     y_pred = np.where(y_binary_prob >= 0.5, 1, 0)

    #calculating loss
    current_loss = log_loss(y_true=y_binary,
                   y_pred=y_binary_prob
                   )
    
    training_losses_df.loc[current_epoch, 'Gradient Descent'] = current_loss
#     print(f'{current_loss}')
    #Caluclating gradient
    current_gradient = np.dot(x_binary.T, (y_binary_prob - y_binary.reshape(y_binary.shape[0], 1))) / x_binary.shape[0]
    
    #Randomly choosing a coordinate
#     curr_coeff_idx = np.random.randint(low = 0, high = (coefficients.shape[0]))
    
    coefficients -= learning_rate * current_gradient
    
evaluate(y_test=y_binary, y_pred=sigmoid_func(np.dot(x_binary, coefficients))>0.5)


# In[17]:


lr_model.coef_


# In[18]:


lr_model.intercept_


# In[19]:


lr_model.n_features_in_


# ### Random feature co-ordinate descent

# ### Training loss

# In[20]:


learning_rate = 0.01
coefficients = np.random.rand(x_binary.shape[1], 1)

for current_epoch in np.arange(1, num_epochs+1):
#     print(f'{current_epoch}')

    #calulating predicted probabilities
    y_binary_prob = sigmoid_func(np.dot(x_binary, coefficients))
#     y_pred = np.where(y_binary_prob >= 0.5, 1, 0)

    #calculating loss
    current_loss = log_loss(y_true=y_binary,
                   y_pred=y_binary_prob
                   )
    
    training_losses_df.loc[current_epoch, 'Random Co-ordinate Descent'] = current_loss

    #Caluclating gradient
    current_gradient = np.dot(x_binary.T, (y_binary_prob - y_binary.reshape(y_binary.shape[0], 1))) / x_binary.shape[0]
    
    #Randomly choosing a coordinate
    curr_coeff_idx = np.random.randint(low = 0, high = (coefficients.shape[0]))
    
    coefficients[curr_coeff_idx] -= learning_rate * current_gradient[curr_coeff_idx]
    
evaluate(y_test=y_binary, y_pred=sigmoid_func(np.dot(x_binary, coefficients))>0.5)


# In[21]:


training_losses_df


# ### Greedy coordinate descent

# In[22]:


learning_rate = 0.05
coefficients = np.random.rand(x_binary.shape[1], 1)
for current_epoch in np.arange(1, num_epochs + 1):
    # if (curr_epoch % 1000 == 0):
    #     print(curr_epoch)

    y_binary_prob = sigmoid_func(np.dot(x_binary, coefficients))
#     y_pred = np.where(y_binary_prob >= 0.5, 1, 0)


    # compute loss
    current_loss = log_loss( y_true=y_binary,
        y_pred=y_binary_prob
       
    )
    training_losses_df.loc[current_epoch, 'Greedy Co-ordinate Descent'] = current_loss

    # update the coefficients based on greedy CD
    # step 1: select the co-ordinate
    current_gradient = np.dot(x_binary.T, (y_binary_prob - y_binary.reshape(y_binary.shape[0], 1))) / x_binary.shape[0]
    curr_coeff_idx = np.argmax(np.abs(current_gradient))

    # step 2: update that coefficient
    coefficients[curr_coeff_idx] -= learning_rate * current_gradient[curr_coeff_idx]


# In[23]:


training_losses_df


# In[24]:


fig = px.line(training_losses_df.astype(float))
fig.update_layout(height=height, width = width, xaxis_title='Epoch', yaxis_title='Log loss')
fig.write_image("Fig1.png")
fig


# ### Variations in GCD 

# ### Variations in the learning rate

# In[25]:


learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

for learning_rate in learning_rates:
    # Randomly initializing the coefficients
    coefficients = np.random.rand(x_binary.shape[1], 1)
    
    for current_epoch in np.arange(1, num_epochs+1):
        y_binary_prob = sigmoid_func(np.dot(x_binary, coefficients))
        y_pred = (y_binary>0.5).astype(float)
        
        current_loss = log_loss(y_true= y_binary,
                        y_pred=y_binary_prob,
                        
                        )
        
        training_losses_df.loc[current_epoch, f'GCD [LR: {learning_rate}, Epochs: {num_epochs}]'] = current_loss
        
        current_gradient = np.dot(x_binary.T, y_binary_prob - y_binary.reshape(y_binary.shape[0], 1)) / x_binary.shape[0]
        
        curr_coeff_idx = np.argmax(np.abs(current_gradient))
        coefficients[curr_coeff_idx] -= learning_rate*current_gradient[curr_coeff_idx]
        
evaluate(y_test=y_binary, y_pred=sigmoid_func(np.dot(x_binary, coefficients))>0.5)


# In[26]:


training_losses_df


# In[27]:


training_losses_df['Random Co-ordinate Descent'][10]


# In[28]:


fig = px.line(training_losses_df.astype(float))
fig.update_layout(height=height, width = width, xaxis_title='Epoch', yaxis_title='Log loss')
fig.write_image("Fig2.png")


# In[29]:


fig


# ### Trying with different number of epochs

# ### Training loss

# In[30]:


learning_rate = 0.05
num_epoch_options = [1000, 5000, 10000, 25000, 50000, 75000, 100000]

for num_epochs in num_epoch_options:
    for current_epoch in np.arange(1, num_epochs + 1):
        # if (curr_epoch % 1000 == 0):
        #     print(curr_epoch)

        y_binary_prob = sigmoid_func(np.dot(x_binary, coefficients))
        

        # compute loss
        current_loss = log_loss( y_true=y_binary,
            y_pred=y_binary_prob
           
        )
        training_losses_df.loc[current_epoch, f'GCD [LR: {learning_rate}, Epochs: {num_epochs}]'] = current_loss

        # update the coefficients based on greedy CD
        # step 1: select the co-ordinate
        current_gradient = np.dot(x_binary.T, (y_binary_prob - y_binary.reshape(y_binary.shape[0], 1))) / x_binary.shape[0]
        curr_coeff_idx = np.argmax(np.abs(current_gradient))

        # step 2: update that coefficient
        coefficients[curr_coeff_idx] -= learning_rate * current_gradient[curr_coeff_idx]


# In[31]:


training_losses_df


# In[32]:


fig = px.line(training_losses_df.astype(float))
fig.update_layout(height=height, width = width, xaxis_title='Epoch', yaxis_title='Log loss')
fig.write_image("Fig3.png")
fig


# ### Sparse coordinate descent 

# In[ ]:


learning_rate = 0.05
k_s = [1, 2, 3, 4, 5]
losses_df_SCD = pd.DataFrame(index=np.arange(1, num_epochs + 1))

for k in k_s:
    coefficients = np.random.rand(x_binary.shape[1], 1)
    #Performing 1st iteration outside of the for loop to get the best k coefficients
    y_binary_prob = sigmoid_func(np.dot(x_binary, coefficients))

    current_loss = log_loss(y_true= y_binary,
                   y_pred=y_binary_prob)

    losses_df_SCD.loc[1, f'Sparse Coordinate Descent [k: {k}, Epochs: {num_epochs}]'] = current_loss

    current_gradient = np.dot(x_binary.T, y_binary_prob - y_binary.reshape(y_binary.shape[0], 1))
#         print(current_gradient)
    sorted_gradients = np.argsort(current_gradient.flatten())[::-1]
#         print(sorted_gradients)
    top_k = sorted_gradients[:k]
#     top_k = np.random.choice(x_binary.shape[1], k)
#         print(top_k)
    for idx in range(0, len(coefficients)):
        if idx in top_k:
            coefficients[idx] -= learning_rate*current_gradient[idx]
        else:
            coefficients[idx] = 0    

    for current_epoch in np.arange(2, num_epochs+1):
        y_binary_prob = sigmoid_func(np.dot(x_binary, coefficients))

        current_loss = log_loss(y_true= y_binary,
                       y_pred=y_binary_prob
                        
        )
        
        losses_df_SCD.loc[current_epoch, f'Sparse Coordinate Descent [k: {k}, Epochs: {num_epochs}]'] = current_loss
        
        current_gradient = np.dot(x_binary.T, y_binary_prob - y_binary.reshape(y_binary.shape[0], 1))

        for idx in top_k:
            coefficients[idx] -= learning_rate*current_gradient[idx]

evaluate(y_test=y_binary, y_pred=sigmoid_func(np.dot(x_binary, coefficients))>0.5)


# In[34]:


losses_df_SCD


# In[37]:


fig = px.line(losses_df_SCD.astype(float))
fig.update_layout(height=height, width = width, xaxis_title='Epoch', yaxis_title='Log loss')
fig.write_image("Fig4.png")
fig.show()


# In[ ]:




