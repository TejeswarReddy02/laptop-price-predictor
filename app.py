#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('laptop_data.csv')
df=pd.DataFrame(data)
df.head()


# In[2]:


df.info()


# In[3]:


df.isnull().sum()


# In[4]:


df=df.drop(['Unnamed: 0'],axis=1)
df.head()


# In[5]:


df['Ram']=df['Ram'].str.replace("GB","")
df.head()


# In[6]:


df['Weight']=df['Weight'].str.replace("kg","")
df.head()


# In[7]:


df['Ram']=df['Ram'].astype('int32')
df['Weight']=df['Weight'].astype('float32')
df.info()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation="vertical")
plt.show()


# In[9]:


df['Company'].value_counts().plot(kind='bar')


# In[10]:


sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation="vertical")
plt.show()


# In[11]:


sns.lineplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation="vertical")
plt.show()


# In[12]:


df.head()


# In[13]:


df["cpu name"]=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[14]:


df.head()


# In[15]:


df.head()


# In[16]:


df['y_res']=df['ScreenResolution'].apply(lambda x:x.split()[-1])
df["x_res"] = df["y_res"].apply(lambda x:x.split("x")[0]).astype('int')
df["y_res"] = df["y_res"].apply(lambda x:x.split("x")[-1]).astype('int')
df['pixel_count'] = df['x_res'] * df['y_res']
df.drop(columns=['x_res','y_res'],inplace=True)


# In[17]:


df.drop(columns=['ScreenResolution'],axis=1)
df.head()


# In[18]:


df.info()


# In[19]:


df['touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if "Touchscreen" in x  else 0)


# In[20]:


df.tail(10)


# In[21]:


df['touchscreen'].value_counts().plot(kind='bar')


# In[22]:


df.sample()


# In[23]:


df.drop(columns=['Cpu'],inplace=True)
df['OpSys'].value_counts()


# In[24]:


def opsys(name):
    if name=='Windows 10' or name == 'Windows 7' or name == 'Windows 10 s':
        return 'Windows'
    elif name=="macOS" or name == 'Mac OS X':
        return 'Mac'
    else:
        return 'others'


# In[25]:


df["os"] = df['OpSys'].apply(opsys)


# In[26]:


df.sample()


# In[27]:


df['os'].value_counts()
df.drop(columns=['OpSys'],inplace=True)


# In[28]:


df.sample()


# In[29]:


df['Gpu'].head()


# In[30]:


df["GPU"] = df['Gpu'].apply(lambda x:x.split(" ")[0]) 


# In[31]:


df['GPU'].value_counts()


# In[32]:


df.drop(columns=['Gpu'],inplace=True)
df.head()


# In[33]:


df['Memory'].sample()


# In[34]:


df['Memory_type'] = df['Memory'].apply(lambda x:x.split(" ")[-1])


# In[35]:


df['Memory_type'] = df['Memory_type'].str.replace('Storage','SSD')


# In[36]:


df['Memory_type'].value_counts()


# In[37]:


df.drop(columns=['Memory'],inplace=True)


# In[38]:


df.drop(columns=['ScreenResolution'],inplace=True)
df.head()


# In[39]:


#x=df.drop(columns=['Price'])
#y=df['Price']


# In[40]:


#x_train


# In[41]:


df['Ram_weight'] = df['Ram']*df['Weight']
df.head()


# In[42]:


df.info()


# In[43]:


df['Inches_weight'] = df['Inches']*df['Weight']
df.head()


# In[44]:


df['cpu'] = df['cpu name'].apply(lambda x:x.split()[2])


# In[45]:


def cpu_name(name):
    n="Intel"
    if n in name:
        return 1
    else:
        return 0
df['cpu_name'] = df['cpu name'].apply(cpu_name)


# In[70]:


import numpy as np
df['Price'] = np.log(df['Price'])
df.head()
df['pixel'] = np.log(df['pixel_count'])
df.head()


# In[79]:


x=df.drop(columns=['Ram','Weight','Inches','cpu name','Price','pixel_count'])
y=df['Price']


# In[80]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[81]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=2)


# In[82]:


from sklearn.ensemble import RandomForestRegressor
RandomForestRegressor(n_estimators=100, random_state=42)


# In[83]:


step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(handle_unknown="ignore",sparse_output=False,drop='first'),[0,1,3,4,5,8]
    )
],remainder='passthrough')
step2=RandomForestRegressor(n_estimators=100, random_state=42)
pipe=Pipeline([
    ('step1',step1),
    ('step2',step2)
])
pipe.fit(x_train,y_train)


# In[95]:


y_pred=pipe.predict(x_test)
from sklearn.metrics import r2_score
score=r2_score(y_pred,y_test)
score


# In[85]:


import pickle
with open('laptop_price_model.pkl', 'wb') as f:
    pickle.dump(pipe, f)


# In[94]:


import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Helper function to extract a simplified CPU identifier
def extract_cpu(cpu_full):
    cpu_full = cpu_full.lower()
    if "i5" in cpu_full:
        return "i5"
    elif "i7" in cpu_full:
        return "i7"
    elif "i3" in cpu_full:
        return "i3"
    elif "quad" in cpu_full:
        return "quad"
    elif "ryzen" in cpu_full:
        return "ryzen"
    else:
        return cpu_full

# Load your pre-trained model
with open('laptop_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Laptop Price Predictor")

st.sidebar.header("Input Features")

# Categorical selections
company = st.sidebar.selectbox("Company", ['Apple', 'HP', 'Lenovo', 'Acer', 'Asus', 'others', 'Toshiba'])
type_name = st.sidebar.selectbox("TypeName", ['Ultrabook', 'Notebook', '2 in 1 Convertible'])
os_value = st.sidebar.selectbox("OS", ['Windows', 'Mac', 'others'])
gpu = st.sidebar.selectbox("GPU", ['Intel', 'AMD', 'Nvidia'])
memory_type = st.sidebar.selectbox("Memory Type", ['SSD', 'HDD'])

# Numeric inputs and sliders
touchscreen = st.sidebar.checkbox("Touchscreen")
ram = st.sidebar.slider("RAM (GB)", min_value=4, max_value=32, value=8, step=4)
inches = st.sidebar.number_input("Inches", min_value=10.0, max_value=20.0, value=15.0, step=0.1)
weight = st.sidebar.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
pixel_count = st.sidebar.number_input("Pixel Count", min_value=1000000, max_value=10000000, value=2073600, step=100000)

# CPU selection: Use the full CPU name for display then derive a simplified version for the 'cpu' feature.
cpu_full = st.sidebar.selectbox("CPU", ['Intel Core i5', 'Intel Core i7', 'Intel Core i3', 'AMD Ryzen', 'Intel Pentium Quad'])
cpu = extract_cpu(cpu_full)

# For the 'cpu_name' feature, based on your training data this seems to be a flag. We'll set it to 1.
cpu_name_feature = 1

# Compute interaction features that were used in training
ram_weight = ram * weight
inches_weight = inches * weight

# Build the input data dictionary.
# Note: Column names must match exactly what your model was trained on.
input_data = {
    'Company': company,
    'TypeName': type_name,
    'touchscreen': int(touchscreen),
    'os': os_value,
    'GPU': gpu,
    'Memory_type': memory_type,
    'Ram_weight': ram_weight,
    'Inches_weight': inches_weight,
    'cpu': cpu,
    'cpu_name': cpu_name_feature,
    'pixel': pixel_count
}

# Create a DataFrame from the input data.
input_df = pd.DataFrame([input_data])

if st.sidebar.button("Predict Price"):
    # The model was trained on log_price.
    # Predict the log_price then reverse the transformation.
    prediction_log = model.predict(input_df)
    predicted_price = np.exp(prediction_log)
    
    st.subheader("Predicted Price")
    st.write(f"${predicted_price[0]:,.2f}")


# In[92]:


x_train


# In[91]:


y_train


# In[ ]:




