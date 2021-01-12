import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('US_Accidents_June20.csv')

# print(df.isna().sum())

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Extract year, month, day, hour and weekday
df['Year']=df['Start_Time'].dt.year
df['Month']=df['Start_Time'].dt.strftime('%b')
df['Date']=df['Start_Time'].dt.day
df['Hour']=df['Start_Time'].dt.hour
df['Day']=df['Start_Time'].dt.strftime('%a')
df['Duration']=round((df['End_Time']-df['Start_Time'])/np.timedelta64(1,'m'))

df.drop(df[df['Year'] != 2019].index, inplace = True)
data = df[0:5000]
print(data.shape)


corr_matrix = data.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="seismic")
plt.gca().patch.set(hatch="X", edgecolor="#666")
plt.show()

data.to_csv('US_Accidents_2019.csv')
print("Saved!")

