import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

df= pd.read_csv('US_Accidents_2019.csv')

pd.set_option('display.max_column',20)

# print(df.isnull().sum().sort_values(ascending=False))
# print('Attributes with > 20% missing values: ', df.columns[(100*df.isnull().sum()/df.shape[0]).round(2)>20].tolist())

# Drop unnecessary columns
df = df.drop(['ID','Source','TMC','End_Lat','End_Lng','Description','Number','Street','Side','County',
              'Country','Zipcode','Timezone','Airport_Code','Weather_Timestamp','Wind_Chill(F)',
              'Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop',
              'Traffic_Calming','Traffic_Signal','Turning_Loop','Sunrise_Sunset','Nautical_Twilight',
              'Astronomical_Twilight','Start_Time','End_Time', "Precipitation(in)","Year", "Month"], axis=1)

# Checking if there is duplicates
print(f'Percentage of duplicate records: {100-(100*df.drop_duplicates().shape[0]/df.shape[0])}')
df.drop_duplicates(inplace=True)
print(f'Percentage of duplicate records: {100-(100*df.drop_duplicates().shape[0]/df.shape[0])}')

# # Checking missing values
# missing_values = df.isna().sum() / len(df)
# print(missing_values)
#
# # Visualization
# plt.figure(figsize=(14, 14))
# plt.title('Missing Data')
# plt.xlabel('% of rows with missing data')
# missing_values.plot(kind='barh');
# plt.show()

# print(df.isnull().sum().sort_values(ascending=False))

# print(df['Weather_Condition'].unique())

# Fill missing numerical columns with the mean
fill_with_mean = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)"]
df[fill_with_mean] = df[fill_with_mean].fillna(df[fill_with_mean].mean())

# Handle Structural Error - Making Wind_Direction more consistent and simpler
df.loc[(df['Wind_Direction']=='WSW')|(df['Wind_Direction']=='WNW'),'Wind_Direction'] = 'W'
df.loc[(df['Wind_Direction']=='SSW')|(df['Wind_Direction']=='SSE'),'Wind_Direction'] = 'S'
df.loc[(df['Wind_Direction']=='NNW')|(df['Wind_Direction']=='NNE'),'Wind_Direction'] = 'N'
df.loc[(df['Wind_Direction']=='ESE')|(df['Wind_Direction']=='ENE'),'Wind_Direction'] = 'E'
df.loc[df['Wind_Direction']=='Variable','Wind_Direction'] = 'VAR'

# Handle Structural Error - Making Weather_Condition more consistent and simpler
df.loc[df["Weather_Condition"].str.contains("Thunder|T-Storm", na=False), "Weather_Condition"] = "Thunderstorm"
df.loc[df["Weather_Condition"].str.contains("Snow|Sleet|Wintry", na=False), "Weather_Condition"] = "Snow"
df.loc[df["Weather_Condition"].str.contains("Rain|Drizzle|Shower", na=False), "Weather_Condition"] = "Rain"
df.loc[df["Weather_Condition"].str.contains("Wind|Squalls", na=False), "Weather_Condition"] = "Windy"
df.loc[df["Weather_Condition"].str.contains("Hail|Pellets", na=False), "Weather_Condition"] = "Hail"
df.loc[df["Weather_Condition"].str.contains("Fair", na=False), "Weather_Condition"] = "Clear"
df.loc[df["Weather_Condition"].str.contains("Cloud|Overcast", na=False), "Weather_Condition"] = "Cloudy"
df.loc[df["Weather_Condition"].str.contains("Mist|Haze|Fog", na=False), "Weather_Condition"] = "Fog"
df.loc[df["Weather_Condition"].str.contains("Sand|Dust", na=False), "Weather_Condition"] = "Sand"
df.loc[df["Weather_Condition"].str.contains("Smoke|Volcanic Ash", na=False), "Weather_Condition"] = "Smoke"
df.loc[df["Weather_Condition"].str.contains("N/A Precipitation", na=False), "Weather_Condition"] = np.nan

# print(df["Weather_Condition"].unique())

# If wind direction 'calm',  change wind_speed to 0, because it means there are little to no wind
df.loc[df['Wind_Direction'] == 'CALM', 'Wind_Speed(mph)'] = 0

# Handle missing data - Filling NaN's with its mode
df['Weather_Condition'].fillna(str(df['Weather_Condition'].mode()[0]),inplace = True)
df['Wind_Direction'].fillna(str(df['Wind_Direction'].mode()[0]),inplace = True)

# print(df.isna().sum())
print(df.isnull().sum().sort_values(ascending=False))

df.to_csv('US_Accidents_2019_cleaned.csv')
print("Saved!")

