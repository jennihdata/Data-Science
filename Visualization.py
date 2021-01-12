import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('US_Accidents_2019_cleaned.csv')
print(df.shape)

#Number of accidents per state
state_counts = df["State"].value_counts()
plt.figure(figsize=(20, 8))
plt.title("Top 10 states with the highest number of accidents")
sns.barplot(state_counts[:10].values, state_counts[:10].index, orient="h")
plt.xlabel("Number of accident")
plt.ylabel("State")
plt.show()

# Distance affecting the severity of an accident
severity_distance = df.groupby("Severity").mean()["Distance(mi)"].sort_values(ascending=False)
plt.figure(figsize=(20, 8))
plt.title("Medium distance by severity")
sns.barplot(severity_distance.values, severity_distance.index, orient="h", order=severity_distance.index)
plt.xlabel("Distance (mi)")
plt.show()


# Weather Condition
plt.figure(figsize=(14,8))
df.groupby('Weather_Condition') \
    .size() \
    .sort_values(ascending = False) \
    .plot.pie(autopct='%1.1f%%')
plt.show()

# Bar plot version
counts = df["Weather_Condition"].value_counts()[:15]
plt.figure(figsize=(20, 8))
plt.title("Histogram distribution of the top 15 weather conditions")
sns.barplot(counts.index, counts.values)
plt.xlabel("Weather Condition")
plt.ylabel("Value")
plt.show()


# # Distance affecting the severity of an accident
# severity_distance = df.groupby("Severity").mean()["Weather_Condition"].sort_values(ascending=False)
# plt.figure(figsize=(20, 8))
# plt.title("Weather Condition by severity")
# sns.barplot(severity_distance.values, severity_distance.index, orient="h", order=severity_distance.index)
# plt.xlabel("Weather Condition")
# plt.show()


# Accident Counts every weekday

# Bar plots
counts = df["Day"].value_counts()
weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
plt.figure(figsize=(20, 8))
plt.title("Number of accidents for each weekday")
sns.barplot(counts.index, counts.values, order=weekdays)
plt.xlabel("Weekday")
plt.ylabel("Value")
plt.show()

plt.figure(figsize=(14,8))
df.groupby('Day') \
    .size() \
    .sort_values(ascending = False) \
    .plot.pie(autopct='%1.1f%%')
plt.show()

# Accident Counts every weekday

# Bar plots

counts = df["Hour"].value_counts()
plt.figure(figsize=(20, 8))
plt.title("Histogram distribution of the time of the accident")
sns.barplot(counts.index, counts.values)
plt.xlabel("Hour")
plt.ylabel("Value")
plt.show()

plt.figure(figsize=(14,8))
df.groupby('Hour') \
    .size() \
    .sort_values(ascending = False) \
    .plot.pie(autopct='%1.1f%%')
plt.show()

# Happens Night or Day


counts = df["Civil_Twilight"].value_counts()
plt.figure(figsize=(20, 8))
plt.title("Histogram distribution of the time of the accident")
sns.barplot(counts.index, counts.values)
plt.xlabel("Civil_Twilight")
plt.ylabel("Value")
plt.show()

plt.figure(figsize=(14,8))
df.groupby('Civil_Twilight') \
    .size() \
    .sort_values(ascending = False) \
    .plot.pie(autopct='%1.1f%%')
plt.show()





# SEVERITY
# f,ax=plt.subplots(1,3,figsize=(15,5))
# df['Severity'].value_counts().plot.pie(explode=[0,0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
# ax[0].set_title('Percentage Severity Distribution')
# #ax[0].set_ylabel('Count')
# sns.countplot('Severity',data=df,ax=ax[1],order=df['Severity'].value_counts().index)
# ax[1].set_title('Count of Severity')
# df.Severity.value_counts(normalize=True).sort_index().plot.bar(ax=ax[2])
# ax[2].set_title('Severity Percentage')
# ax[2].set_xlabel('Severity')
# ax[2].set_ylabel('Percentage')
# #plt.grid()
# #plt.title('Severity')
# #plt.xlabel('Severity')
# #plt.ylabel('Fraction');
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(15,5))
# sns.countplot(x='Hour', hue='Severity4', data=df ,palette="Set2")
# plt.title('Count of Accidents by Hour (resampled data)', size=15, y=1.05)
# plt.show()

