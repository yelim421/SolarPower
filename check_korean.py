
import pandas as pd
import os

#files = [f for f in os.listdir('.') if f.endswith('.csv')]
#dfs = []

#with open('module20210102.csv', 'r', encoding = 'cp949', errors = "ignore") as f:
#	df = pd.read_csv(f, dtype={4:str})
#print(df.columns)

#unique_loc = df['Unnamed: 1'].unique()
#print(unique_loc)

with open('module202107.csv', 'r', encoding='cp949', errors="ignore") as f:
	df = pd.read_csv(f, dtype={4:str})
df=df.iloc[:, :-1]
print(len(df.columns))
df.to_csv('module202107_new.csv', index=False, encoding='cp949')

with open('module202107_new.csv', 'r', encoding = 'cp949', errors="ignore") as f:
	df = pd.read_csv(f, dtype={4:str})
print(len(df.columns))
print(df.columns)
