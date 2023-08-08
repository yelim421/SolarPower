import pandas as pd

with open('merged0110_kr.csv', 'r', encoding='cp949', errors = "ignore") as f:
	df = pd.read_csv(f, dtype={4:str})

print(df.columns)
print(len(df.columns))

new_names = ['inverter_id', 'location', 'date', 'dataYN', 'inverter_V(R)', 'inverter_V(S)', 'inverter_V(T)', 'inverter_I(R)', 'inverter_I(S)', 'inverter_I(T)', 'inverter_f', 'active_P(total)', 'reactive_P(total)', 'PF(total)', 'module_P_BI(PV)', 'module_V_BI(PV)', 'module_I_BI(PV)', 'inverter_P_AI',
		'cumulative_P_AI', 'today_P_AI', 'out_T', 'inclined_solar', 'horizontal_solar', 'module_T']

df.columns = new_names

print(df.columns)
df.to_csv('merged_col.csv', index=False, encoding='cp949')

with open('merged_col.csv', 'r', encoding='cp949', errors = "ignore") as f:
	df = pd.read_csv(f, dtype={4:str})

unique_locations = df['location'].unique()
print("unique loc.:", unique_locations)

num_u_l = df['location'].nunique()
print("num of uniq loc:", num_u_l)

loc_count = df['location'].value_counts()
print("counts:")
print(loc_count)

#cp949
#euc-kr
