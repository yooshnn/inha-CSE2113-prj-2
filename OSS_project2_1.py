import pandas as pd

def requirement_1(df):
  def top_10(df, by):
    res = df.sort_values(by, ascending=False).head(10)['batter_name']
    return ', '.join(res.tolist())
  
  for year in range(2015, 2019):
    year_df = df[df['year'] == year]
    
    print(f'Top 10 players for year {year}')
    print(f'hits : {top_10(year_df, "H")}')
    print(f'batting average : {top_10(year_df, "avg")}')
    print(f'homerun : {top_10(year_df, "HR")}')
    print(f'on-base percentage : {top_10(year_df, "OBP")}')
    print()

def requirement_2(df):
  def top(df):
    res = df.sort_values('war', ascending=False).head(1)['batter_name']
    return res.to_string(index=False)

  positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
  df_2018 = df[df["year"] == 2018]
  
  print('Player with the highest war by position in 2018')
  for position in positions:
    print(f'{position} : {top(df_2018[df_2018["cp"] == position])}')
  print()

def requirement_3(df):
  interest = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']
  correlation = df[interest].corr(method='pearson').loc['salary']
  res = correlation.drop('salary').idxmax()
  print(f'{res} has the highest correlation with salary.')
  
if __name__=='__main__':
  data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
  requirement_1(data_df)
  requirement_2(data_df)
  requirement_3(data_df)
