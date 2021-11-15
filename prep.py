import pandas as pd
import numpy as np
data = pd.read_pickle('data.p')
# print(data)
df_user = data.drop(['movie_id', 'rank', 'movie_title', 'movie_type'], axis=1)
df_movie = data.drop(['user_id', 'rank', 'user_gender', 'user_age', 'user_job'], axis=1)
# print(df_user)
# print(df_movie)
df_user = df_user.drop_duplicates(['user_id'])
df_movie = df_movie.drop_duplicates(['movie_id'])
# print(df_user)
# print(df_movie)

df_user = df_user.sort_values(by="user_id",ascending=True)
df_movie = df_movie.sort_values(by="movie_id",ascending=True)
df_user = df_user.reset_index(drop=True)
df_movie = df_movie.reset_index(drop=True)
print(df_user)
print(df_movie)

j = 0
cp_line = df_user.iloc[0]
user = pd.DataFrame(np.ones(4*6041, dtype=int).reshape(6041, 4),columns=['user_id','user_gender','user_age', 'user_job'])
for i in range(6041):
    line = df_user.iloc[j]
    uid = line[0]
    if uid > i:
        user.loc[i] = cp_line
    else:
        user.loc[i] = line
        j += 1

j = 0
cp_line = df_movie.iloc[0]

movie = pd.DataFrame(columns=['movie_id','movie_title','movie_type'])
movie = movie.append(cp_line, ignore_index=True)
print("##############")
print(movie)
print("##############")
for i in range(3953):
    line = df_movie.iloc[j]
    mid = line[0]
    if mid > i:
        movie.loc[i] = cp_line
    else:
        movie.loc[i] = line
        j += 1
print(movie)

user.to_pickle('user.pickle')
movie.to_pickle('movie.pickle')
