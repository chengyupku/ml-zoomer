
from model import rec_model
import torch
import pandas as pd
from tqdm import tqdm
import pickle
moviedf = pd.read_csv('./ml-25m/movies.csv')
movie_len = len(moviedf)
genre_map = {}
for i in range(movie_len):
    line = moviedf.iloc[i]
    genrestr = line[-1]
    genres_list = genrestr.strip().split("|")
    for g in genres_list:
        if g not in genre_map:
            genre_map[g] = len(genre_map)+1

print(genre_map)
movie_genre = []
mg_len = 0
for i in tqdm(range(movie_len)):
    line = moviedf.iloc[i]
    mid = int(line[0])
    if mid > mg_len:
        for j in range(mid-mg_len):
            movie_genre.append([0])
        mg_len = mid
    genrestr = line[-1]
    genres_list = genrestr.strip().split("|")
    gl = []
    for g in genres_list:
        gl.append(genre_map[g])
    movie_genre.append(gl)
    mg_len += 1
    # movie_genre[str(mid)] = gl

print(movie_genre)
print(len(movie_genre))
with open('./datalist/movie_genre_list.pkl', 'wb') as f:
	# pickle.dump(u_movie_list, f, pickle.HIGHEST_PROTOCOL)
	# pickle.dump(m_user_list, f, pickle.HIGHEST_PROTOCOL)
	# pickle.dump(m_query_list, f, pickle.HIGHEST_PROTOCOL)
	# pickle.dump(q_movie_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(movie_genre, f, pickle.HIGHEST_PROTOCOL)