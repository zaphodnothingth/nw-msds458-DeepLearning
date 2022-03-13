import pandas as pd
import glob
import matplotlib.pyplot as plt
from itertools import combinations, islice
import csv
from datetime import datetime, timedelta
import time
import pickle

from itertools import islice



# process tracks data file
rows = []
for row in open('ee627a-2019fall/trainItem2.txt'):
    if '|' in row:
        cur_user = row.strip('\n').split('|')[0] # pull user ID. don't need song count
        continue # skip to the user's ratings
    row = row.strip('\n').split('\t')
    row_dict = {'user': cur_user, 
                'track': row[0], 
                'rating': int(row[1])
               }
    rows.append(row_dict)

df_train = pd.DataFrame(rows)
df_train['rating'] = df_train['rating'].replace({0:1})

# prepare data for surprise
from surprise import Reader, Dataset, SVD, KNNWithMeans, NMF
from surprise.model_selection.validation import cross_validate
svd = SVD()

reader = Reader(rating_scale=(0, 100))
data = Dataset.load_from_df(df_train, reader)
trainset = data.build_full_trainset()


# build model
start_time = time.time()
print('start time: {}'.format(datetime.now().strftime("%Y-%m-%d-%H.%M.%S")))




## KNN cosine similarity
#sim_options = {
#    "name": "cosine",
#    "user_based": True,  # Compute  similarities between items
#}
#algo = KNNWithMeans(sim_options=sim_options)

## NMF
algo = NMF(n_factors = 16)


algo.fit(trainset)
pickle.dump(algo, open('model.pkl', 'wb'))

print('finished training. end time: {}'.format(datetime.now().strftime("%Y-%m-%d-%H.%M.%S")))
print('completed in {}'.format(timedelta(seconds=int(time.time() - start_time))))




