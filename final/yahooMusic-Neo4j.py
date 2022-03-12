import pandas as pd
import glob
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, islice
import csv
from datetime import datetime, timedelta
import time

# parallelization
import multiprocessing as mp
import threading
import queue
global lck 
lck = threading.Lock()
import tqdm

from neo4j import GraphDatabase


class Neo4jConnection:
    
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response

conn = Neo4jConnection(uri="bolt://localhost:7687", user="pyWrite", pwd="pyWrite")    
# conn = Neo4jConnection(uri="bolt://localhost:7687", user="pyRead", pwd="pyRead")


def write_user(package, outfile='./recommendations-neo4j.csv'):  
    lck.acquire()
    with open(outfile, 'a',encoding='utf-8-sig', newline='\n') as g:
        keys = package[0].keys() # get keys off first rating
        dict_writer = csv.DictWriter(g, keys)
        dict_writer.writerows(package)
    
    lck.release()


def process_user(i, infile='ee627a-2019fall/testItem2.txt'):
    with open(infile, "r") as f:
        lines_gen = islice(f, i*7, (i+1)*7) # get line index for userID & their last target song
        cur_set = [x.strip('\n').split('|') for x in lines_gen]
    test_user = cur_set[0][0] # pull user ID. don't need song count  199810_208019
    distances = []
    for song in cur_set[1:]:
        query = f'''
        MATCH (p1:User {{id: {test_user}}})-[x:RATED]->(m:Track)
        WITH p1, gds.alpha.similarity.asVector(m, x.weight) AS p1avg
        MATCH (p2:User)-[y:RATED]->(m:Track) WHERE p2 <> p1
        WITH p1, p2, p1avg, gds.alpha.similarity.asVector(m, y.weight) AS p2avg
        WITH p1, p2, gds.alpha.similarity.pearson(p1avg, p2avg, {{vectorType: "maps"}}) AS pearson
        ORDER BY pearson DESC
        MATCH (p2)-[r:RATED]->(m:Track {{id: {song[0]}}}) 
        RETURN m.id as targetSongID, SUM(pearson * r.weight) AS score
        ORDER BY score DESC'''
        dist=conn.query(query, db='neo4j') # create track belongs to genre edge
        distances.append((f'{test_user}_{song[0]}', dist[0]['score']))
    distances.sort(key=lambda x:x[1])
    predictions = []
    for i, j in enumerate(distances):
        cur_dict = {}
        cur_dict['TrackID'] = j[0]
        if i < 3:
            cur_dict['Predictor'] = 0 # sorted ascending, poor pearson score should not be recommended
        else:
            cur_dict['Predictor'] = 1
        predictions.append(cur_dict)

    return predictions




infile='ee627a-2019fall/testItem2.txt'
outfile='./recommendations-neo4j.csv'

def run_mp(j):            
    predictions_processed = process_user(j)
    write_user(predictions_processed)
    
if __name__ == '__main__':
    ################### add desired output columns
    with open(outfile ,'w') as oufl, open(infile, 'r', encoding='utf-8') as infl:
        oufl.write('TrackID,Predictor\n')
        row_count = sum(1 for row in infl)
    print('total rows:', row_count) 
    n_users = int((row_count)/7)
    tasks = list(range(n_users))



        
    print('start time: {}'.format(datetime.now().strftime("%Y-%m-%d-%H.%M.%S")))
    start_time = time.time()


    num_workers = 8 #mp.cpu_count()
    print('num workers avail:', num_workers)

    ## multi proc
    pool = mp.Pool(num_workers)
    pool.imap_unordered(run_mp, tasks)

    pool.close() 
    pool.join() 

    ## single proc
    # for t in tasks:
        # run_mp(t)

    print('finished. end time: {}'.format(datetime.now().strftime("%Y-%m-%d-%H.%M.%S")))
    print('completed in {}'.format(timedelta(seconds=int(time.time() - start_time))))
