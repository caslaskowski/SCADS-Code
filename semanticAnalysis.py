"""

"""
__author__ = "Cas Laskowski"
__contact__ = "caslaskowski@arizona.edu"
__date__ = "08/06/2022"
__license__ = "GPLv3"
__status__ = "Production"
__version__ = "1.0.0"

import time
import numpy as np
# For NLP methods
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
# from transformers import pipeline
# summarization = pipeline("summarization",model="roberta-large-mnli",max_length=20)
# For clustering sentences
from sklearn.cluster import AgglomerativeClustering
import hdbscan

def embed_divisions(segment_list):

    print('='*50)
    print("Embedding the transcript sentences. This may take a while.")
    start_time = time.time()

    if isinstance(segment_list,str):
        embeddings_list = model.encode(segment_list)

    else:
        # 1. Get the embeddings for each sentences using the sentence_transformers model.
        corpus_embeddings = model.encode(segment_list)

        # 2. Add the embeddings to a column of self._transcript_df.
        embeddings_list = np.array(corpus_embeddings).tolist()
    
        minutes_processing, seconds_processing = divmod((time.time() - start_time), 60)

        
        print("Embeddings done after {0:.0f} mins and {1:.2f} secs.".format(minutes_processing, seconds_processing))
        # print(self._transcript_df)
        print('='*50)
    
    return embeddings_list

    # def produce_summary(self,input_text):
    #     summary_text = summarization(input_text)[0]['summary_text']
    #     print('='*50)
    #     print("Summary:", summary_text)
    #     print('='*50)
    

def cluster_divisions(embeddings_list, method = 'hdbscan', distance_threshold=1.5,min_cluster_mem=5):

    print('='*50)
    start_time = time.strftime("%H:%M:%S", time.localtime())
    print("Clustering the transcript sentences using {0} started at {1}.".format(method,start_time))
    print("This may take hours. Your patience is appreciated.")
    start_time = time.time()

    if method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_mem)
        clusterer.fit(embeddings_list)
        # From HDBSCAN documentation:
        # HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
        #     gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
        #     metric='euclidean', min_cluster_size=5, min_samples=None, p=None)

        minutes_processing, seconds_processing = divmod((time.time() - start_time), 60)
        hours_processing, minutes_processing = divmod(minutes_processing, 60)

        print("Clustering done after {0:.0f} hours {1:.0f} mins and {1:.2f} secs".format(hours_processing, minutes_processing, seconds_processing))
        print('='*50)

        return clusterer.labels_, clusterer.probabilities_

    elif method == 'agglomerative':
    
        # Normalize the embeddings to unit length
        corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        # Perform kmean clustering
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
        #, affinity='cosine', linkage='average', distance_threshold=0.4)
        clustering_model.fit(corpus_embeddings)
        
        minutes_processing, seconds_processing = divmod((time.time() - start_time), 60)
        hours_processing, minutes_processing = divmod(minutes_processing, 60)

        print("Clustering done after {0:.0f} hours {1:.0f} mins and {1:.2f} secs".format(hours_processing, minutes_processing, seconds_processing))
        print('='*50)

        return clustering_model.labels_

def determine_similarity(text_to_compare, embeddings_list):

    base_embedding = embed_divisions(text_to_compare)

    similarity_scores = util.cos_sim(base_embedding, embeddings_list)

    return similarity_scores[0]
    
def find_similar_segments(transcript_df, text_to_compare, threshold = .5, max_number_files = 5):
    
    transcript_df["CosineScores"] = determine_similarity(text_to_compare, transcript_df["Embeddings"])
    filtered_df = transcript_df.nlargest(max_number_files,"CosineScores")
    

# * Use below code to run similiarity analysis on small subset
# from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('all-MiniLM-L6-v2')

# with open('SCADS/processed_json_20220223_improved_dates/007-023.mp3.json', 'r') as f:
#     file = f.read()

#     transcript = tp.Transcript(file)

#     transcript.divide_by_sentence()
#     transcript._transcript_df['Embeddings'] = sa.embed_divisions(transcript._transcript_df['CleanSeg'].tolist())

#     base_embedding= sa.embed_divisions("Apollo viewfinder gifts for kids")

#     similarity_scores = util.cos_sim(base_embedding, transcript._transcript_df['Embeddings'].tolist())
#     print("First score is {0}".format(similarity_scores[0][0]))

#     transcript._transcript_df["CosineScores"] = similarity_scores[0]
#     print (transcript._transcript_df)

#     filtered_df = transcript._transcript_df.nlargest(10,"CosineScores")

#     print(filtered_df['CleanSeg'])