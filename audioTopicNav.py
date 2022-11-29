"""
This file allows for the batch processing of 
"""
__author__ = "Cas Laskowski"
__contact__ = "caslaskowski@arizona.edu"
__date__ = "08/06/2022"
__license__ = "GPLv3"
__status__ = "Production"
__version__ = "1.0.0"

import os
import time
import transcriptProcessing as tp # has methods for sorting json and processing the transcript text
import semanticAnalysis as sa # has methods for semantic analysis through sentence_transformers model
import pandas as pd
from ast import literal_eval # for reading in the Embddings column as a list not as a string


# * set phrases, words, and regex that should be removed from transcript files

# removes slutations and these random solo Es and Os in the transcript 
regex_expressions = ["(Hello, \w+\s*\w*\.)","((?<!\w)[EeOoUu](?!\w))"]
    # repeat punctuation and whitespace already handled in the function

# removes phrases that were seen repeatedly in the transcripts and create insignificant clusters
stop_phrases = ['Yeah, yeah, yeah.','Yeah, right.','Yeah, Yeah.', 'You understand', 'No way', 'Wait a minute.', 'Yes, way.', 'Right, right.','Thank you.','That\'s good. ','Don\'t you think so?','You see.','I see','That\'s that.','Of course.','You see what I mean?','See what I mean.','You agree?','I agree','my point','your point','Hello','Mm hmm.','Uh huh.','You have.','I have.','That\'s great.','Yeah, I know.','Is that right?','Is that correct?','Why not?','It is.','Okay, Okay.','No, no.','no, no','Oh, hey.','You see.','Don\'t you agree','Well, okay.', 'Okay, well,','That\'s right.','I mean.','That\'s true.','But what?','You know what?','I know.','Oh, God.','Oh, my God.','So what?','All right, fine.','I understand.','Yes, yes', 'No, sir.','What happened?','That happened','I think so','Oh, yeah','Uh, yeah','Come on.','Okay, fine.','That\'s correct', 'Yeah, that\'s right.','Yeah, sure.','yeah, yeah,','I don\'t know.','That\'s what I mean','I think.','Yeah, yeah, yeah.','Uh oh.','Oh, Okay.','Uh, no.','Is it?','I did.','Very good, very good.','Right, right, right.', 'Oh, no.','That\'s all.','Yeah, Yeah, all right.','Yeah, Yeah, There are.','Uh Yeah.','Well, yes, actually.','In a way that way.','So yeah, All right.','a uh, Right.','Yeah, a all right.','a a Yeah, yeah.','Uh, that\'s pretty good.','Well, that\'s all this.','Well, that\'s true.','Oh, uh, yes.','I mean, that.','Well, you know.','That\'s this Fine.','Yeah, You know what I mean.','Yeah, yeah, no.']

# removes filler words - it is not recommended to remove all stop words as it will affect the accuracy of the embeddings
stop_words = ['huh','uhh','uh','um','mhm','hmm']

# * Preprocess and clusters batches of transcript files. Creates csv for each

# 1. Get list of files to processes
file_directory = 'SCADS/processed_json_20220223_improved_dates'
file_list = os.listdir(file_directory)

# 2. Get a list of batches for list slicing
batch_sizes = 100
slices = [*range(0, len(file_list), batch_sizes)]
slices.pop(0)
slices.append(len(file_list))

# 3. Set working directory for outputs and output file names
working_directory = 'SCADS/SliceClusters2'
    # don't add csv or txt as code adds iteration and file type to the name
df_csv_file = 'transcripts_clusters_tab' 
clusters_txt_file = "HDBscan_clusters"

# 4. Set variables for iterating over the file_list
start = 0
iteration = 0

# 5. Iterate over file list to process and cluster sections of the file_list
for slice in slices:
    
    # 5.a Get a slice of the list
    slice_of_list = file_list[start:slice]

    # 5.b Set iteration variables
    start = slice # set for next iteration
    iteration += 1 # ensures final iteration value will match the number of iterations carried out 
    print('='*50)
    print('Beginning processing for batch {0} OF {1}!!'.format(str(iteration),len(slices)))
    print('='*50)  

    # 5.c Preprocess and get sentence embeddings of all files in the list
    transcripts_df = tp.batch_processing(slice_of_list, file_directory, working_directory, 'duplicates_log', iteration, regex_expressions, stop_phrases, stop_words) 
    # batch_processing(file_list, file_directory, working_directory, duplicate_log, iteration, regex_list=[], phrase_list=[], word_list=[], diarize = False, divide_by = 'sentence', min_char=6, min_words=2):

    # 5.d Get sentence embeddings for all items in the datafrome
    transcripts_df['Embeddings'] = sa.embed_divisions(transcripts_df['CleanSeg'].tolist())

    # 5.e Determine clusters for sentences in the dataframe
    transcripts_df['Clusters'], transcripts_df['Probabilities'] = sa.cluster_divisions(transcripts_df['Embeddings'].tolist(), min_cluster_mem=5)
   
    # 5.f Create a text file with all significant (higher than -1) clusters
    cluster_file = working_directory + '/' + clusters_txt_file + '_iter' + str(iteration) + '.txt'

    with open(cluster_file, 'a') as cluster_file:
        clustered_sentences = {}
        for df_row, cluster_id in enumerate(transcripts_df['Clusters'].tolist()):
            if cluster_id > 0: 
                if cluster_id not in clustered_sentences:
                    clustered_sentences[cluster_id] = []

                clustered_sentences[cluster_id].append(transcripts_df.iloc[df_row]['CleanSeg'])

        cluster_file.write("Topical Clusters from Transcripts \n")
        cluster_file.write('-'*60)
        for i, cluster in clustered_sentences.items():
            cluster_file.write("\n")
            cluster_number = "Cluster " + str(i+1)
            cluster_file.write(cluster_number)
            cluster_file.write("\n")
            cluster_file.write('. '.join(cluster))
            cluster_file.write("\n")
            cluster_file.write('-----\n')

    # 5.g Store the dataframe in a csv
    csv_file = df_csv_file + '_iter' + str(iteration) + '.csv'

    tp.create_csv(transcripts_df, working_directory, csv_file)
    # create_csv(dataframe, working_directory, output_file)
    
    print("\n\n")
    print('='*50)
    print('PROCESSING FULLY COMPLETED FOR ITERATION {0} OF {1}!'.format(str(iteration),len(slices)))
    print('='*50)  

# * testing csv accuracy and load time

# csv_file = output_filename + '.csv'

# with open(csv_file) as file:
#     print('='*50)
#     print("Opening csv file")
#     start_time = time.time()
    
#     transcripts_df = pd.read_csv(file,sep="\t",converters={'Embeddings': literal_eval})
        
#     minutes_processing, seconds_processing = divmod((time.time() - start_time), 60)

#     print("Processing done after {0:.0f} mins and {1:.2f} secs".format(minutes_processing, seconds_processing))
#     print("There are {0} rows in the csv file.".format(transcripts_df.shape[0]))
#     print("The column names are: {0}.".format(list(transcripts_df)))
#     print("The embeddings column has values of type: {0} ".format(type(transcripts_df['Embeddings'].iloc[0])))
#     print('='*50)

# * get Apollo gist list
# apollo_gists = 'SCADS/apolloGists2.csv'

# apollo_conversations = []
# with open(apollo_gists,'r') as file:
#     apollo_df = pd.read_csv(file, sep=',')
#     apollo_conversations = apollo_df['convoCAP'].tolist()

# * Get similarity from apollo gists
# tp.create_embeddings_csv(apollo_files, directory, final_csv)

# with open(final_csv) as file:
#     transcripts_df = pd.read_csv(file,sep="\t",converters={'Embeddings': literal_eval})

#     sample_text = 'viewfinder for children'
#     embeddings = transcripts_df['Embeddings'].tolist()

#     transcripts_df['SimilarityScores'] = sa.determine_similarity(sample_text, embeddings)

#     filtered_df = transcripts_df.nlargest(10,"SimilarityScores")

#     print(filtered_df['CleanSeg'])

# * Notes for temporal clustering return efforts
# Create self._clusteredSentences(?)
# Get max value in cluster column
# For i in range (0,max_value):
#     Get the first timestamp where that value appears
#     Get the last value
#     Create time range of 'topic'

# * for inevitable get of time range section of transcript
    # print(transcript._startTime)

    # audio_date = transcript._startTime
    # timestamp = datetime.strptime('0:14:11', '%H:%M:%S').time()
    # new_date = datetime.combine(audio_date, timestamp)
    # print(new_date)

    # timestamp_list = transcript._transcript_df['TimeStamp']
                
    # # printing original list
    # print("The original list is : " + str(test_date_list))
    
    # # initializing test date 
    # test_date = datetime(2017, 6, 6)
    
    # # shorthand using lambda function for compact solution
    # res = min(test_date_list, key=lambda sub: abs(sub - test_date))
    
    # # printing result
    # print("Nearest date from list : " + str(res))