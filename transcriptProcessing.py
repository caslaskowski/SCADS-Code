#!/usr/bin/env python
""" Audio transcript segmentation and summarization model.
Program takes audio trancript data in string or json format and feeds it into
a class that pulls the metadata and provides several processing functions. 
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Cas Laskowski"
__contact__ = "caslaskowski@arizona.edu"
__date__ = "08/06/2022"
__license__ = "GPLv3"
__status__ = "Production"
__version__ = "1.0.0"

# For timestamp conversion and processing and time logs
from datetime import datetime, timedelta
import time
import pytz
# For procesing the json ingest and output
import json
# from python_dict_wrapper import wrap, unwrap, add_attribute
# For datawrangling
import re
import numpy as np
import pandas as pd
import os
from ast import literal_eval
# For semantic analysis
import semanticAnalysis as sa

# For managing progress
# ? why are they not loading
from tqdm import tqdm
tqdm.pandas() 

# * Set expected timezone for transcript start time
tz = pytz.timezone('America/New_York')

class Transcript(object):

    def __init__(self, transcript):
        if type(transcript) == str:
            transcript = json.loads(transcript)
        
        # add_attribute(self._transcript, 'divTranscript', [2,2,4,5])

        # Assign class variables from JSON
        self._transcript_id = transcript.get('conversationNumber', None)
        self._title = transcript.get('conversationTitle', None)
        self._conversation = transcript.get('conversationNumber',None)
        self._startTimeUTC = transcript.get('startDateTime', None)
        self._startTime = datetime.fromtimestamp(self._startTimeUTC/1000 if self._startTimeUTC != 'None' else None, tz)
        self._duration = transcript.get('duration', None)
        self._summaryURL = transcript.get('summaryURL', None)
        # This will pull the speakers from the json and put them in a list
        self._speakers = [p for p in [p.get('name', None) for p in transcript.get('speakers', [])] if p != None]
        self._audioURL = transcript.get('audioURL', None)
        self._location = transcript.get('location')
        self._locationType = self._location[self._location.find("(")+1:self._location.find(")")]
        self._transcript = transcript.get('transcriptBySpeaker','')
        # These variables will remain empty until transcripts are processed to keep the original clean for experimental processing.
        self._cleanTranscript = {}
        self._transcript_div = {}
        self._transcript_df = pd.DataFrame()
        # All processes will add attributes to this variable to allow for easier return of json
        # self._json = wrap(transcript)

    # def get_json(self):
    #     """
    #     Creates a json file that includes any transcript processing conducted using class methods.
    #     """
    #     file = unwrap(self._json)
    #     return file

    # def get_location_type(self):
    #     if self._location

    def clean_transcript(self, regex_list=[], phrase_list=[], word_list=[]):

        transcript = self._transcript

        replacement_list = phrase_list + word_list + regex_list
        replacement_dict = {}

        for item in replacement_list:
            replacement_dict[item] = 0

        if regex_list:
            for expression in regex_list:

                replacement_count = len(re.findall(expression,transcript))
                replacement_dict[expression] = (replacement_count)

                transcript = re.sub(expression,'', transcript, flags=re.IGNORECASE)

        if phrase_list:
            for phrase in phrase_list:

                replacement_count = transcript.count(phrase)
                replacement_dict[phrase] = (replacement_count)
                
                transcript = transcript.replace(phrase,"")

        if word_list:
            for word in word_list:

                replacement_count = transcript.count(word)
                replacement_dict[word] = (replacement_count)

                

        transcript = re.sub(' +', ' ', transcript) # remove extra whitespace

        repeating_punctuation = re.findall("([.?,!]\W[.?,!]*)",transcript)
        
        for item in repeating_punctuation:
            first_punctuation = item.split(' ')[0] +' '
            transcript = transcript.replace(item,first_punctuation)

        self._cleanTranscript = transcript

        return replacement_dict
        
    def remove_diarization(self):

        if self._cleanTranscript:
            transcript = self._cleanTranscript
        else:
            transcript = self._transcript

        # 1. Remove the any reference to speaker
        transcript = re.sub('spk_[0-9]+:', '', transcript)
        # 2. Remove any double spaces
        transcript = re.sub(' +', ' ', transcript)
        
        self._cleanTranscript = transcript

    def divide_by_speaker(self):
        
        # 1. Clean up artifact of bad data (space between \n)
        self._cleanTranscript = self._cleanTranscript.replace('\n \n','\n\n')

        # 2. Split into timestamp segments and remove empty strings
        split_transcript = [i for i in self._cleanTranscript.split('\n\n') if i]
        
        # 3. Index segements by timestamp
        self._transcript_div = {i[0:i.find(']')].replace('[', '').strip() : i[i.find(']') + 1:].strip() for i in split_transcript}

        # 4. Create a dataframe to support future methods
        self._transcript_df['TimeStamp'] = self._transcript_div.keys()
        self._transcript_df['Segment'] = self._transcript_div.values()
        self._transcript_df['File'] = self._title
        # Add date to timestamp
        transcript_date = self._startTime.strftime("%m/%d/%Y")
        self._transcript_df['TimeStamp'] = transcript_date + ', ' + self._transcript_df['TimeStamp']
        self._transcript_df['File'] = self._title

        # 5. Add endtimes to the dataframe
        segmentTimes = self._transcript_df['TimeStamp'].tolist()
        segmentTimes.pop(0) 
        # Create a endtime for the last segment by calculating transcript end from duration and start.
        transcript_end_time = self._startTime + timedelta(seconds =self._duration)
        
        segmentTimes.append(transcript_end_time.strftime("%H:%M:%S"))
        self._transcript_df.insert(1,'SegEnd',segmentTimes)

    def divide_by_sentence(self, min_char=6, min_words=2):
        
        # 1. Split the resulting segments by puntuation.
        self._transcript_df['CleanSeg'] = self._transcript_df['Segment'].str.split(pat='[?.!]',regex=True)
     
        # 2. Clean segments remove flag if works
        for row_ID, sentence_list in enumerate(self._transcript_df['CleanSeg'].tolist()):
            clean_sentences = [s.strip() for s in sentence_list if s and len(s.strip()) >= min_char and len(s.split()) >= min_words and isinstance(s, str)]
            self._transcript_df.at[row_ID,'CleanSeg']  = clean_sentences

        # 3. Create a new row for each resulting split sentence, and add secondary index by sentence order.
        self._transcript_df = self._transcript_df.explode('CleanSeg',ignore_index = True)
        self._transcript_df.insert(1,'SentenceID', self._transcript_df.groupby('TimeStamp').cumcount())

# TODO: Check this to see if it does what I think it does
def temporal_clusters(dataframe,multiple_files=True):

    clusters = dataframe['Clusters'].unique()

    cluster_file_segments = {}

    for cluster in clusters:
        cluster_file_segments[cluster] = {}

    if multiple_files == True:

        filenames = dataframe['File'].unique()

        for file in filenames:
            file_dataframe = dataframe.loc[dataframe['File'] == file]

            cluster_starttimes = {}
            clusters_processed = []

            for index, row in file_dataframe.iterrows():
                cluster = row['Clusters']
                if cluster != -1:
                    if cluster not in cluster_starttimes:
                        cluster_starttimes[cluster] = index
                        # cluster_timestamps[cluster] = {'start':row['Timestamp'],'end':''}
                        clusters_processed.append(cluster)

            for cluster in clusters_processed:
                end = file_dataframe.where(file_dataframe['Clusters']==cluster).last_valid_index()
                # cluster_starttimes[cluster]['end'] = end
                # cluster_timestamps[cluster]['end'] = file_dataframe.iloc[end]['TimeStamp']
                start = cluster_starttimes[cluster]

                cluster_segments = file_dataframe.iloc[start:end]
                starttime = cluster_segments.iloc['TimeStamp'][0]
                endtime = cluster_segments['TimeStamp'].iat[-1]
                
                segments = cluster_segments['Segments'].tolist()
                if segments[-1] == segments[-2]:
                    segments.pop()
                conversation = '. '.join(segments)


                cluster_file_segments[cluster][file] = {'starttime':starttime,'endtime':endtime,'conversation':conversation}

        print('='*60)
        print('Transcript Segments by Cluster')
        for cluster in cluster_file_segments:
            files = ': '.join(cluster.keys())
            print("The following files were all discussed cluster{0}:\n {1}".format(cluster,files))
            for file in cluster:
                print("File {0} discussed it from {1} to {2}.".format(file,file['starttime'],file['endtime']))
                print(file['conversation'])
                print('\n')


    elif multiple_files == False:
        for cluster in clusters:
            start = dataframe.where(dataframe['Clusters']==cluster).first_valid_index()
            end = dataframe.where(dataframe['Clusters']==cluster).last_valid_index()

            cluster_segments = file_dataframe.iloc[start:end]
            starttime = cluster_segments.iloc['TimeStamp'][0]
            endtime = cluster_segments['TimeStamp'].iat[-1]
            
            segments = cluster_segments['Segments'].tolist()
            if segments[-1] == segments[-2]:
                segments.pop()
            conversation = '. '.join(segments)

            cluster_file_segments[cluster] = {'starttime':starttime,'endtime':endtime,'conversation':conversation}
        
        print('='*60)
        print('Transcript Segments by Cluster')
        for cluster in cluster_file_segments:
            print("Speakers discussed cluster{0} from {1} to {2}.".format(cluster,cluster['starttime'],cluster['endtime']))
            print(cluster['conversation'])
            print('\n')

    else:
        raise('The variable multiple_files must be True or False')

def preprocess_transcript(file, regex_list=[], phrase_list=[], word_list=[], diarize = False, divide_by = 'sentence', min_char=6, min_words=2):

    replacements = {}

    with open(file, 'r') as f:
        transcript_file = f.read()

        try:
            transcript = Transcript(transcript_file)

            replacements = transcript.clean_transcript(regex_list, phrase_list, word_list)

            if diarize == False:
                transcript.remove_diarization()
            else:
                transcript._cleanTranscript = transcript._transcript
            
            transcript.divide_by_speaker()

            if divide_by == 'sentence':
                transcript.divide_by_sentence(min_char, min_words)

            return transcript._transcript_df, replacements

        except TypeError:
            print("*"*60)
            print("Unable to process {0} as a TypeError was found. It may be due to the lack of a startTime variable.".format(file))
            print("*"*60)

        except:
            print("*"*60)
            print("Unable to process {0}. Ensure it is correctly formatted json file.".format(file))
            print("*"*60)

def batch_processing(file_list, file_directory, working_directory, duplicate_log, iteration, regex_list=[], phrase_list=[], word_list=[], diarize = False, divide_by = 'sentence', min_char=6, min_words=2):
    
    transcript_files_df = pd.DataFrame()
    
    print('='*60)
    print('Processing {0:,} files. Your patience is appreciated.'.format(len(file_list)))
    print('='*60)
    start_time = time.time()
    
    replacement_list = phrase_list + word_list + regex_list
    all_replacements = {}

    for item in replacement_list:
        all_replacements[item] = 0

    for filename in file_list:
        
        try:
            transcript_file = os.path.join(file_directory, filename)
                
            file_df, replacements = preprocess_transcript(transcript_file, regex_list, phrase_list, word_list, diarize, divide_by, min_char, min_words)
            # preprocess_transcript(file, regex_list=[], phrase_list=[], word_list=[], diarize = False, divide_by = 'sentence', min_char=6, min_words=2)

            if transcript_files_df.empty:
                transcript_files_df = file_df
        
            else:
                transcript_files_df = pd.concat([transcript_files_df, file_df], axis=0, ignore_index=True)

            if replacement_list:
                for word in replacement_list:
                    all_replacements[word] = all_replacements[word] + replacements[word]

        except:
            print("Nothing added to the dataframe for {0}".format(filename))
            
    print('='*50)
    print("The following number of replacements were made:")        
    if replacement_list:
        for word in replacement_list:
            print("{0} was replaced {1:,} times.".format(word,all_replacements[word]))

    print("There were a total of {0:,} replacements.".format(sum(all_replacements.values())))

    transcript_files_df.dropna(subset = ['CleanSeg'], inplace = True)

    #find the duplicate sentences and the audio files with the duplicates
    duplicates_df = pd.concat(g for _, g in transcript_files_df.groupby("CleanSeg") if len(g) > 1)

    # * below can be used to rebuild dataframe with full sentences if desired.
    files_with_duplicates = {} 
    for row, file in enumerate(duplicates_df['File']):
        if file not in files_with_duplicates:
            files_with_duplicates[file] = []
        files_with_duplicates[file].append(duplicates_df.iloc[row]['CleanSeg'])
    
    dedup_df = duplicates_df.drop_duplicates(subset = ['CleanSeg'])
    duplicate_sentences = dedup_df['CleanSeg'].tolist()

    # dedup the 'CleanSeg' column to reduce the bandwidth on embeddings processing
    transcript_files_df.drop_duplicates(subset = ['CleanSeg'], inplace = True) 
    minutes_processing, seconds_processing = divmod((time.time() - start_time), 60)

    duplicate_file = working_directory + '/' + duplicate_log + str(iteration) + '.txt'

    with open(duplicate_file, 'a') as dup_file:
        dup_file.write('Duplicated Sentences by Audio File\n\n')
        for audio_file, sentences in files_with_duplicates.items():
            dup_file.write('Audio File: ' + audio_file + '\n')
            dup_file.write('; '.join(sentences))
            dup_file.write('\n\n')

    print('='*50)
    print("Processing done after {0:.0f} mins and {1:.2f} secs".format(minutes_processing, seconds_processing))
    print("There are {0:,} rows in the final dataframe.".format(transcript_files_df.shape[0]))
    print("There were {0:,} rows that were dropped for having duplicate values.".format(duplicates_df.shape[0]))
    print("Full record of the deplication can be found in the {0} file".format(duplicate_file))
    print('='*50)

    # transcript_files_df['Clusters'], transcript_files_df['Probabilities'] = sa.cluster_divisions(transcript_files_df['Embeddings'].tolist())

    return transcript_files_df

def create_csv(dataframe, working_directory, output_file):

    print("\n")
    print('='*50)
    print("Storing the dataframe to csv.")
    start_time = time.time()
    
    csv_file = os.path.join(working_directory, output_file)

    try:
        dataframe.to_csv(csv_file, sep='\t', encoding='utf-8', index=False)
    except OSError:
        print("*"*60)
        print("OSError raised when attempting to save file in csv. Check that the directory exists.")
        print("*"*60)
    except:
        print("*"*60)
        print("Unable to save file in csv.")
        print("*"*60)

    minutes_processing, seconds_processing = divmod((time.time() - start_time), 60)
    print("Storing attempt complete after {0:.0f} mins and {1:.2f} secs".format(minutes_processing, seconds_processing))
    print('='*50)     

def update_embeddings_csv(file_list, working_directory, output_file, input_file = '', phrase_list=[], word_list=[], regex_list=[], diarize = False, divide_by = 'sentence', min_char=6, min_words=2):

    if input_file == '':
        raise Exception("You must provide a file to use in the update.")
    
    transcript_files_df = pd.DataFrame()

    try:
        with open(input_file, 'r') as update_file:
            transcript_files_df = pd.read_csv(update_file,sep='\t',index_col=False,converters={'Embeddings': literal_eval})

            if divide_by=='sentence' and 'SentenceID' not in transcript_files_df.columns:
                print('='*50)
                raise Exception("You're current data is not divided by sentence. Operation cancelled.")
    
    except FileNotFoundError:
        print("*"*100)
        print("Update file does not exist. Please check the file name and path")
        print("*"*100)
        exit

    except:
        print("*"*100)
        print("There was another error.")
        print("*"*100)
        exit

        print('='*60)
    print('Processing {0} files. Your patience is appreciated.'.format(len(file_list)))
    print('='*60)
    start_time = time.time()
    
    update_df = pd.DataFrame()

    for filename in file_list:
        
        try:
            transcript_file = os.path.join(working_directory, filename)
                
            file_df = preprocess_transcript(transcript_file, diarize, divide_by, min_char, min_words)

            if update_df.empty:
                update_df = file_df
            
            else:
                update_df = pd.concat([update_df, file_df], axis=0, ignore_index=True)
        
        except:
            # Passing because errors should be in the preprocess_transcript method
            pass

    minutes_processing, seconds_processing = divmod((time.time() - start_time), 60)
    print('='*50)
    print("Processing done after {0:.0f} mins and {1:.2f} secs".format(minutes_processing, seconds_processing))
    print("There are {0} rows in the final dataframe.".format(update_df.shape[0]))  
    print('='*50)

    update_df['Embeddings'] = sa.embed_divisions(update_df['CleanSeg'].tolist())
    # transcript_files_df['Clusters'], transcript_files_df['Probabilities'] = sa.cluster_divisions(transcript_files_df['Embeddings'].tolist())
    print('='*50)

    print('Concatenating both dataframes.'.format(len(file_list)))
    print('='*60)
    start_time = time.time()    
    
    transcript_files_df = pd.concat([transcript_files_df, update_df], axis=0, ignore_index=True)

    minutes_processing, seconds_processing = divmod((time.time() - start_time), 60)
    print('='*50)
    print("Processing done after {0:.0f} mins and {1:.2f} secs".format(minutes_processing, seconds_processing))
    print('='*50)

    print("\n")
    print('='*50)
    print("Storing the dataframe to csv.")
    start_time = time.time()
    csv_file = output_file + '.csv'
    try:
        transcript_files_df.to_csv(csv_file, sep='\t', encoding='utf-8', index=False)
    except:
        print("*"*60)
        print("Unable to save file in csv")
        print("*"*60)
    minutes_processing, seconds_processing = divmod((time.time() - start_time), 60)
    print("Storing attempt complete after {0:.0f} mins and {1:.2f} secs".format(minutes_processing, seconds_processing))
    print('='*50)

    print("\n\n")
    print('='*50)
    print('PROCESSING FULLY COMPLETED!!')
    print('='*50)