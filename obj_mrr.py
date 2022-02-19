import pandas as pd
import argparse
import os
import numpy as np
import statistics

TOP_N_OBJS = 3
MRR_SUM = 0
MRR_COUNT = 1
MRR_RANKS = 2



def print_all_mrr(mrr):
   """Calculate and print the MRR scores for all objects, as well as the mean MRR.
   Also Calculates the mean scores for all n scores

    Keyword arguments:
    mrr -- dictionary containing object names as keys and the sum, count, and rank records and fields
    """
   count = 0
   sum =  0
   mrr_list = [] 
   for key in mrr.keys():
      mrr_out = mrr[key][MRR_SUM]/mrr[key][MRR_COUNT]  
      print("MRR:",key, mrr_out)
      count +=1
      sum += mrr_out

      mrr_list.append(mrr_out)
   avg_mrr = sum / count
   print("Mean MRR:", avg_mrr)

   print("MRR Standard Deviation",statistics.pstdev(mrr_list), "\n\n")

def print_top_n_scores(mrr, n):
   """Calculate and print the top rank scores for all objects in the MRR dictionary.

    Keyword arguments:
    mrr -- dictionary containing object names as keys and the sum, count, and rank records and fields
    n -- number of top scores to calculate 
    """
   mean  = [0 for x in range(n)]

   for key in mrr.keys():
      # initialize counts
      top_count = 0
      top_n_occurances = [0 for x in range(n)]

      for rank in mrr[key][MRR_RANKS]:
         top_count += 1         
         for rank_ladder in range(rank-1, n):     
            top_n_occurances[rank_ladder] += 1
        
      for i in range(len(top_n_occurances)):
         top_score = top_n_occurances[i] / mrr[key][MRR_COUNT]
         print(key, " Rank =", i +1, " ", top_score)
         mean[i] += top_score
   
   for top_nth in range(len(mean)):
      print("Average top", str(top_nth + 1), " Mean =", mean[top_nth] / (len(mrr.keys()) ) )
   
if __name__ =="__main__":
   count_cosine = 0
   mrr = {}

   for root, dirs, files in os.walk("./gaze_data", topdown=False):
      for name in dirs:
         if "cosine_distance" in name  :
            count_cosine += 1
            currrent_dir = os.path.join(root, name)
            #print(currrent_dir)

            audio_csv = pd.read_csv(os.path.join(currrent_dir,'audio.csv'))
            #print(audio_csv)
            min_time = 9999999
            max_time = 0
            for audio in audio_csv.itertuples():
               if audio.start_timestamp <= min_time:
                  min_time = audio.start_timestamp
               if max_time <= audio.end_timestamp:
                  max_time = audio.end_timestamp
               if audio.transcript not in mrr.keys():
                  mrr[audio.transcript ] = [0,0, []]

            distances = pd.read_csv(os.path.join(currrent_dir,'distances.csv'))            

            for distance_row in distances.itertuples():
               try:
                  # makes sure timestamps outside of description times are ignored
                  if distance_row.timestamp >= min_time and distance_row.timestamp <= max_time:
                     break_audio_timesearch = False

                     for audio in audio_csv.itertuples():
                        # Object description time period is found
                        if audio.start_timestamp <= distance_row.timestamp <= audio.end_timestamp:
                           # MRR's Q+1
                           mrr[audio.transcript][MRR_COUNT] += 1  

                           # Break search of audio times since description time period is found
                           break_audio_timesearch = True          
                           
                           # exclude panda index and timestamp
                           distance_row_modded = list(distance_row)[1:len(distance_row)-1] 

                           # order by shortest distance first
                           distance_row_modded.sort()                                        

                           for i in range(TOP_N_OBJS):
                              index_of_nth_item = distance_row.index(distance_row_modded[i]) -1
                              
                              # Looks for an object match
                              if (audio.transcript in distances.columns[index_of_nth_item]):
                                 # MRR's sum of 1/rank
                                 mrr[audio.transcript][MRR_SUM] += (1/(i+1)) 
                                 # MRR's RANK Records 
                                 mrr[audio.transcript][MRR_RANKS].append(i+1) 

                        if (break_audio_timesearch):
                           break

               # to catch indexing errors in develoment
               except BaseException as err:
                  print(f"Unexpected {err=}, {type(err)=}")

   print_all_mrr(mrr)
   print_top_n_scores(mrr, TOP_N_OBJS)
