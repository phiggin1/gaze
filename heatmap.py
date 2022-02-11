import altair as alt
import pandas as pd
import argparse
import os
import numpy as np
from operator import itemgetter
from math import floor

table_objects = [
"/Objects/Bowl_1", 
"/Objects/Tomato_7d6fd278", 
"/Objects/waterbottle", 
"/Objects/Apple_34d5f204", 
"/Objects/Lettuce_b97186e2", 
"/Objects/Bread_a13c4e42", 
"/Objects/firstaid", 
"/Objects/Mug_77db6e4d", 
"/Objects/drill_textured/default", 
"/Objects/hammer_textured/default", 
]

parser = argparse.ArgumentParser(description='Generate gantt plot for gaze from files generated by headPgaze.py.')
parser.add_argument('--basefilename', type=str, required=True, help='Filename of the bagfile')

parser.add_argument("-d", "--distancetype", default='cosine_distance', 
    help='type of distance to use (euclidien_distance, cosine_distance)')
args = parser.parse_args()
#print(args.basefilename)
#print(args.distancetype)

out_file = os.path.splitext(os.path.basename(args.basefilename))[0].split('_')[0]
#print(out_file)


objects_file = os.path.join('gaze_data', out_file, args.distancetype,'objects.csv')
#print(objects_file)
audio_file = os.path.join('gaze_data', out_file, args.distancetype,'audio.csv')
#print(audio_file)
distances_file = os.path.join('gaze_data', out_file, args.distancetype,'distances.csv')
#print(distances_file)
#


df_audio = []
audio_csv = pd.read_csv(audio_file)
audio_list = []
for index, frame in audio_csv.iterrows():
    audio_list.append(frame)
for i in range(len(audio_list)):
    transcript = audio_list[i]['transcript']+"-Ground Truth"
    start = audio_list[i]['start_timestamp']
    end =  audio_list[i]['end_timestamp']
    df_audio.append(
        {"Objects":transcript ,"start":start, "end":end}
    )

'''
alt.X('start',
        scale=alt.Scale(zero=False)
    ),
'''

source_obj = pd.DataFrame(df_audio)
chart_obj = alt.Chart(source_obj).mark_bar().encode(
    x = 'start',
    x2='end',
    y = alt.Y('Objects', sort=['Bread', "Bread-Ground Truth",'Lettuce',"Lettuce-Ground Truth",'Apple',"Apple-Ground Truth",'Tomato',"Tomato-Ground Truth",'Mug',"Mug-Ground Truth",'Bowl',"Bowl-Ground Truth",'watterbottle',"watterbottle-Ground Truth",'firstaid',"firstaid-Ground Truth",'drill',"drill-Ground Truth",'hammer',"hammer-Ground Truth"], title=""),
    color=alt.Color('Objects', scale=alt.Scale(scheme='dark2'))
).properties(
    width=1200,
    height=300
)


# #heat map graph
df_obj = []
dist_csv = pd.read_csv(distances_file)
dist_list = []
for index, frame in dist_csv.iterrows():
    dist_list.append(frame)

start_time = audio_list[0]['start_timestamp']
end_time = audio_list[-1]['end_timestamp']


distances_list = []
for frame in dist_list:
    if start_time <= frame['timestamp'] <= end_time:
        dist =[]
        for obj in table_objects:
            if not np.isnan(frame[obj]): 
                dist.append( (frame[obj], obj[9:].split('_')[0]) )

        dist = sorted(dist,key=itemgetter(0))
        i = 1

        for o in dist:
            '''
            if len(distances_list)>1:
                print(distances_list[-1])
                print('obj=',o[1],' dist=',o[0])
                ind = [y[1] for y in distances_list[-1]].index(o[1])
                print('index=',ind)
                print('prev dist=',distances_list[-1][ind][0])
                #print([y[1] for y in distances_list][-1].index(o[1]))
            '''
            if o[0] != 2.0:
                d = {'timestamp':frame['timestamp']}
                d['object'] = o[1]
                d['distance'] = i
                df_obj.append(d)
                i += 1
        distances_list.append(dist)


#print(list(range(floor(start_time), floor(end_time), 5)))  // , values=list(range(floor(start_time), floor(end_time), 5))
source_obj = pd.DataFrame(df_obj)
heat = alt.Chart(source_obj).mark_rect().encode(
    x = alt.X('timestamp:O',  scale=alt.Scale(zero=False), axis=alt.Axis(title='Time')),
    y = alt.Y('object:O', sort=['Bread', "Bread-Ground Truth",'Lettuce',"Lettuce-Ground Truth",'Apple',"Apple-Ground Truth",'Tomato',"Tomato-Ground Truth",'Mug',"Mug-Ground Truth",'Bowl',"Bowl-Ground Truth",'watterbottle',"watterbottle-Ground Truth",'firstaid',"firstaid-Ground Truth",'drill',"drill-Ground Truth",'hammer',"hammer-Ground Truth"], title=""),
    color = alt.Color('distance:O', scale=alt.Scale(scheme='plasma'), legend=None)
).properties(
    width=3000,
    height=300
)


heat.save(os.path.join('gaze_data', out_file, args.distancetype,out_file+'_heat.html'))

#(heat+chart_obj).save('gaze_data/'+out_file+'test.html')