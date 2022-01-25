import rosbag
import argparse
import os
import json
import pandas as pd

SAMPLE_RATE = 16000

parser = argparse.ArgumentParser(description='Process gaze from a a bagfile.')
parser.add_argument('--bagfile', type=str, required=True, help='Bag filename')
parser.add_argument("-d", "--distancetype", default='cosine_distance', 
    help='type of distance to use (euclidien_distance, cosine_distance, euclidien_distance, cosine_distance)')
args = parser.parse_args()

scene_transform_topic = "/scene/transform"
audio_topic = "/rawspeech"
button_topic = "/buttons"
object_topic = '/object_clusters'

out_file = os.path.splitext(os.path.basename(args.bagfile))[0].split('_')[0]
if not os.path.isdir( os.path.join('gaze_data', out_file, args.distancetype) ):
    #print(os.path.join('gaze_data', out_file, args.distancetype))
    os.makedirs(os.path.join('gaze_data', out_file, args.distancetype))

audio_file = os.path.join('gaze_data', out_file, args.distancetype,'audio.csv')
#print(audio_file)
button_file = os.path.join('gaze_data', out_file, args.distancetype,'buttons.csv')
#print(button_file)

bag = rosbag.Bag(args.bagfile)

button_timestamps = []
for topic, msg, t in bag.read_messages(topics=[button_topic]):
    #print(t.to_sec(), msg.data)
    button_timestamps.append( (t.to_sec(), msg.data) )

audio_timestamps = []
for topic, msg, t in bag.read_messages(topics=[audio_topic]):
    data = json.loads(msg.data)
    audio_length = float(len(data))/float(SAMPLE_RATE)
    gen = bag.read_messages(start_time=t, topics=['/speech'])
    text = ''
    try:
        text_topic, text_msg, text_t = gen.next()
        text = text_msg.data
    except StopIteration:
        print('The transcription generator was empty')
    if text != '':
    	#print(t.to_sec(), t.to_sec()-audio_length, text)
        audio_timestamps.append( (t.to_sec(), t.to_sec()-audio_length, text) )


if not os.path.exists(audio_file):
    print('writing to'+audio_file)
    audio_csv = pd.DataFrame(audio_timestamps, columns=['end_timestamp', 'start_timestamp', 'transcript'])
    audio_csv.to_csv(audio_file, index=False)

if not os.path.exists(button_file):
    print('writing to'+button_file)
    button_csv = pd.DataFrame(button_timestamps, columns=['timestamp', 'message'])
    button_csv.to_csv(button_file, index=False)
