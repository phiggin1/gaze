import json
import pandas as pd
import rosbag
import math
import numpy as np
from scipy.spatial.distance import cosine as cos_distance
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import argparse
import os
import tf
from tf_bag import BagTfTransformer
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

SAMPLE_RATE = 16000

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
     
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
     
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
     
    return roll_x, pitch_y, yaw_z # in radians

def distance(a, b):
    x = a[0]-b[0]
    y = a[1]-b[1]
    z = a[2]-b[2]

    return math.sqrt(x*x + y*y + z*z)

def distance_point_to_ray(pA, pB, p):
    d = np.linalg.norm( np.cross( p-pA, p-pB ) ) / np.linalg.norm( pB-pA)
    return d

def PositionUnity2Ros(vector3):
    #vector3.z, -vector3.x, vector3.y);
    return [vector3[2], -vector3[0], vector3[1]]


def QuaternionUnity2Ros(quaternion):
    #return new Quaternion(-quaternion.z, quaternion.x, -quaternion.y, quaternion.w);
    return [ -quaternion[2], quaternion[0], -quaternion[1], quaternion[3]]

def xyz_to_mat44(pos):
    return tf.transformations.translation_matrix((pos.x, pos.y, pos.z))

def xyzw_to_mat44(ori):
    return tf.transformations.quaternion_matrix((ori.x, ori.y, ori.z, ori.w))

def get_head_pose(head, translation, rotation):
    head_pos = [head['position']['x'], 
                head['position']['y'], 
                head['position']['z']]

    head_rot_quat = [head['rotation']['x'], 
                        head['rotation']['y'], 
                        head['rotation']['z'], 
                        head['rotation']['w']]
    
    ros_head_pos = PositionUnity2Ros(head_pos)
    ros_head_rot_quat = QuaternionUnity2Ros(head_rot_quat)

    ps = PoseStamped()
    ps.header.frame_id = odom_frame
    ps.header.stamp = t

    ps.pose.position.x = ros_head_pos[0]
    ps.pose.position.y = ros_head_pos[1]
    ps.pose.position.z = ros_head_pos[2]

    ps.pose.orientation.x = ros_head_rot_quat[0]
    ps.pose.orientation.y = ros_head_rot_quat[1]
    ps.pose.orientation.z = ros_head_rot_quat[2]
    ps.pose.orientation.w = ros_head_rot_quat[3]

    mat44 = np.dot(tf.transformations.translation_matrix(translation), tf.transformations.quaternion_matrix(rotation))
    pose44 = np.dot(xyz_to_mat44(ps.pose.position), xyzw_to_mat44(ps.pose.orientation))

    # txpose is the new pose in target_frame as a 4x4
    txpose = np.dot(mat44, pose44)

    # xyz and quat are txpose's position and orientation
    xyz = tuple(tf.transformations.translation_from_matrix(txpose))[:3]
    quat = tuple(tf.transformations.quaternion_from_matrix(txpose))

    head_in_kinect_frame = (Point(*xyz))
    gaze_in_kinect_frame = R.from_quat(quat).apply([0.0, 0.0, 1.0])

    return head_in_kinect_frame, gaze_in_kinect_frame

def get_closeest_object(position, object_positions):
    min_dist = 10000.0
    name = 'other'
    print(position)
    for obj_name in object_positions.keys():
        #d = distance(position, object_positions[obj_name])
        d =  math.sqrt( (position[0]-object_positions[obj_name][0])**2 + (position[2]-object_positions[obj_name][2])**2 )
        print(obj_name, object_positions[obj_name], d)
        if d < min_dist:
            name = obj_name
            min_dist = d
    print(name, min_dist)
    return name

scene_transform_topic = "/scene/transform"
audio_topic = "/rawspeech"
button_topic = "/buttons"
object_topic = '/object_clusters'

pointcloud_frame = '/kinect2_link'
odom_frame = '/odom'

distance_type=['euclidien_distance',
               'cosine_distance']

parser = argparse.ArgumentParser(description='Process gaze from a a bagfile.')
parser.add_argument('--bagfile', type=str, required=True, help='Bag filename')
parser.add_argument("-d", "--distancetype", default='cosine_distance', 
    help='type of distance to use (euclidien_distance, cosine_distance, euclidien_distance, cosine_distance)')
args = parser.parse_args()

print(args.bagfile)
print(args.distancetype)

if not args.distancetype in distance_type:
    print('Distance type needs to be (euclidien_distance_3d, cosine_distance_3d, euclidien_distance_2d, cosine_distance_2d)')
    exit()

out_file = os.path.splitext(os.path.basename(args.bagfile))[0].split('_')[0]
print(out_file)
if not os.path.isdir( os.path.join('gaze', out_file, args.distancetype) ):
    print(os.path.join('gaze', out_file, args.distancetype))
    os.makedirs(os.path.join('gaze', out_file, args.distancetype))

distances_file = os.path.join('gaze', out_file, args.distancetype,'distances.csv')
print(distances_file)
objects_file = os.path.join('gaze', out_file, args.distancetype,'objects.csv')
print(objects_file)
audio_file = os.path.join('gaze', out_file, args.distancetype,'audio.csv')
print(audio_file)
button_file = os.path.join('gaze', out_file, args.distancetype,'buttons.csv')
print(button_file)

bag = rosbag.Bag(args.bagfile)

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
        audio_timestamps.append( (t.to_sec(), t.to_sec()-audio_length, text) )

head_index = 0
closest_objects = []
distances = []
object_positions = {}

has_head_pose = False

print('pre tf')
bag_transformer = BagTfTransformer(args.bagfile)
print('post tf')

for topic, msg, t in bag.read_messages(topics=[scene_transform_topic, object_topic]):
    if topic == scene_transform_topic:
        transforms = json.loads(msg.data)
        translation, rotation = bag_transformer.lookupTransform(pointcloud_frame, odom_frame, t)
        head_in_kinect_frame, gaze_in_kinect_frame = get_head_pose(transforms[head_index], translation, rotation)

        mat44 = np.dot(tf.transformations.translation_matrix(translation), tf.transformations.quaternion_matrix(rotation))
        #print(translation)
        for i, transform  in enumerate(transforms[6:]):
            name = str(transform['name'])
            if name in table_objects:

                ros_pos = PositionUnity2Ros([transform['position']['x'],transform['position']['y'],transform['position']['z']])

                p = Point()
                p.x = ros_pos[0]
                p.y = ros_pos[1]
                p.z = ros_pos[2]
                q = Quaternion()
                q.x = 0.0
                q.y = 0.0
                q.z = 0.0
                q.w = 1.0
                pose44 = np.dot(xyz_to_mat44(p), xyzw_to_mat44(q))
                #print(pose44)
                # txpose is the new pose in target_frame as a 4x4
                txpose = np.dot(mat44, pose44)
                object_positions[name] = tuple(tf.transformations.translation_from_matrix(txpose))[:3]
                #print('--------')
                #print(translation)
                #print(p.x,p.y,p.z)
                #print(object_positions[name])
                
                object_positions[name] =[ transform['position']['x'], transform['position']['y'], transform['position']['z'] ]

        has_head_pose = True

    elif topic == object_topic and has_head_pose:
        #if len(msg.clusters) > 10:
        print('-----------------')
        print( len(msg.clusters) )
        head_pos = np.asarray( [head_in_kinect_frame.x,
                                head_in_kinect_frame.y,
                                head_in_kinect_frame.z])
        gaze = gaze_in_kinect_frame

        min_dist = 1000000.0
        min_index = 0

        #print('================')
        #print(head_pos)
        #print(gaze)
        dist = {'timestamp':t.to_sec(), 'head_pos':head_pos, 'gaze':gaze}

        for i, obj in enumerate(msg.clusters):
            #print(i, len(obj.data), obj.header.frame_id)

            for p in point_cloud2.read_points(obj):
                position = np.asarray( p[0:3] )
                name = get_closeest_object(position, object_positions)
                
                if args.distancetype == 'euclidien_distance':
                    euclidien_distance_3d = distance_point_to_ray(np.asfarray(head_pos), np.asfarray(head_pos + gaze), position)
                    d = euclidien_distance_3d
                elif args.distancetype == 'cosine_distance':
                    cosine_distance_3d = cos_distance(gaze, position-head_pos)
                    d = cosine_distance_3d

                #print(i, name, d)
                dist[name]=d
                if d < min_dist:
                    min_dist = d
                    min_name = name
                    min_index = i
                break

        print(t.to_sec(), min_name, min_index, min_dist)
          
        if len(closest_objects) > 0:
            if closest_objects[-1][1] != min_index:
                closest_objects.append( (t.to_sec(), min_index, min_dist) )
        else:
            closest_objects.append( (t.to_sec(), min_index, min_dist) )

        distances.append(dist)
        
        
    
bag.close()

distances_csv = pd.DataFrame(distances, columns=['timestamp', 'head_pos', 'head_quat'].extend(table_objects))
distances_csv.to_csv(distances_file, index=False)

objects_csv = pd.DataFrame(closest_objects, columns=['timestamp', 'object', 'distance'])
objects_csv.to_csv(objects_file, index=False)

audio_csv = pd.DataFrame(audio_timestamps, columns=['end_timestamp', 'start_timestamp', 'transcript'])
audio_csv.to_csv(audio_file, index=False)

button_csv = pd.DataFrame(button_timestamps, columns=['timestamp', 'message'])
button_csv.to_csv(button_file, index=False)