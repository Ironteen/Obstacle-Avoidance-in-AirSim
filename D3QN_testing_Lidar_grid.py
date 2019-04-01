import tensorflow as tf
import setup_path
import airsim
from collections import deque

import random
import numpy as np
import time
import os
import pickle
# basic setting
ACTION_NUMS = 13        # number of valid actions
MAX_EPISODE = 20000
DEPTH_IMAGE_WIDTH = 256
DEPTH_IMAGE_HEIGHT = 144

flatten_len = 9216      # the input shape before full connect layer
NumBufferFrames = 4     # take the latest 4 frames as input

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name="weights")

def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name="bias")

def conv2d(x, W, stride_h, stride_w):
    return tf.nn.conv2d(x, W, strides=[1, stride_h, stride_w, 1], padding="SAME")

class Deep_Q_Network(object):
    """docstring for ClassName"""
    def __init__(self, sess):
        # network weights and biases
        # input 144x256x4
        with tf.name_scope("Conv1"):
            W_conv1 = weight_variable([8, 8, NumBufferFrames, 32])
            variable_summaries(W_conv1)
            b_conv1 = bias_variable([32])
        with tf.name_scope("Conv2"):
            W_conv2 = weight_variable([4, 4, 32, 64])
            variable_summaries(W_conv2)
            b_conv2 = bias_variable([64])
        with tf.name_scope("Conv3"):
            W_conv3 = weight_variable([3, 3, 64, 64])
            variable_summaries(W_conv3)
            b_conv3 = bias_variable([64])
        with tf.name_scope("Value_Dense"):
            W_value = weight_variable([flatten_len, 512])
            variable_summaries(W_value)
            b_value = bias_variable([512])
        with tf.name_scope("FCAdv"):
            W_adv = weight_variable([flatten_len, 512])
            variable_summaries(W_adv)
            b_adv = bias_variable([512])
        with tf.name_scope("FCValueOut"):
            W_value_out = weight_variable([512, 1])
            variable_summaries(W_value_out)
            b_value_out = bias_variable([1])
        with tf.name_scope("FCAdvOut"):
            W_adv_out = weight_variable([512, ACTION_NUMS])
            variable_summaries(W_adv_out)
            b_adv_out = bias_variable([ACTION_NUMS])
        # input layer
        self.state = tf.placeholder("float", [None, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, NumBufferFrames])
        # Conv1 layer
        h_conv1 = tf.nn.relu(conv2d(self.state, W_conv1, 8, 8) + b_conv1)
        # Conv2 layer
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2, 2) + b_conv2)
        # Conv2 layer
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1, 1) + b_conv3)
        h_conv3_flat = tf.layers.flatten(h_conv3)
        # FC ob value layer
        h_fc_value = tf.nn.relu(tf.matmul(h_conv3_flat, W_value) + b_value)
        value = tf.matmul(h_fc_value, W_value_out) + b_value_out
        # FC ob adv layer
        h_fc_adv = tf.nn.relu(tf.matmul(h_conv3_flat, W_adv) + b_adv)
        advantage = tf.matmul(h_fc_adv, W_adv_out) + b_adv_out
        # Q = value + (adv - advAvg)
        advAvg = tf.expand_dims(tf.reduce_mean(advantage, axis=1), axis=1)
        advIdentifiable = tf.subtract(advantage, advAvg)
        self.readout = tf.add(value, advIdentifiable)
        # define the cost function
        self.actions = tf.placeholder("float", [None, ACTION_NUMS])
        self.y = tf.placeholder("float", [None])
        self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.actions), axis=1)
        self.td_error = tf.square(self.y - self.readout_action)
        self.cost = tf.reduce_mean(self.td_error)
        self.train_step = tf.train.AdamOptimizer(1e-5).minimize(self.cost)

def get_image(client,image_type):
    if (image_type == 'Scene'):
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgba = img1d.reshape(response.height, response.width, 4)
        img_rgba = np.flipud(img_rgba)
        observation = img_rgba[:, :, 0:3]
    elif (image_type == 'Segmentation'):
        responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgba = img1d.reshape(response.height, response.width, 4)  # reshape array to 4 channel image array H X W X 4
        observation = img_rgba[:, :, 0:3]
    elif (image_type == 'DepthPlanner'):
        try:
            responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPlanner, pixels_as_float=True)])
            response = responses[0]
            img1d = np.array(response.image_data_float, dtype=np.float)
            img1d = img1d * 3.5 + 30
            img1d[img1d > 255] = 255
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
            observation = img2d
        except:
            print('######### Error: I can not get a depth image correctly! #########################' )
            observation = np.ones([DEPTH_IMAGE_HEIGHT,DEPTH_IMAGE_WIDTH])
    else:
        observation = None
    observation_size = np.shape(observation)
    if(observation_size[0]==DEPTH_IMAGE_HEIGHT and observation_size[1]==DEPTH_IMAGE_WIDTH):
        return observation
    else:
        print('######### Error: The depth image shape: ',observation_size)
        return np.ones([DEPTH_IMAGE_HEIGHT,DEPTH_IMAGE_WIDTH])

def env_feedback(client,map_grid,latest_distance):
    terminal_position = [-130, -210]
    collision_info = client.simGetCollisionInfo()
    car_position = client.getCarState().kinematics_estimated.position
    distance = np.sqrt((car_position.x_val-terminal_position[0])**2+(car_position.y_val-terminal_position[1])**2)
    if collision_info.has_collided:
        reset = True
    else:
        x_val, y_val = car_position.x_val, car_position.y_val
        x_val, y_val = int(np.abs(x_val * 2)), int(np.abs(y_val * 2))
        map_grid[x_val, y_val] = 1 / (distance + 0.001)
        reset = False
    if (distance <= 10):
        terminal = 1
        reset = True
    else:
        terminal = 0
    if distance<latest_distance:
        latest_distance = distance
    return terminal,reset,distance,map_grid,latest_distance

def excute_action(client,car_controls,steer):
    car_controls.steering = steer
    car_speed = client.getCarState().speed
    straight_range = [-0.1,0.1]
    # when swerved wildly, slow down
    BigTurn_range = [-0.6,0.6]
    if(steer>=straight_range[0] and steer<=straight_range[1]):
        car_controls.throttle = car_controls.throttle + 0.1
        car_controls.throttle = 2 if car_controls.throttle>=2 else car_controls.throttle
    elif(steer<=BigTurn_range[0] or steer>=BigTurn_range[1]):
        car_controls.throttle = 0.5
    else:
        car_controls.throttle = 1
    car_controls.throttle = 0 if car_speed>=3 else car_controls.throttle
    return car_controls

def print_action(episode,t,car_controls,distance,steer):
    print('Episode:%05d,Step:%05d '%(episode,t),end='')
    if(steer<0):
        print('The car is turning left    , ',end='')
    elif(steer>0):
        print('The car is turning right   , ',end='')
    else:
        print('The car is going straightly, ',end='')
    print('throttle=%.1f,  steer=%.3f, distance=%.2f'%(car_controls.throttle,steer,distance),end='')

def testNetwork():
    client = airsim.CarClient()
    client.confirmConnection()
    print('Connect succcefully！')
    client.enableApiControl(True)
    car_controls = airsim.CarControls()
    car_controls.throttle = 0.5
    car_controls.steering = 0
    client.reset()
    print('Environment initialized!')
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.InteractiveSession()
    with tf.name_scope("TargetNetwork"):
        Q_net = Deep_Q_Network(sess)
    time.sleep(1)

    reward_var = tf.Variable(0., trainable=False)
    tf.summary.scalar('reward', reward_var)
    # define summary
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    # get the first state
    observe_init = get_image(client,'DepthPlanner')
    state_pre = np.stack((observe_init, observe_init, observe_init, observe_init), axis=2)
    # saving and loading networks
    trainables = tf.trainable_variables()
    trainable_saver = tf.train.Saver(trainables)
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks/new_model_lidar/")
    print('checkpoint:', checkpoint)
    if checkpoint and checkpoint.model_checkpoint_path:
        trainable_saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        if not os.path.exists("saved_networks/new_model_lidar"):
            os.mkdir("saved_networks/new_model_lidar")
            print('The file not exists, is created successfully')
        print("Could not find old network weights")
    # start training
    episode = 1
    print('Number of trainable variables:', len(trainables))
    inner_loop_time_start = time.time()
    terminal_record = 0
    collision_record = 0
    steps_record = []
    map_grid = -1 * np.ones([500, 500])
    while episode < MAX_EPISODE:
        step = 1
        reset = False
        latest_distance = 300
        while not reset:
            # take the latest 4 frames as an input
            observe = get_image(client,'DepthPlanner')
            observe = np.reshape(observe, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1))
            state_current = np.append(observe, state_pre[:, :, :(NumBufferFrames - 1)], axis=2)
            terminal,reset,distance,map_grid,latest_distance= env_feedback(client,map_grid,latest_distance)
            # store the experience
            state_pre = state_current
            # choose an action epsilon greedily
            actions = sess.run(Q_net.readout, feed_dict={Q_net.state: [state_current]})
            readout_t = actions[0]
            action_current = np.zeros([ACTION_NUMS])
            # fill the reply experience
            action_index = np.argmax(readout_t)
            action_current[action_index] = 1
            # Control the agent
            side_num = int(ACTION_NUMS - 1) // 2
            steer = float((action_index - side_num) / side_num)
            inner_loop_time_end = time.time()
            car_controls = excute_action(client,car_controls,steer)
            client.setCarControls(car_controls)
            print_action(episode, step, car_controls, distance, steer)
            print(',inner loop=%.4fs\n'%(inner_loop_time_end-inner_loop_time_start),end='')
            inner_loop_time_start = time.time()
            time.sleep(0.5)
            step = step+1
        steps_record.append(step-1)
        map_path = './test_result/grid_map_%d.pkl' % episode
        if not os.path.exists('./test_result'):
            os.mkdir('./test_result')
        map_file = open(map_path, 'wb')
        pickle.dump(map_grid, map_file)
        map_file.close()
        episode = episode + 1
        if(terminal==1):
            terminal_record +=1
        else:
            collision_record +=1
        print('terminal nums=%d,collision num=%d,average steps is %04d,the shortest distance=%.2f'%(terminal_record,collision_record,int(np.mean(steps_record)),latest_distance))
        client.reset()

def main():
    testNetwork()

if __name__ == "__main__":
    main()
