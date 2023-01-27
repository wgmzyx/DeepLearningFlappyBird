#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import cv2
# 没有cv2包，安装opencv-python即可
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
tf.compat.v1.disable_v2_behavior()

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions,行为上，下
GAMMA = 0.99 # decay rate of past observations，
OBSERVE = 1000. # timesteps to observe before training观察100000
EXPLORE = 2000000. # frames over which to anneal epsilon迭代次数
FINAL_EPSILON = 0.1 # final value of epsilon探索或者开发的概率，会发生变化，类似于学习率
INITIAL_EPSILON = 0.0001 # starting value of epsilon暂时设置一样
REPLAY_MEMORY = 50000 # number of previous transitions to remember观察到的东西存到这里
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1 #每次额外走一帧


def weight_variable(shape):

    # initial = tf.truncated_normal(shape, stddev = 0.01) #随机高斯初始化矩阵，指定标准差
    initial = tf.compat.v1.random.truncated_normal(shape, stddev = 0.01)
    
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape) #常量初始化
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights权重值初始化
    W_conv1 = weight_variable([8, 8, 4, 32])#权重值初始化
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])#将上面的图翻倍，得到64
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])#用前面1600个特征值得到512个结果
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    tf.compat.v1.disable_eager_execution()

    s = tf.compat.v1.placeholder("float", [None, 80, 80, 4]) #每四帧作为一个图像输入

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1) #第一个卷积操作
    h_pool1 = max_pool_2x2(h_conv1)  #第一个池化

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])  #全连接层，把特征图拉长，1600自己算

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)  #倒数第二个全连接

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2  #最终结果，先上或向下走的得分值

    return s, readout, h_fc1


def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function定义结构
    a = tf.compat.v1.placeholder("float", [None, ACTIONS]) #行为
    y = tf.compat.v1.placeholder("float", [None])#下一个状态的结果
    #当前阶段的输出，即当前状态预测值
    readout_action = tf.reduce_sum(tf.multiply(readout, a), axis=[1])
    #损失函数 
    cost = tf.reduce_mean(tf.square(y - readout_action))
    #train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    train_step =  tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator游戏环境
    game_state = game.GameState()

    # store the previous observations in replay memory，先观察游戏再保存结果
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)  #刚开始的图像，什么都没有做
    do_nothing[0] = 1 #定义第一步的步骤，怎么飞都可以
    x_t, r_0, terminal = game_state.frame_step(do_nothing)  #先跑一帧
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY) #对图像进行预处理，定义大小并且转化为灰度图
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)  #对图二值化变换，用ret占位
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) #起始图像将四个堆叠，作为一个状态8

    # saving and loading networks
    saver = tf.compat.v1.train.Saver() #保存模型初始化
    sess.run(tf.compat.v1.global_variables_initializer())  #用secc run全局变量初始化
    checkpoint = tf.train.get_checkpoint_state("saved_networks") #用于保存模型
    if checkpoint and checkpoint.model_checkpoint_path: #首先指定一个模型，如果模型之前训练国
        saver.restore(sess, checkpoint.model_checkpoint_path) #接着之前的继续训练
        print("Successfully loaded:", checkpoint.model_checkpoint_path) #成功加载之前模型
    else:
        print("Could not find old network weights") #要从头开始训练

    # start training
    epsilon = INITIAL_EPSILON #初始
    t = 0 #第一次迭代
    while "flappy bird" != "angry bird": #不可能相同，则无限迭代
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0] #最终输出值，需要指定一个输入，初始输入
        a_t = np.zeros([ACTIONS]) #
        action_index = 0
        if t % FRAME_PER_ACTION == 0: #每一步
            if random.random() <= epsilon:#随机值
                print("----------Random Action----------") #探索
                action_index = random.randrange(ACTIONS) #执行的方向随机选
                a_t[random.randrange(ACTIONS)] = 1 #实际向所选的方向走了
            else:
                action_index = np.argmax(readout_t) #找出分值大的行为
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t) #跑第一帧，并保存
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1)) #将新跑的一个图形处理
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2) #和原来三个图形链接起来

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal)) #存储状态
        if len(D) > REPLAY_MEMORY: #如果超出上限
            D.popleft() #删除之前的存储

        # only train if done observing
        if t > OBSERVE: #观测完了
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH) #训练数据，随机取32个状态

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch] #输入的状态
            a_batch = [d[1] for d in minibatch]  #当前执行的操作
            r_batch = [d[2] for d in minibatch] #当前的奖励
            s_j1_batch = [d[3] for d in minibatch] #下一个状态

            y_batch = [] #结果收益
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch}) #最终神经网络结果
            #对于结果写for循环
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4] #判断是否结束
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i]) #直接完成奖励
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i])) #算下一个状态的值，gamma折扣系数

            # perform gradient step执行优化
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values状态更新
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations模型保存，每1000次保存
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info指定当前状态
        state = ""
        if t <= OBSERVE:
            state = "observe"#观测
        elif t > OBSERVE and t <= OBSERVE + EXPLORE: 
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''


def playGame():

    # sess = tf.InteractiveSession() # 构造secc，保存模型
    sess = tf.compat.v1.InteractiveSession()
    s, readout, h_fc1 = createNetwork()  # 用于构造神经网络结构
    trainNetwork(s, readout, h_fc1, sess)


def main():

    playGame()


if __name__ == "__main__":

    main()
