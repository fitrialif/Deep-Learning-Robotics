import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
sns.set()

# Estimation parameter of EKF
Q = np.diag([0.1, 0.1, math.radians(1.0), 1.0])**2
R = np.diag([1.0, math.radians(40.0)])**2

#  Simulation parameter
Qsim = np.diag([0.5, 0.5])**2
Rsim = np.diag([1.0, math.radians(30.0)])**2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

show_animation = True
on_random = False
v = 5
yawrate = 0.1

F = np.matrix([[1.0, 0, 0, 0],
               [0, 1.0, 0, 0],
               [0, 0, 1.0, 0],
               [0, 0, 0, 0]])

class NeuralNetwork:
    def __init__(self):

        self.X = tf.Variable(tf.zeros((1,4)),trainable=True)
        self.z = tf.placeholder(tf.float32,[None,2])
        self.u = tf.placeholder(tf.float32,[2,None])
        self.F = tf.Variable([[1.0, 0, 0, 0],
                       [0, 1.0, 0, 0],
                       [0, 0, 1.0, 0],
                       [0, 0, 0, 0]],trainable=False)
        self.B = tf.Variable([[DT * tf.cos(self.X[0,2]), 0],
                     [DT * tf.cos(self.X[0,2]), 0],
                     [0.0, DT],
                     [1.0, 0.0]],trainable=False)
        self.observation = tf.Variable([[1.0, 0, 0, 0],[0, 1.0, 0, 0]],trainable=False)
        self.logits = tf.matmul(self.F,tf.transpose(self.X)) + tf.matmul(self.B,self.u)
        self.jitter_observation = tf.matmul(self.observation,self.logits)

        self.cost = tf.reduce_mean(tf.abs(self.z - tf.transpose(self.jitter_observation)))
        self.optimizer = tf.train.AdamOptimizer(0.95).minimize(self.cost)

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = NeuralNetwork()
sess.run(tf.global_variables_initializer())

def random_yaw():
    global yawrate
    yawrate = np.random.randn()

def left_yaw():
    global yawrate
    yawrate = np.random.rand()

def right_yaw():
    global yawrate
    yawrate = np.random.rand()*-1

def calc_input():
    u = np.matrix([v, yawrate]).T
    return u

def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)
    # add noise to gps x-y
    zx = xTrue[0, 0] + np.random.randn() * Qsim[0, 0]
    zy = xTrue[1, 0] + np.random.randn() * Qsim[1, 1]
    z = np.matrix([zx, zy])
    # add noise to input
    ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
    ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
    ud = np.matrix([ud1, ud2]).T
    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud

def motion_model(x, u):
    B = np.matrix([[DT * math.cos(x[2, 0]), 0],
                   [DT * math.sin(x[2, 0]), 0],
                   [0.0, DT],
                   [1.0, 0.0]])
    x = F * x + B * u
    return x

def observation_model(x):
    #  Observation Model
    H = np.matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0]])
    return H * x

def jacobF(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.matrix([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF

def jacobH(x):
    # Jacobian of Observation Model
    jH = np.matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0]])
    return jH

def ekf_estimation(xEst, PEst, z, u):

    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacobF(xPred, u)
    PPred = jF * PEst * jF.T + Q

    #  Update
    jH = jacobH(xPred)
    zPred = observation_model(xPred)
    y = z.T - zPred
    S = jH * PPred * jH.T + R
    K = PPred * jH.T * np.linalg.inv(S)
    xEst = xPred + K * y
    PEst = (np.eye(len(xEst)) - K * jH) * PPred
    return xEst, PEst


def plot_covariance_ellipse(xEst, PEst):
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    R = np.matrix([[math.cos(angle), math.sin(angle)],
                   [-math.sin(angle), math.cos(angle)]])
    fx = R * np.matrix([x, y])
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def main():
    time = 0.0
    # State Vector [x y yaw v]'
    xEst = np.matrix(np.zeros((4, 1)))
    xTrue = np.matrix(np.zeros((4, 1)))
    PEst = np.eye(4)
    xDR = np.matrix(np.zeros((4, 1)))  # Dead reckoning
    # history
    hxEst = xEst
    hxTrue = xTrue
    hxNN = np.matrix(np.zeros((4, 1)))
    hxDR = xTrue
    hz = np.zeros((1, 2))
    real_z = np.zeros((1, 2))
    plt.figure(figsize=(6,6))
    while True:
        u = calc_input()
        xTrue, z, xDR, ud = observation(xTrue, xDR, u)
        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)
        cost,_ = sess.run([model.cost,model.optimizer],feed_dict={model.z:z,model.u:ud})

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        logits = sess.run(model.logits,feed_dict={model.u:u})
        hxNN = np.hstack((hxNN,logits))
        hz = np.vstack((hz, z))
        real_z = np.vstack((real_z,xTrue[:2].T))

        plt.cla()
        plt.plot(hz[:, 0], hz[:, 1], ".g",label='noise gps signals')
        plt.plot(real_z[:, 0], real_z[:, 1], ".g", marker='x',label='true gps signals')
        plt.plot(np.array(hxTrue[0, :]).flatten(), np.array(hxTrue[1, :]).flatten(), "-b",label='true path')
        plt.plot(np.array(hxDR[0, :]).flatten(), np.array(hxDR[1, :]).flatten(), "-k",label='dead reckoning')
        plt.plot(np.array(hxEst[0, :]).flatten(),np.array(hxEst[1, :]).flatten(), "-r",label='predict path EKF')
        plt.plot(np.array(hxNN[0, :]).flatten(),np.array(hxNN[1, :]).flatten(), "-y",label='predict path NN without Q R')
        plot_covariance_ellipse(xEst, PEst)
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        print('yawrate: %f'%(yawrate))
        print('NN, true path: (%f,%f), predict path: (%f,%f), abs loss: %f'%(hxTrue[0, -1],
        hxTrue[1, -1],hxNN[0, -1],hxNN[1, -1],np.mean(np.abs(hxTrue[:2,-1] - hxNN[:2,-1]))))
        print('EKF, true path: (%f,%f), predict path: (%f,%f), abs loss: %f'%(hxTrue[0, -1],
        hxTrue[1, -1],hxEst[0, -1],hxEst[1, -1],np.mean(np.abs(hxTrue[:2,-1] - hxEst[:2,-1]))))
        return_input = input("Action: ")
        print()
        if return_input == 'random':
            random_yaw()
        if return_input == 'right':
            right_yaw()
        if return_input == 'left':
            left_yaw()
        plt.tight_layout()
        plt.pause(0.001)

if __name__ == '__main__':
    main()
