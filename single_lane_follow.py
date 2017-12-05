import numpy as np
from numpy.linalg import norm
import math
import random
import cvxpy as cvx
import matplotlib.pyplot as plt

import simulator
from simulator import p_state
from simulator import msgs
import MPC

class leading_agent(simulator.agent):
    v_max = 5

    def set_data(self, data):
        self.x_est = np.zeros([ 2])
        self.v_est = 0
        self.theta_est = 0
        self.omega = 0
        self.a_p = 0
        self.agent_num = data[0]
        self.comm_radius = data[1]
        self.rel_dist = data[2]
        
        self.state_exp = np.zeros([4,comm_depth])

        self.t = 0



    def est_init(self):
        self.x_est = self.measure(simulator.absolute_pos)
        p_state.control(self.id, v=self.v_est * np.array([math.cos(self.theta_est),math.sin(self.theta_est)]))
    def control_algorithm(self):
        self.state_exp[0,0] = self.x_est[0]
        self.state_exp[2,0] = self.v_est
        for i in range(1,comm_depth):
            self.state_exp[:,i] = np.dot(self.state_exp[:,i-1],A)
        #print self.state_exp
        for id in range(self.comm_radius):
            if self.id+id+1 < self.agent_num:
                self.send(self.id+id+1, 'prev_node', [self.x_est,self.v_est,self.theta_est])
                self.send(self.id+id+1,'prev_state_exp',self.state_exp)
        self.v_est = 1+0.5*random.random()-0.25
        
        #0.5*math.sin(self.t)
        p_state.control(self.id, v = self.v_est * np.array([math.cos(self.theta_est),math.sin(self.theta_est)]))
    def update(self, time_interval):
        self.x_est += self.v_est * time_interval * np.array([math.cos(self.theta_est),math.sin(self.theta_est)])
        self.v_est += self.a_p * time_interval
        self.theta_est += self.omega * time_interval
        self.t += time_interval




class follow_agent(simulator.agent):
    v_max = 5

    def set_data(self, data):
        self.x_est =  np.zeros([ 2])
        self.v_est = 0
        self.theta_est = 0
        self.omega = 0
        self.a_p = 0
        self.agent_num = data[0]
        self.comm_radius = data[1]
        self.rel_dist = data[2]
        self.state_exp = np.zeros([4,comm_depth])
        self.target_exp = np.zeros([4,comm_depth])

        self.err_cost = []
        self.ctl_cost = []

    def est_init(self):
        self.x_est = self.measure(simulator.absolute_pos)

    def control_algorithm(self):
        for id in range(self.comm_radius):
            if self.id+id+1 < self.agent_num:
                self.send(self.id+id+1, 'prev_node', [self.x_est,self.v_est,self.theta_est])
                self.send(self.id+id+1,'prev_state_exp',self.state_exp)
        rel_pos_x = []
        v_x = []
        prev_state = []

        res_msg = self.receive()
        for msg in res_msg:
            if msg.tag == 'prev_node':
                rel_pos_x.append(msg.data[0][0])
                v_x.append(msg.data[1]* math.cos(msg.data[2]))
            elif msg.tag == 'prev_state_exp':
                prev_state.append(msg.data)
        m = len(rel_pos_x)
        
        err_cost = 0
        ctl_cost = 0
        if m!=0:
            self.target_exp = sum(prev_state)/m
            bias = np.matlib.repmat(np.array([-float(m+1)/2*self.rel_dist,0,0,0]),comm_depth,1)  
            #x_goal = sum(rel_pos_x)/m-float(m+1)/2*self.rel_dist
            #v_goal = sum(v_x)/m
            x_0 = np.zeros(4)
            x_0[0] = self.x_est[0]
            x_0[1] = self.x_est[1]
            x_0[2] = self.v_est * math.cos(self.theta_est)
            x_0[3] = self.v_est * math.sin(self.theta_est)
            a,state_lst,err_cost,ctl_cost = MPC.update(x_0 = x_0,x_t = self.target_exp + np.transpose(bias),comm_depth = comm_depth)
            self.state_exp = state_lst
            self.a_p = a[0]*math.cos(self.theta_est)+a[1]*math.sin(self.theta_est)
            self.a_p = float(self.a_p)
            if self.v_est != 0:
                self.omega = (-a[0]*math.sin(self.theta_est)+a[1]*math.cos(self.theta_est))/self.v_est
                self.omega = float(self.omega)
            p_state.control(self.id, a = np.array([float(a[0]),float(a[1])]))
        self.err_cost.append(err_cost)
        self.ctl_cost.append(ctl_cost)
    def update(self, time_interval):
        self.x_est += self.v_est * time_interval * np.array([math.cos(self.theta_est),math.sin(self.theta_est)])
        self.v_est += self.a_p * time_interval
        self.theta_est += self.omega * time_interval


class single_lane_follow_simulator(simulator.simulator):

    def state_init(self, method, param):
        [x, v, a] = method(self.agent_num, self.dimension, param)
        p_state.set_state(x, v, a)
        for agent in self.agents:
            agent.est_init()


 


n = 10
connected_node = 1
comm_depth = 5
dt = 0.1

A = np.eye(4, dtype=float)
A[2, 0] = dt
A[3, 1] = dt


smlt = single_lane_follow_simulator(
    [leading_agent,follow_agent], data=[1+n,connected_node,1], agent_num=[1,n], dimension=2)
smlt.state_init(simulator.line_pos_x, param=[1,[0,0]])
#print p_state.x
#print smlt.agents[0].x_est


#init round 50*4
for j in range(50):
    smlt.iterate(iter_round=4, time_interval=dt)
    #print p_state.v
    smlt.plot(plotfig=False,saveimage = True,plotarrow=False)

print 'end'

ccost = []
ecost = []
for j in range(len( smlt.agents[1].ctl_cost)):
    temp = 0
    for i in range(n):
        temp += smlt.agents[i+1].ctl_cost[j]
    ccost.append(temp)

    #plt.plot(smlt.agents[i+1].ctl_cost)


for j in range(len( smlt.agents[1].err_cost)):
    temp = 0
    for i in range(n):
        temp += smlt.agents[i+1].err_cost[j]
    ecost.append(temp)
    #plt.plot(smlt.agents[i+1].ctl_cost)

np.savez('comm_depth_5_test.npz',ccost = ccost,ecost = ecost)
plt.plot(ccost)
plt.show()
plt.plot(ecost)
plt.show()