# MPC.py

import numpy as np

import cvxpy as cvx


def update(x_0, x_t, comm_depth):
    # f,df,x_new,u_old,M

    x_list = np.zeros([4, depth+1])
    x_list[:, 0:comm_depth] = x_t
    for i in range(comm_depth, depth+1):
        x_list[:, i] = np.dot(A, x_list[:, i-1])

    '''
    x_list = np.zeros([depth+1, 4])
    x_list[0, :] = x_t
    for i in range(depth):
        x_t = np.dot(x_t, A)
        x_list[i+1, :] = x_t
	'''
    x_init.value = x_0  # np.zeros(4)
    x_target.value = x_list  # np.ones([5,4])
    prob.solve()
    
    # print x_init.value
    err_cost = np.dot(np.dot((x_0-x_list[:, 0]), P), (x_0-x_list[:, 0]))
    ctl_cost = float(
        np.dot(np.transpose(u.value[:, 0]), np.dot(Q, u.value[:, 0])))
    x = np.zeros([4, comm_depth+1])
    x[:, 0] = x_0
    for i in range(1, comm_depth+1):
        x[:, i] = np.dot(A,x[:, i-1])+np.reshape(B*u.value[:, i-1],(4))


    #print u.value
    #print x_list 
    #print x
    return u.value[:, 0], x[:, 1:comm_depth+1], err_cost, ctl_cost
    '''
	for i in range(M)-1:
		u_old
	'''


def a_2D_quad_problem(P, Q, dt, depth):
    a_max = 2

    u = cvx.Variable(2, depth)
    x_init = cvx.Parameter(4)
    x_target = cvx.Parameter(4, depth+1)
    #x_init.value = x0
    x_temp = x_init
    J = 0
    constriants = []
    for i in range(depth):
        # x.append(x_temp)
        J += cvx.quad_form(x_temp-x_target[:, i], P)+cvx.quad_form(u[:, i], Q)
        # constriants.append(cvx.abs(u[0,i]*x_temp[2]+u[1,i]*x_temp[3])-cvx.norm(x_temp[2:3])*a_max<=0)

        #constriants.append(cvx.norm(u[:, i]) < a_max)

        #J += np.dot(np.dot((x_temp-x_target[i]),P),x_temp-x_target[i])+cvx.quad_form(u[:,i],Q)
        x_temp = A*x_temp+B*u[:, i]
    i += 1
    J += cvx.quad_form(x_temp-x_target[:, i], P)
    prob = cvx.Problem(cvx.Minimize(J), constriants)
    return prob, u, x_init, x_target, dt




depth = 5
dt = 0.1


P = np.eye(4)
Q = 0.1*np.eye(2)
A = np.eye(4, dtype=float)
A[0, 2] = dt
A[1, 3] = dt
#A[2, 0] = dt
#A[3, 1] = dt
B = np.zeros([4, 2])
B[2, 0] = dt
B[3, 1] = dt


prob, u, x_init, x_target, dt = a_2D_quad_problem(P, Q, dt, depth)
print "MPC Model Constructed"
if __name__ == '__main__':
    update(x_0=np.array([4.33717921, 0., 0.00950343, 0.]),
           x_t=np.array([[5], [0], [1], [0]]),comm_depth = 1)
