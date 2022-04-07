import numpy as np 
import matplotlib.pyplot as plt

def smoothen(data_list):
    last = data_list[0]  # First value in the plot (first timestep)
    weight = 0.9
    smoothed = list()
    for point in data_list:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed


def get_dist(a_state, o_state):
    d = np.sqrt((o_state[0] - a_state[0])**2 + (o_state[1] - a_state[1])**2)
    return d      

def draw_circle(x, y, radius):
    th = np.arange(0,2*np.pi,0.01)
    xunit = radius * np.cos(th) + x
    yunit = radius * np.sin(th) + y
    return xunit, yunit  

def draw(agent_list):
    for i in range(len(agent_list)):
        a = agent_list[i]
        if(a.id == 1):
            col = 'g'
            x_s, y_s = draw_circle(a.c_state[0], a.c_state[1], 20)
            plt.plot(x_s,y_s,'k',linewidth=1)
            plt.scatter(a.g_state[0], a.g_state[1], marker='x', color='r')
            plt.scatter(a.x_traj, a.y_traj, marker='.', color='blue', s=1)
        else:
            col = 'r'
            plt.scatter(a.x_traj, a.y_traj, marker='.', color='cyan', s=1)

        x, y = draw_circle(a.c_state[0], a.c_state[1], a.radius)
        x_s, y_s = draw_circle(a.c_state[0], a.c_state[1], 20)
        # x2, y2 = draw_circle(a.c_state2[0], a.c_state2[1], a.a_radius)
        # x3, y3 = draw_circle(a.c_state3[0], a.c_state3[1], a.a_radius)
        
        plt.plot(x, y, col, linewidth=1)
        # plt.plot(x2, y2, col, linewidth=1)
        # plt.plot(x3, y3, col, linewidth=1)

        plt.annotate(str(a.id), xy=(a.c_state[0], a.c_state[1]+2.0))
        plt.annotate(str(round(a.v)), xy=(a.c_state[0]-0.6, a.c_state[1]-0.5), size=7)

        #plt.plot([a.c_state[0], a.g_state[0]], [a.c_state[1],a.g_state[1]], linestyle='dotted', c='k')

