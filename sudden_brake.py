from cvxopt import matrix, solvers
from agent import Agent
from utils import get_dist
import numpy as np
import matplotlib.pyplot as plt
import time
        
def main():
    ag1_y_update = 50
    ag3_y_update = 50


    p_horizon = 50
    u_horizon = 2
    # t_l_change = 100

    ### initialize vg and wg
    vg = 10*np.ones((p_horizon,1))
    wg = 0*np.ones((p_horizon,1))

    y_lane = np.arange(-1000,1000)
    x_l_lane = -7.5*np.ones(y_lane.shape)
    x_m_lane = -2*np.ones(y_lane.shape)
    x_r_lane = 2*np.ones(y_lane.shape)


    agent1 = Agent(1, [0,0,np.deg2rad(90)],[0,50,np.deg2rad(85)], vg, wg, p_horizon, u_horizon)
    # agent2 = Agent(2, [-5,0,np.deg2rad(90)],[-5,50,np.deg2rad(85)], vg, wg, p_horizon, u_horizon)
    agent3 = Agent(3, [0,agent1.c_state[1]+30,np.deg2rad(90)],[0,45,np.deg2rad(85)], vg, wg, p_horizon, u_horizon)
    # agent4 = Agent(4, [-5,agent2.c_state[1]+5,np.deg2rad(90)],[-5,45,np.deg2rad(85)], vg, wg, p_horizon, u_horizon)


    agent1.obstacles = [agent3]
    agent1.avoid_obs = True
    agent1.vl = 2

    # agent2.obstacles = [agent1, agent3, agent4]
    # agent2.avoid_obs = False
    # agent2.vl = 10

    agent3.obstacles = [agent1]
    agent3.avoid_obs = False
    agent3.vl = 0

    # agent4.obstacles = [agent1, agent2, agent3]
    # agent4.avoid_obs = False
    # agent4.vl = 10

    th = 0.5
    timeout = 50
    rec_video = False
    if(rec_video):
        plt_sv_dir = "frames/"
        p = 0
    # dist1 = [] 
    # dist1.append(get_dist(agent1.c_state, agent2.c_state))
    count = 0
    plt.ion()
    plt.show()
    y_l_lim = -1
    y_u_lim = 100
    update_y = 0
    while( ( (np.linalg.norm(agent1.c_state-agent1.g_state)>th) or (np.linalg.norm(agent3.c_state-agent3.g_state)>th) ) and timeout>0):
        agent1.pred_controls()
        #if(agent3.brake == False):
        #    agent3.pred_controls()
        for i in range(u_horizon):
            if(np.linalg.norm(agent1.c_state-agent1.g_state)>th):
                agent1.v = agent1.vg[i]
                agent1.w = agent1.wg[i]
                agent1.v_list.append(agent1.v)
                agent1.x_traj = []
                agent1.y_traj = []
                agent1.get_traj(i)
                agent1.non_hol_update()
                agent1.v_y = agent1.v*np.sin(agent1.c_state[2])

            
            # if(np.linalg.norm(agent3.c_state-agent3.g_state)>th):
            #     agent3.v = agent3.vg[i]
            #     agent3.w = agent3.wg[i]
            #     agent3.v_list.append(agent3.v)
            #     agent3.x_traj = []
            #     agent3.y_traj = []
            #     agent3.get_traj(i)
            #     if agent3.brake == True:
            #         agent3.v = 0
            #         agent3.w = 0
            #         agent3.vg = matrix(np.zeros((p_horizon,1)))  # initial velocity
            #         agent3.wg = matrix(np.zeros((p_horizon,1))) # initial angular velocity

            #     agent3.non_hol_update()
            #     agent3.v_y = agent3.v*np.sin(agent3.c_state[2])

                
    #         dist1.append(get_dist(agent1.c_state, agent2.c_state))
            
            plt.annotate('Velocity:'+str(round(agent1.v,2)), xy=(-30,agent1.c_state[1]))

            plt.scatter(agent1.g_state[0],agent1.g_state[1],marker='x', color='r')
            plt.scatter(agent1.x_traj, agent1.y_traj,marker='.', color='cyan', s=1)
            plt.plot([agent1.c_state[0],agent1.g_state[0]],[agent1.c_state[1],agent1.g_state[1]], linestyle='dotted', c='k')
            
            plt.scatter(agent3.g_state[0],agent3.g_state[1],marker='x', color='r')
            plt.scatter(agent3.x_traj, agent3.y_traj,marker='.', color='cyan', s=1)
            plt.plot([agent3.c_state[0],agent3.g_state[0]],[agent3.c_state[1],agent3.g_state[1]], linestyle='dotted', c='k')
            
            plt.plot(x_r_lane,y_lane,'r')
            plt.plot(x_m_lane,y_lane,'r')
            plt.plot(x_l_lane,y_lane,'r')

            xa,ya = agent1.draw_circle(agent1.c_state[0], agent1.c_state[1])
            xc,yc = agent3.draw_circle(agent3.c_state[0], agent3.c_state[1])
            #xd, yd = agent1.draw_dmin(agent1.c_state[0], agent1.c_state[1])
            #plt.plot(xd,yd,'g',linewidth=1)
            plt.plot(xa,ya,'b',linewidth=1)
            plt.annotate('1', xy=(agent1.c_state[0], agent1.c_state[1]+2.5))
            plt.plot(xc,yc,'k',linewidth=1)
            plt.annotate('3', xy=(agent3.c_state[0], agent3.c_state[1]+2.5))
            
            plt.xlim([-35,35])
            plt.ylim([y_l_lim, y_u_lim])

            plt.title("Agent 1 is active agent | Update interval: "+str(u_horizon))  
            if(rec_video):
                plt.savefig(plt_sv_dir+str(p)+".png",dpi=500, bbox_inches='tight')
                p = p+1
                plt.clf()
            else:
                plt.pause(1e-10)
                plt.clf()
            timeout = timeout - agent1.dt

        update_y = update_y + 1
        if(update_y>= 5):
            update_y = 0
            y_l_lim = agent1.c_state[1] - 10
            y_u_lim = agent1.c_state[1] + 100

        agent1.g_state[1] = agent1.c_state[1] + ag1_y_update    
        # agent3.g_state[1] = agent3.c_state[1] + ag3_y_update
        agent1.vl = agent1.v
        agent1.wl = agent1.w
        # agent3.vl = agent3.v
        # agent3.wl = agent3.w
    
        
        count = count + 1
        if(count == 20):
            agent3.brake = True
  
    agent1.avg_time = sum(agent1.time_list[1:])/len(agent1.time_list[1:])
    max_time = max(agent1.time_list[1:])
    min_time = min(agent1.time_list[1:])
    print("average time taken for each optimization step: {} secs".format(agent1.avg_time))
    print("max time taken for each optimization step: {} secs".format(max_time))
    print("min time taken for each optimization step: {} secs".format(min_time))

    if(timeout <= 0):
        print("Stopped because of timeout.")


if __name__ == "__main__":
    main()
