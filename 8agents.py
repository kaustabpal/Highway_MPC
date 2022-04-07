from agent import Agent
from utils import get_dist
import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    p_horizon = 50
    u_horizon = 3 
    
    ### initialize vg and wg
    vg = 0*np.ones((p_horizon,1))
    wg = 0*np.ones((p_horizon,1))
    # vg = np.random.random((p_horizon,1))
    # wg = np.random.random((p_horizon,1))
    
    agent1 = Agent(1,[-60,-10,0.78],[80,30,0.78], vg, wg, p_horizon, u_horizon);
    agent2 = Agent(2,[-60,40,-0.78],[60,-80,-0.78], vg, wg, p_horizon, u_horizon);
    agent3 = Agent(3, [-80,0,0], [80,0,0], vg, wg, p_horizon, u_horizon);
    agent4 = Agent(4,[0,-80,1.5],[0,80,1.5], vg, wg, p_horizon, u_horizon);
    agent5 = Agent(5,[-30,-60,0.7],[80,70,0.7], vg, wg, p_horizon, u_horizon);
    agent6 = Agent(6,[40,-70,2.3],[-40,80,2.3], vg, wg, p_horizon, u_horizon);
    agent7 = Agent(7,[50,-60,1.5],[50,70,1.5], vg, wg, p_horizon, u_horizon);
    agent8 = Agent(8,[-30,60,-1.5],[-30,-60,-1.5], vg, wg, p_horizon, u_horizon);
    
    agent1.obstacles = [agent2, agent3, agent4, agent5, agent6, agent7, agent8]
    # agent2.obstacles = [agent1, agent3, agent4]
    # agent3.obstacles = [agent1, agent2, agent4]
    agent1.avoid_obs = True
    # agent2.avoid_obs = False
    # agent3.avoid_obs = False
    # agent4.avoid_obs = False
    
    th = 2.5
    timeout = 200
    dist2 = [] # dist between 1 and 2 
    dist3 = [] # dist between 1 and 3
    dist4 = [] # dist between 1 and 4
    dist5 = [] # dist between 1 and 5
    dist6 = [] # dist between 1 and 6
    dist7 = [] # dist between 1 and 7
    dist8 = [] # dist between 1 and 8
    
    dist2.append(get_dist(agent1.c_state, agent2.c_state))
    dist3.append(get_dist(agent1.c_state, agent3.c_state))
    dist4.append(get_dist(agent1.c_state, agent4.c_state))
    dist5.append(get_dist(agent1.c_state, agent5.c_state))
    dist6.append(get_dist(agent1.c_state, agent6.c_state))
    dist7.append(get_dist(agent1.c_state, agent7.c_state))
    dist8.append(get_dist(agent1.c_state, agent8.c_state))
    
    rec_video = True 
    if(rec_video):
        plt_sv_dir = "frames/"
        p = 0
    
    while( ( (np.linalg.norm(agent1.c_state-agent1.g_state)>th) or \
            (np.linalg.norm(agent2.c_state-agent2.g_state)>th) or \
            (np.linalg.norm(agent3.c_state-agent3.g_state)>th) or \
            (np.linalg.norm(agent4.c_state-agent4.g_state)>th) or \
            (np.linalg.norm(agent5.c_state-agent5.g_state)>th) or \
            (np.linalg.norm(agent6.c_state-agent6.g_state)>th) or \
            (np.linalg.norm(agent7.c_state-agent7.g_state)>th) or \
            (np.linalg.norm(agent8.c_state-agent8.g_state)>th) ) \
           and timeout>0):
        
        agent2.pred_controls()
        agent3.pred_controls()
        agent4.pred_controls()
        agent5.pred_controls()
        agent6.pred_controls()
        agent7.pred_controls()
        agent8.pred_controls()
        agent1.pred_controls()
        
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

            if(np.linalg.norm(agent2.c_state-agent2.g_state)>th):
                agent2.v = agent2.vg[i]
                agent2.w = agent2.wg[i]
                agent2.v_list.append(agent2.v)
                agent2.x_traj = []
                agent2.y_traj = []
                agent2.get_traj(i)
                agent2.non_hol_update()
            if(np.linalg.norm(agent3.c_state-agent3.g_state)>th):
                agent3.v = agent3.vg[i]
                agent3.w = agent3.wg[i]
                agent3.v_list.append(agent3.v)
                agent3.x_traj = []
                agent3.y_traj = []
                agent3.get_traj(i)
                agent3.non_hol_update()
            if(np.linalg.norm(agent4.c_state-agent4.g_state)>th):
                agent4.v = agent4.vg[i]
                agent4.w = agent4.wg[i]
                agent4.v_list.append(agent4.v)
                agent4.x_traj = []
                agent4.y_traj = []
                agent4.get_traj(i)
                agent4.non_hol_update()
            if(np.linalg.norm(agent5.c_state-agent5.g_state)>th):
                agent5.v = agent5.vg[i]
                agent5.w = agent5.wg[i]
                agent5.v_list.append(agent5.v)
                agent5.x_traj = []
                agent5.y_traj = []
                agent5.get_traj(i)
                agent5.non_hol_update()
            if(np.linalg.norm(agent6.c_state-agent6.g_state)>th):
                agent6.v = agent6.vg[i]
                agent6.w = agent6.wg[i]
                agent6.v_list.append(agent6.v)
                agent6.x_traj = []
                agent6.y_traj = []
                agent6.get_traj(i)
                agent6.non_hol_update()
            if(np.linalg.norm(agent7.c_state-agent7.g_state)>th):
                agent7.v = agent7.vg[i]
                agent7.w = agent7.wg[i]
                agent7.v_list.append(agent7.v)
                agent7.x_traj = []
                agent7.y_traj = []
                agent7.get_traj(i)
                agent7.non_hol_update()
            if(np.linalg.norm(agent8.c_state-agent8.g_state)>th):
                agent8.v = agent8.vg[i]
                agent8.w = agent8.wg[i]
                agent8.v_list.append(agent8.v)
                agent8.x_traj = []
                agent8.y_traj = []
                agent8.get_traj(i)
                agent8.non_hol_update()
            dist2.append(get_dist(agent1.c_state, agent2.c_state))
            dist3.append(get_dist(agent1.c_state, agent3.c_state))
            dist4.append(get_dist(agent1.c_state, agent4.c_state))
            dist5.append(get_dist(agent1.c_state, agent5.c_state))
            dist6.append(get_dist(agent1.c_state, agent6.c_state))
            dist7.append(get_dist(agent1.c_state, agent7.c_state))
            dist8.append(get_dist(agent1.c_state, agent8.c_state))
            
                
            xa,ya = agent1.draw_circle(agent1.c_state[0], agent1.c_state[1])
            xb,yb = agent2.draw_circle(agent2.c_state[0], agent2.c_state[1])
            xc,yc = agent3.draw_circle(agent3.c_state[0], agent3.c_state[1])
            xd,yd = agent4.draw_circle(agent4.c_state[0], agent4.c_state[1])
            xe,ye = agent5.draw_circle(agent5.c_state[0], agent5.c_state[1])
            xf,yf = agent6.draw_circle(agent6.c_state[0], agent6.c_state[1])
            xg,yg = agent7.draw_circle(agent7.c_state[0], agent7.c_state[1])
            xh,yh = agent8.draw_circle(agent8.c_state[0], agent8.c_state[1])
    
            plt.plot(xa,ya,'red',linewidth=1)
            plt.annotate('1', xy=(agent1.c_state[0], agent1.c_state[1]+2.5))
            plt.plot(xb,yb,'b',linewidth=1)
            plt.annotate('2', xy=(agent2.c_state[0], agent2.c_state[1]+2.5))
            plt.plot(xc,yc,'b',linewidth=1)
            plt.annotate('3', xy=(agent3.c_state[0], agent3.c_state[1]+2.5))
            plt.plot(xd,yd,'b',linewidth=1)
            plt.annotate('4', xy=(agent4.c_state[0], agent4.c_state[1]+2.5))
            plt.plot(xe,ye,'b',linewidth=1)
            plt.annotate('5', xy=(agent5.c_state[0], agent5.c_state[1]+2.5))
            plt.plot(xf,yf,'b',linewidth=1)
            plt.annotate('6', xy=(agent6.c_state[0], agent6.c_state[1]+2.5))
            plt.plot(xg,yg,'b',linewidth=1)
            plt.annotate('7', xy=(agent7.c_state[0], agent7.c_state[1]+2.5))
            plt.plot(xh,yh,'b',linewidth=1)
            plt.annotate('8', xy=(agent8.c_state[0], agent8.c_state[1]+2.5))
                
            plt.scatter(agent1.g_state[0],agent1.g_state[1],marker='x', color='r')
            plt.scatter(agent1.x_traj, agent1.y_traj,marker='.', color='cyan', s=1)
            plt.plot([agent1.c_state[0],agent1.g_state[0]],[agent1.c_state[1],agent1.g_state[1]], linestyle='dotted', c='k')
            
            plt.scatter(agent2.g_state[0],agent2.g_state[1],marker='x', color='r')
            plt.scatter(agent2.x_traj, agent2.y_traj,marker='.', color='cyan', s=1)
            plt.plot([agent2.c_state[0],agent2.g_state[0]],[agent2.c_state[1],agent2.g_state[1]], linestyle='dotted', c='k')
            
            plt.scatter(agent3.g_state[0],agent3.g_state[1],marker='x', color='r')
            plt.scatter(agent3.x_traj, agent3.y_traj,marker='.', color='cyan', s=1)
            plt.plot([agent3.c_state[0],agent3.g_state[0]],[agent3.c_state[1],agent3.g_state[1]], linestyle='dotted', c='k')
            
            plt.scatter(agent4.g_state[0],agent4.g_state[1],marker='x', color='r')
            plt.scatter(agent4.x_traj, agent4.y_traj,marker='.', color='cyan', s=1)
            plt.plot([agent4.c_state[0],agent4.g_state[0]],[agent4.c_state[1],agent4.g_state[1]], linestyle='dotted', c='k')
            
            plt.scatter(agent5.g_state[0],agent5.g_state[1],marker='x', color='r')
            plt.scatter(agent5.x_traj, agent5.y_traj,marker='.', color='cyan', s=1)
            plt.plot([agent5.c_state[0],agent5.g_state[0]],[agent5.c_state[1],agent5.g_state[1]], linestyle='dotted', c='k')
            
            plt.scatter(agent6.g_state[0],agent6.g_state[1],marker='x', color='r')
            plt.scatter(agent6.x_traj, agent6.y_traj,marker='.', color='cyan', s=1)
            plt.plot([agent6.c_state[0],agent6.g_state[0]],[agent6.c_state[1],agent6.g_state[1]], linestyle='dotted', c='k')
            
            plt.scatter(agent7.g_state[0],agent7.g_state[1],marker='x', color='r')
            plt.scatter(agent7.x_traj, agent7.y_traj,marker='.', color='cyan', s=1)
            plt.plot([agent7.c_state[0],agent7.g_state[0]],[agent7.c_state[1],agent7.g_state[1]], linestyle='dotted', c='k')
            
            plt.scatter(agent8.g_state[0],agent8.g_state[1],marker='x', color='r')
            plt.scatter(agent8.x_traj, agent8.y_traj,marker='.', color='cyan', s=1)
            plt.plot([agent8.c_state[0],agent8.g_state[0]],[agent8.c_state[1],agent8.g_state[1]], linestyle='dotted', c='k')
            
            plt.xlim([-100,100])
            plt.ylim([-100,100])
            plt.title("Agent 1 has Obstacle avoidance")  
            
            if(rec_video):
                plt.savefig(plt_sv_dir+str(p)+".png",dpi=500, bbox_inches='tight')
                p = p+1
                plt.clf()
            else:
                plt.pause(1e-10)
                plt.clf()
            timeout = timeout - agent1.dt
            
        agent1.vl = agent1.v
        agent1.wl = agent1.w
        agent2.vl = agent2.v
        agent2.wl = agent2.w
        agent3.vl = agent3.v
        agent3.wl = agent3.w
        agent4.vl = agent4.v
        agent4.wl = agent4.w
        agent5.vl = agent5.v
        agent5.wl = agent5.w
        agent6.vl = agent6.v
        agent6.wl = agent6.w
        agent7.vl = agent7.v
        agent7.wl = agent7.w
        agent8.vl = agent8.v
        agent8.wl = agent8.w
        
    agent1.avg_time = sum(agent1.time_list[1:])/len(agent1.time_list[1:])
    print("average time taken by agent1 for each optimization step: {} secs".format(agent1.avg_time))
    # agent2.avg_time = sum(agent2.time_list[1:])/len(agent2.time_list[1:])
    # print("average time taken by agent2 for each optimization step: {} secs".format(agent2.avg_time))
    # agent3.avg_time = sum(agent3.time_list[1:])/len(agent3.time_list[1:])
    # print("average time taken by agent3 for each optimization step: {} secs".format(agent3.avg_time))
    # agent4.avg_time = sum(agent4.time_list[1:])/len(agent4.time_list[1:])
    # print("average time taken by agent4 for each optimization step: {} secs".format(agent4.avg_time))
    
    print("Minimum distance between the agent1 and agent2: ",min(np.array(dist2)))
    print("Minimum distance between the agent1 and agent3: ",min(np.array(dist3)))
    print("Minimum distance between the agent1 and agent4: ",min(np.array(dist4)))
    print("Minimum distance between the agent1 and agent5: ",min(np.array(dist5)))
    print("Minimum distance between the agent1 and agent6: ",min(np.array(dist6)))
    print("Minimum distance between the agent1 and agent7: ",min(np.array(dist7)))
    print("Minimum distance between the agent1 and agent8: ",min(np.array(dist8)))
    
    if(timeout <= 0):
        print("Stopped because of timeout.")

if __name__ == "__main__":
    main()

