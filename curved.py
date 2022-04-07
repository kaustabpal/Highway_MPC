from agent import Agent
import utils
from utils import get_dist
import numpy as np
import matplotlib.pyplot as plt

def main():
    p_horizon = 50
    u_horizon = 3
    ### initialize vg and wg
    vg = 11*np.ones((p_horizon,1))
    wg = 0*np.ones((p_horizon,1))

    xr_lane = np.arange(-100,100,0.1)
    xmid1_lane = np.arange(-97,97,0.1)
    xmid2_lane = np.arange(-94,94,0.1)
    xl_lane = np.arange(-91,91,0.1)
    y_r_lane = np.sqrt(100**2 - xr_lane **2)
    y_mid1_lane = np.sqrt(97**2 - xmid1_lane**2)
    y_mid2_lane = np.sqrt(94**2 - xmid2_lane**2)
    y_l_lane = np.sqrt(91**2 - xl_lane**2)
    drawlist = []
    c_radius = 95.5 
    th_goal = np.deg2rad(20) 
    axgoal = c_radius * np.cos(th_goal)
    aygoal = c_radius * np.sin(th_goal) 
    agent1 = Agent(1, [98.5,5,np.deg2rad(90)],[axgoal,aygoal,np.deg2rad(90)+th_goal], vg, wg, p_horizon, u_horizon, curve=True)
    agent1.v_ub = 20
    agent1.v_lb = 10
    agent1.vl = 11.11
    drawlist.append(agent1)
    radiuses = [92.5,95.5,98.5]
    obstacles = []
    th = np.deg2rad(10)
    th_goal = th+np.deg2rad(15)
    for i in range(10):
        c_radius = radiuses[np.random.randint(0,3)]
        xspawn = c_radius * np.cos(th)
        yspawn = c_radius * np.sin(th)
        xgoal = c_radius * np.cos(th_goal)
        ygoal = c_radius * np.sin(th_goal)
        obstacles.append(Agent(i+2,[xspawn, yspawn, np.deg2rad(90)+th],[xgoal,ygoal,np.deg2rad(90)+th_goal], vg, wg, p_horizon, u_horizon, curve=True))
        th = th + np.deg2rad(10)
        th_goal = th+np.deg2rad(20)

    for i in range(len(obstacles)):
        obstacles[i].v_ub =11
        obstacles[i].v_lb = 11
        obstacles[i].vl = 11
        drawlist.append(obstacles[i])
        #agent1.obstacles.append(obstacles[i])
        agent1.avoid_obs = True

   # for o in obstacles:
   #     eo_id = o.id
   #     o.avoid_obs = True
   #     for oo in obstacles:
   #         if(eo_id == oo.id):
   #             continue
   #         else:
   #             o.obstacles.append(oo)
    timeout = 50
    rec_video =  False
    if(rec_video):
        plt_sv_dir = "../../2_pipeline/tmp/"
        p = 0

    while( agent1.c_state[0]>0 and timeout > 0):
        agent1.pred_controls()
        for o in obstacles:
            o.pred_controls()
        for i in range(u_horizon):
            agent1.v = agent1.vg[i]
            agent1.w = agent1.wg[i]
            agent1.v_list.append(agent1.v)
            agent1.x_traj = []
            agent1.y_traj = []
            agent1.get_traj(i)
            agent1.non_hol_update()
            print("Orientation: ", np.rad2deg(agent1.c_state[2]))
            for o in obstacles:
                o.v = o.vg[i]
                o.w = o.wg[i]
                o.v_list.append(o.v)
                o.x_traj = []
                o.y_traj = []
                o.get_traj(i)
                o.non_hol_update()
            
            utils.draw(drawlist)
            plt.plot(xr_lane,y_r_lane,'k')
            plt.plot(xmid1_lane,y_mid1_lane,'k')
            plt.plot(xmid2_lane,y_mid2_lane,'k')
            plt.plot(xl_lane,y_l_lane,'k')

            plt.title("Agent 1 has Obstacle avoidance | Update interval: "+str(u_horizon))  
            plt.xlim([agent1.c_state[0]-25, agent1.c_state[0]+25])
            plt.ylim([agent1.c_state[1]-25, agent1.c_state[1]+25])
            
            if(rec_video):
                plt.savefig(plt_sv_dir+str(p)+".png",dpi=500, bbox_inches='tight')
                p = p+1
                plt.clf()
            else:
                plt.pause(1e-10)
                plt.clf()
            timeout = timeout - agent1.dt

        th = np.arctan2(agent1.c_state[1],agent1.c_state[0]) 
        th = th + np.deg2rad(20) #np.arange(0,2*np.pi,0.01)
        c_radius = 95.5
        agent1.g_state[0] = c_radius * np.cos(th)
        agent1.g_state[1] = c_radius * np.sin(th)
        agent1.g_state[2] = np.deg2rad(90)+th
        #update_y = update_y + 1
        #if(update_y>= 2):
        #    update_y = 0
        #    y_l_lim = agent1.c_state[1] - 30
        #    y_u_lim = agent1.c_state[1] + 30
        #    x_l_lim = agent1.c_state[0] - 30
        #    x_u_lim = agent1.c_state[0] + 30
        
#     agent1.g_state[1] = agent1.c_state[1] + 55    
        agent1.vl = agent1.v
        agent1.wl = agent1.w
       
    agent1.avg_time = sum(agent1.time_list[1:])/len(agent1.time_list[1:])
    print("average time taken for each optimization step: {} secs".format(agent1.avg_time))

    if(timeout <= 0):
        print("Stopped because of timeout.")


if __name__ == "__main__":
    main()
