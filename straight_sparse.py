from agent import Agent
import pickle
import time
import utils
from utils import get_dist
import os
import numpy as np
import matplotlib.pyplot as plt
  
def main():
    rec_video = False
    exp_num =  "highway/Congestion"
    os.makedirs(exp_num+"/tmp/", exist_ok=True)
    p_horizon = 50
    u_horizon = 5
    agent_v_ub = 20
    agent_v_lb = 11.11
    agent_w_ub = 0.1
    agent_w_lb = -0.1
    ### initialize vg and wg
    vg = 12*np.ones((p_horizon,1))
    wg = 0*np.ones((p_horizon,1))
   
    y_lane = np.arange(-1000,1000)
    x1_l_lane = 1.5*np.ones(y_lane.shape)
    x1_r_lane = 4.5*np.ones(y_lane.shape)
    x2_l_lane = -1.5*np.ones(y_lane.shape)
    x3_l_lane = -4.5*np.ones(y_lane.shape)
    draw_list = []
    agent1 = Agent(1, [0,0,np.deg2rad(90)],[0,40,np.deg2rad(90)], vg, wg, p_horizon, u_horizon)
    draw_list.append(agent1)
    obstacles = []
    oy = 0
    obs_x = [-3,0,3]
    obs_y = [15,20,30]
    ox = [3,-3,0,3,0,0,-3,-3,0,3,-3]
    oy = [18,32,50,63,78,89,105,115,109,129,142]
    for i in range(10):
#        oy = oy+ np.random.randint(12,21)
#        ox = obs_x[np.random.randint(0,3)]
#        print("######")
#        print(ox)
#        print(oy)
        obstacles.append(Agent(i+2,[ox[i],oy[i],np.deg2rad(90)],[ox[i],40,np.deg2rad(85)], vg, wg, p_horizon, u_horizon))
    agent1.v_ub = agent_v_ub
    agent1.v_lb = agent_v_lb 
    agent1.w_lb = agent_w_lb
    agent1.w_ub = agent_w_ub
    agent1.vl = 12
    agent1.v_list.append(agent1.vl)
    o_v_ub = [13,14,13,12,11,13,13,13,13,12]
    for i in range(len(obstacles)):
        obstacles[i].v_ub = o_v_ub[i] #np.random.randint(11,15)
        obstacles[i].v_lb = 0
        obstacles[i].w_ub = 0
        obstacles[i].w_lb = 0
        obstacles[i].vl = np.random.randint(10,13) 
        agent1.obstacles.append(obstacles[i])
        draw_list.append(obstacles[i])
    for o in obstacles:
        eo_id = o.id
        o.avoid_obs = True
        for oo in obstacles:
            if(eo_id == oo.id):
                continue
            else:
                o.obstacles.append(oo)

#    agent1.obstacles = [agent2]
   # agent2.obstacles = [agent1]
    agent1.avoid_obs = True
   # agent2.avoid_obs = True

    th = 1
    timeout = 500

    dist = [999999999, 999999999,999999999,999999999,999999999,999999999,999999999,999999999,999999999,999999999] # dist between agent and obstacles 
    min_dist = 999999999
    for i in range(len(obstacles)):
        d = get_dist(agent1.c_state, obstacles[i].c_state)
        if(d<=dist[i]):
            dist[i] = d

    if(rec_video):
        plt_sv_dir = exp_num+"/tmp/"
        p = 0

#    plt.ion()
#    plt.show()
    y_l_lim = -10
    y_u_lim = 40
    update_y = 0
    agent_start_time = time.time()
    while(agent1.c_state[1]<500 and timeout>0):
        agent1.pred_controls()
        for o in obstacles:
            o.pred_controls()
        for i in range(u_horizon):
           #if(np.linalg.norm(agent1.c_state-agent1.g_state)>th):
           agent1.v = agent1.vg[i]
           agent1.w = agent1.wg[i]
           agent1.v_list.append(agent1.v)
           agent1.w_list.append(agent1.w)
           agent1.x_traj = []
           agent1.y_traj = []
           agent1.get_traj(i)
           agent1.non_hol_update()
           for o in obstacles:
               o.v = o.vg[i]
               o.w = o.wg[i]
               o.v_list.append(o.v)
               o.x_traj = []
               o.y_traj = []
               o.get_traj(i)
               o.non_hol_update()
        
           for i in range(len(obstacles)):
               d = get_dist(agent1.c_state, obstacles[i].c_state)
               if(d<=dist[i]):
                   dist[i] = d
                  #dist2.append(get_dist(agent1.c_state, agent2.c_state))

           utils.draw(draw_list)
           plt.plot(x1_r_lane,y_lane,'k', linewidth=1)
           plt.plot(x1_l_lane,y_lane,'k', linewidth=1)
           plt.plot(x2_l_lane,y_lane,'k', linewidth=1)
           plt.plot(x3_l_lane,y_lane,'k', linewidth=1)
           plt.xlim([-25,25])
           plt.ylim([y_l_lim, y_u_lim])
           plt.title("Agent 1 has Obstacle avoidance")  
           
           if(rec_video):
               plt.savefig(plt_sv_dir+str(p)+".png",dpi=500, bbox_inches='tight')
               p = p+1
               plt.clf()
           else:
               plt.pause(1e-10)
               plt.clf()

           update_y = update_y + 1
           if(update_y>= 30):
            update_y = 0
            y_l_lim = agent1.c_state[1] - 10
            y_u_lim = agent1.c_state[1] + 40

           timeout = timeout - agent1.dt
           
           agent1.vl = agent1.v
           agent1.wl = agent1.w
           for o in obstacles:
               o.vl = o.v
               o.wl = o.w

        if(agent1.g_state[1] <=500):
            agent1.g_state[1] = agent1.c_state[1]+40
        for o in obstacles:
            o.g_state[1] = o.c_state[1]+40

        #print(agent1.c_state[1])
    agent_end_time = time.time()

    with open(exp_num+"/tlist", "wb") as fp:   #Pickling
      pickle.dump(agent1.time_list, fp)

    with open(exp_num+"/min_d", "wb") as fp:   #Pickling
      pickle.dump(dist, fp)

    with open(exp_num+"/vlist", "wb") as fp:   #Pickling
      pickle.dump(agent1.v_list, fp)

    with open(exp_num+"/wlist", "wb") as fp:   # Unpickling
      pickle.dump(agent1.w_list,fp)

    print("Time Taken to travel 500m: ", agent_end_time - agent_start_time)
    ### Degugging data ###
    agent1.avg_time = sum(agent1.time_list[1:])/len(agent1.time_list[2:])
    agent1.max_time = max(agent1.time_list[1:])
    agent1.min_time = min(agent1.time_list[1:])

    print("Agent-1 avg time: {} secs".format(agent1.avg_time))
    print("Agent-1 max time: {} secs".format(agent1.max_time))
    print("Agent-1 min time: {} secs".format(agent1.min_time))
    #print("Minimum distance between the agent1 and agent2:",min(np.array(dist2)))
    if(timeout <= 0):
        print("Stopped because of timeout.")
    if(rec_video):
        os.system('ffmpeg -r 10 -f image2 -i '+exp_num+'/tmp/%d.png -s 1000x1000 -pix_fmt yuv420p -y '+exp_num+'/'+exp_num+'.mp4')
    ######################
   # plt.close()
   # plt.plot(b)
   # plt.ylim([0,21])
   # plt.show()

   # plt.plot(agent1.w_list)
   # plt.ylim([-0.5,0.5])
   # plt.show()

if __name__ == "__main__":
    main()
