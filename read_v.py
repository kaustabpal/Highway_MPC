import pickle
import utils
from utils import smoothen
import matplotlib.pyplot as plt
import numpy as np
src1 = "highway/v_upper/vel_upper_15"
src2 = "highway/v_upper/vel_upper_17"
src3 = "highway/v_upper/vel_upper_20"
src4 = "highway/v_upper/vel_upper_25"

with open(src1+"/tlist", "rb") as fp:   # Unpickling
      tlist1 = pickle.load(fp)

with open(src2+"/tlist", "rb") as fp:   # Unpickling
      tlist2 = pickle.load(fp)

with open(src3+"/tlist", "rb") as fp:   # Unpickling
      tlist3 = pickle.load(fp)

#bplt = [tlist1[1:], tlist2[1:], tlist3[1:]]
#bp = plt.boxplot(bplt,labels=["vmax = 15","vmax = 17","vmax = 20"])
#for line in bp['medians']:
#    # Get the position of the element. y is the label you want
#    (x_l, y),(x_r, _) = line.get_xydata()
#    # Make sure datapoints exist 
#    # (I've been working with intervals, should not be problem for this case)
#    if not np.isnan(y): 
#        x_line_center = x_l + (x_r - x_l)/2
#        y_line_center = y  # Since it's a line and it's horisontal
#        # overlay the value:  on the line, from center to right
#        plt.text(x_line_center-0.05, y_line_center+0.002, # Position
#                        '%.3f' % y, fontsize=6)
#plt.ylabel("Computation time in seconds")
#plt.title("Trajectory computation time vs vmax")
#plt.savefig("highway/v_upper/t_vs_v_max.png",dpi=300)
#plt.show()

#with open(src1+"/vlist", "rb") as fp:   # Unpickling
#      vlist1 = pickle.load(fp)
#
#with open(src2+"/vlist", "rb") as fp:   # Unpickling
#      vlist2 = pickle.load(fp)
#
#with open(src3+"/vlist", "rb") as fp:   # Unpickling
#      vlist3 = pickle.load(fp)
#vlist1= smoothen(vlist1)
#vlist2= smoothen(vlist2)
#vlist3= smoothen(vlist3)
#
#plt.plot(vlist1,'r',label="vmax = 15",linewidth='1')
#plt.plot(vlist2,'g',label="vmax = 17",linewidth='1')
#plt.plot(vlist3,'b',label="vmax = 20",linewidth='1')
#plt.ylim([10,21])
#plt.ylabel("Velocity in m/s")
#plt.title("Velocity vs timesteps for different vmax")
#plt.legend()
#plt.savefig("highway/v_upper/v_vs_v_max.png",dpi=300)
#plt.show()

#with open(src1+"/wlist", "rb") as fp:   # Unpickling
#      wlist1 = pickle.load(fp)
#
#with open(src2+"/wlist", "rb") as fp:   # Unpickling
#      wlist2 = pickle.load(fp)
#
#with open(src3+"/wlist", "rb") as fp:   # Unpickling
#      wlist3 = pickle.load(fp)
#
#with open(src4+"/wlist", "rb") as fp:   # Unpickling
#      wlist4 = pickle.load(fp)
#
#wlist1= smoothen(wlist1)
#wlist2= smoothen(wlist2)
#wlist3= smoothen(wlist3)
#plt.plot(wlist1,'r',label="vmax = 15",linewidth='1')
#plt.plot(wlist2,'g',label="vmax = 17",linewidth='1')
#plt.plot(wlist3,'b',label="vmax = 20",linewidth='1')
#plt.ylim([-0.6,0.6])
#plt.ylabel("Angular velocity in rad/s")
#plt.xlabel("Timesteps")
#plt.title("Angular velocity vs timesteps for different vmax")
#plt.legend()
#plt.savefig("highway/v_upper/w_vs_v_max.png",dpi=300)
#plt.show()

with open(src1+"/min_d", "rb") as fp:   # Unpickling
      dlist1 = pickle.load(fp)

with open(src2+"/min_d", "rb") as fp:   # Unpickling
      dlist2 = pickle.load(fp)

with open(src3+"/min_d", "rb") as fp:   # Unpickling
      dlist3 = pickle.load(fp)

print(min(dlist1))
print(min(dlist2))
print(min(dlist3))
