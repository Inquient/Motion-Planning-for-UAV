#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:47:50 2018

@author: taisia
"""

from DubinsAirplaneFunctions import DubinsAirplanePath, MinTurnRadius_DubinsAirplane, ExtractDubinsAirplanePath
from PlottingTools import plot3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import argparse

pi = np.pi

class VehicleParameters(object):
    """
        Vehicle Parameters
    """ 
    def __init__(self, Vairspeed_0, Bank_max, Gamma_max):
        self.Vairspeed_0 = Vairspeed_0
        self.Bank_max = Bank_max
        self.Gamma_max = Gamma_max
        
def main(draw = False, saving = False):
    VehiclePars = VehicleParameters(15, pi/4, pi/3 )

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.xticks([-100, -50, 0, 50, 100])
    plt.yticks([-100, -50, 0, 50, 100])
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_aspect(1)
    ax.set_title("Dubins airplane trajectory")
        
    x = [-100,-100,100,100]
    y = [-100,100,100,-100]
    z = [0,0,0,0]
    verts = [list(zip(x, y,z))]
    ax.add_collection3d(Poly3DCollection(verts))
    ax.scatter(0,0,5,alpha=0.0)

    route_scatter = [np.array( [-100, 100, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [100, 100, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [170, 100, 1, -60*pi/180, VehiclePars.Vairspeed_0] ),

                         np.array( [100, 80, 1, 180*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [-100, 80, 1, 180*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [-170, 80, 1, -120*pi/180, VehiclePars.Vairspeed_0] ),

                         np.array( [-100, 60, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [100, 60, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [170, 60, 1, -60*pi/180, VehiclePars.Vairspeed_0] ),

                         np.array( [100, 40, 1, 180*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [-100, 40, 1, 180*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [-170, 40, 1, -120*pi/180, VehiclePars.Vairspeed_0] ),

                         np.array( [-100, 20, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [100, 20, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [170, 20, 1, -60*pi/180, VehiclePars.Vairspeed_0] ),

                         np.array( [100, 0, 1, 180*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [-100, 0, 1, 180*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [-170, 0, 1, -120*pi/180, VehiclePars.Vairspeed_0] ),

                         np.array( [-100, -20, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [100, -20, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [170, -20, 1, -60*pi/180, VehiclePars.Vairspeed_0] ),

                         np.array( [100, -40, 1, 180*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [-100, -40, 1, 180*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [-170, -40, 1, -120*pi/180, VehiclePars.Vairspeed_0] ),

                         np.array( [-100, -60, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [100, -60, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [170, -60, 1, -60*pi/180, VehiclePars.Vairspeed_0] ),

                         np.array( [100, -80, 1, 180*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [-100, -80, 1, 180*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [-170, -80, 1, -120*pi/180, VehiclePars.Vairspeed_0] ),

                         np.array( [-100, -100, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
                         np.array( [100, -100, 1, 0*pi/180, VehiclePars.Vairspeed_0] ),
        ]

    color = 'k'

    full_path = np.empty((0, 3))
    

        
    print ('### Case loaded.')
    print ('### computing path...')

    for i in range(0, len(route_scatter)-1):
        start_node = route_scatter[i]
        end_node = route_scatter[i+1]
        
        R_min = MinTurnRadius_DubinsAirplane( VehiclePars.Vairspeed_0, VehiclePars.Bank_max )
        DubinsAirplaneSolution1 = DubinsAirplanePath( start_node, end_node, R_min, VehiclePars.Gamma_max )

        path_dubins_airplane1 = ExtractDubinsAirplanePath( DubinsAirplaneSolution1 )
        path_dubins_airplane1 = path_dubins_airplane1.T  

        full_path = np.vstack((full_path, path_dubins_airplane1))
        
    if draw:
        
        def animate(i):
            mul = i*100
            ax.plot( full_path[:mul,0], full_path[:mul,1], full_path[:mul,2], color=color)
            
            
        ani = animation.FuncAnimation(fig, animate, save_count=1500, blit=False, repeat=False)
        
        if saving:
            print('### Saving trajectory animation')
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=30)
            ani.save('trajectory.mp4',writer=writer,dpi=250)  
            return
        
        print ('### Showing animated trajectory')
        plt.show()
        return
    
    ax.plot( full_path[:,0], full_path[:,1], full_path[:,2], color)
    print ('### Dubins airplane solution plot')
    plt.show()
    

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trajectory Properties")
    parser.add_argument('--ani', action='store_true')
    parser.add_argument('--save', action='store_true')
    
    args = parser.parse_args()
    
    draw = args.ani
    save = args.save

    main(draw, save)
