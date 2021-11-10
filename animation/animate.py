import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import matplotlib
from matplotlib import cm



x = np.linspace(0,3,100)
y = np.linspace(0.2,1.2, 100)
x,y = np.meshgrid(x,y)
z_list = [np.loadtxt(f'animation/TFI{i+1}.csv', delimiter=',') for i in range(50)]
z_list[0] = 0*z_list[0]+1
fps = 5 # frame per sec
frn = len(z_list) # frame number of the animation
print(frn)
zarray = np.stack(z_list, axis=2)


def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, zarray[:,:,frame_number], cmap="magma", rcount=100, ccount=100, vmin=0, vmax=1)

fig, ax = plt.subplots(figsize=(10,10/1.618), subplot_kw={"projection": "3d"})

plot = [ax.plot_surface(x, y, zarray[:,:,0], cmap=cm.inferno, rcount=100, ccount=100)]
ax.set_zlim(0,1)
ax.set_xlim(0,3)
ax.set_ylim(0.2,1.2)
ax.set_xlabel('t')
ax.set_ylabel('h')
ax.set_zlabel(r'$ \langle ' + r'\sigma_x' + r' \rangle$', fontsize=8)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(zarray, plot), interval=1000/fps)
ani.save('training_animation_1'+'.gif',writer='imagemagick',fps=fps)