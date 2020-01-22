from ipywidgets import *
import numpy as np
import matplotlib.pyplot as plt

def imageBrowse(im, im_ax=0, rang_min = 0, rang_max = 1.5, colormap_c='gray', fig_title='T1 weighted series'):
    """imageBrowse: Simple visualisation tool. It accepts 3D or 4D matrices.
        This method plots the 3D or 4D images (if 4D, multiple views are available).
        It also shows the signal evolution for a given point, which can be chosen by mouse interaction.
        
        Input: [C x H x W x D] or [C x H x W], where C is the channel dimension (i.e. Time series)
        Output: None
    """
    
    dim = len(np.shape(im))
    
    if dim == 4:
        x1 = np.shape(im)[1]-1
        x2 = np.shape(im)[2]-1
        x3 = np.shape(im)[3]-1
    if dim == 3:
        x1 = np.shape(im)[2]-1
        x2 = np.shape(im)[1]-1
        x3 = 0
    
    global lr
    global ax_dim
    ax_dim = []
    lr=[]
    
    ax1=50
    ax2=50
    ax3=10
    
    coord = {}
    coord["x"] = 50
    coord["y"] = 50
    coord["z"] = 10
    
    vmin_range=rang_min
    vmax_range=rang_max
    
    it = np.linspace(1,len(im), len(im))
    
    if dim == 4:
        fig, ax = plt.subplots(1,4)
    if dim == 3:
        fig, ax = plt.subplots(1,2, figsize=(8,8))
    
    if dim == 4:
        line0, = [ax[0].imshow(im[im_ax, ax1, :, :], vmin=vmin_range, vmax=vmax_range, cmap=colormap_c)]
        line1, = [ax[1].imshow(im[im_ax, :, ax2, :], vmin=vmin_range, vmax=vmax_range, cmap=colormap_c)]
        line2, = [ax[2].imshow(im[im_ax, :, :, ax3], vmin=vmin_range, vmax=vmax_range, cmap=colormap_c)]
        line3, = ax[3].plot(it, im[:, ax1, ax2, ax3], '--xr', linewidth=0.5)
        ax[3].set_ylim([np.min(im[:, ax1, ax2, ax3])-0.1, np.max(im[:, ax1, ax2, ax3])+0.1])
    if dim == 3:
        line0, = [ax[0].imshow(im[im_ax], vmin=vmin_range, vmax=vmax_range, cmap=colormap_c)]
        ax[0].set_title(fig_title)

        cbar = fig.colorbar(line0, ax=ax[0],fraction=0.040, pad=0.04)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('[s]', rotation=270)

        line3, = ax[1].plot(it, im[:, ax1, ax2], '--xr', linewidth=0.5)
        ax[1].set_ylim([np.min(im[:, ax1, ax2])-0.1, np.max(im[:, ax1, ax2])+0.1])
        ax[1].set_xlabel('Inversion time Index')
        ax[1].set_ylabel('[s]')
        x0,x1 = ax[1].get_xlim()
        y0,y1 = ax[1].get_ylim()
        ax[1].set_aspect((x1-x0)/(y1-y0))
        fig.tight_layout(pad=3.0)
        
    
    lr.append(ax[0].axvline(x=int(coord["x"]),color='red', linewidth=0.5))
    lr.append(ax[0].axhline(y=int(coord["y"]),color='red', linewidth=0.5))
    if dim == 4:
        lr.append(ax[1].axvline(x=int(coord["x"]),color='red', linewidth=0.5))
        lr.append(ax[1].axhline(y=int(coord["z"]),color='red', linewidth=0.5))
        lr.append(ax[2].axvline(x=int(coord["y"]),color='red', linewidth=0.5))
        lr.append(ax[2].axhline(y=int(coord["z"]),color='red', linewidth=0.5))
    
    def update(inversion_time = (0,len(im)-1)):
        
        if dim == 4:
            line0.set_data(im[inversion_time, dim1, :, :])
            line1.set_data(im[inversion_time, :, dim2, :])
            line2.set_data(im[inversion_time, :, :, dim3])
            line3.set_ydata(im[:, dim1, dim2, dim3])
            ax[3].set_ylim([np.min(im[:, dim1, dim2, dim3])-0.1, np.max(im[:, dim1, dim2, dim3])+0.1])
        if dim == 3:
            line0.set_data(im[inversion_time])
        
        if ax_dim:
            ax_dim.pop(-1)
            
        ax_dim.append(inversion_time)
        fig.canvas.draw_idle()
        
    def onclick(event):
        if not ax_dim:
            ax_dim.append(im_ax)
        
        if dim == 4:
            lr[-6].remove()
            lr[-5].remove()
            lr[-4].remove()
            lr[-3].remove()
        lr[-2].remove()
        lr[-1].remove()
        
        if dim == 4:
            if event.x>200 and event.x<320:
                coord["x"] = event.xdata
                coord["z"] = event.ydata
                line2.set_data(im[im_ax, :, :, int(coord["x"])])
                line0.set_data(im[im_ax, int(coord["z"]), :, :])
            elif event.x>320:
                coord["y"] = event.xdata
                coord["z"] = event.ydata
                line1.set_data(im[im_ax, :, int(coord["y"]), :])
                line0.set_data(im[im_ax, int(coord["z"]), :, :])
            elif event.x<190:
                coord["x"] = event.xdata
                coord["y"] = event.ydata
                line1.set_data(im[im_ax, :, int(coord["y"]), :])
                line2.set_data(im[im_ax, :, :, int(coord["x"])])
                
            line3.set_ydata(im[:, int(coord["x"]), int(coord["y"]), int(coord["z"])])
            ax[3].set_ylim([np.min(im[:, int(coord["x"]), int(coord["y"]), int(coord["z"])])-0.1, np.max(im[:, int(coord["x"]), int(coord["y"]), int(coord["z"])])+0.1])
            lr.append(ax[0].axvline(x=int(coord["x"]),color='red', linewidth=0.5))
            lr.append(ax[0].axhline(y=int(coord["y"]),color='red', linewidth=0.5))
            lr.append(ax[1].axvline(x=int(coord["x"]),color='red', linewidth=0.5))
            lr.append(ax[1].axhline(y=int(coord["z"]),color='red', linewidth=0.5))
            lr.append(ax[2].axvline(x=int(coord["y"]),color='red', linewidth=0.5))
            lr.append(ax[2].axhline(y=int(coord["z"]),color='red', linewidth=0.5))
            
        if dim == 3:
            coord["x"] = event.xdata
            coord["y"] = event.ydata
            line0.set_data(im[ax_dim[-1]])
            line3.set_ydata(im[:, int(coord["y"]), int(coord["x"])])
            ax[1].set_ylim([np.min(im[:, int(coord["x"]), int(coord["y"])])-0.1, np.max(im[:, int(coord["x"]), int(coord["y"])])+0.1])
            
            lr.append(ax[0].axvline(x=int(coord["x"]),color='red', linewidth=0.5))
            lr.append(ax[0].axhline(y=int(coord["y"]),color='red', linewidth=0.5))
        
        fig.canvas.draw_idle()
        return lr

    l_hand = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.show()
    interact(update, x=0);