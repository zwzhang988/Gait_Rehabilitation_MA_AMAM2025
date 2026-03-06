import numpy as np
import wandb
import matplotlib.pyplot as plt
import pdb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def mplfig_to_npimage(fig):
    """ Converts a matplotlib figure to a RGB frame after updating the canvas"""
    #  only the Agg backend now supports the tostring_rgb function
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw() # update/draw the elements

    # get the width and the height to resize the matrix
    l,b,w,h = canvas.figure.bbox.bounds
    w, h = int(w), int(h)

    #  exports the canvas to a string buffer and then to a numpy nd.array
    buf = canvas.tostring_rgb()
    image= np.frombuffer(buf,dtype=np.uint8)
    return image.reshape(h,w,3)


x = np.linspace(0,1,100)
fig, axs = plt.subplots(5,5)
pdb.set_trace()
axs = np.array(axs)
ax1 = axs[0,0]
ax2 = axs[0,1]
ax1.plot(np.sin(x))
ax2.plot(np.sin(x))

plt.tight_layout()

img = mplfig_to_npimage(fig)

wandb_img = wandb.Image(img)
# wandb.log({"analysis": wandb_img})

plt.show()


