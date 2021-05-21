# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig = plt.figure(figsize=(20,30))

img=mpimg.imread('../images/rubik.jpeg')

def onclick(event):
    ix, iy = event.xdata, event.ydata
    print(ix, iy)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

imgplot = plt.imshow(img)
plt.show()