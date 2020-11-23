import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def mandelbrot( x, y, threshold):

    c = (complex(x, y))
    z = (complex(0, 0))

    for i in range(threshold):
        z = z**2 + c
        if abs(z) > 4.:
            return i
    
    return threshold -1


x_start, y_start = -2, -1.5
width, height = 3, 3
density_per_unit = 500

re = np.linspace(x_start, x_start + width, width * density_per_unit )
im = np.linspace(y_start, y_start + height, height + density_per_unit )

fig = plt.figure(figsize=(10,10))
ax = plt.axes()

def animate(i):
    ax.clear()
    ax.set_xticks([], [])
    ax.set_yticks([], [])

    X = np.empty((len(re), len(im)))
    threshold = round(1.15**(i + 1))

    for i in range(len(re)):
        for j in range(len(im)):
            X[i, j] = mandelbrot(re[i], im[j], threshold)

    img = ax.imshow(X.T, interpolation='bicubic', cmap='viridis')
    return [img]

anim = animation.FuncAnimation(fig, animate, frames=300, interval=50, blit=True)
plt.show(anim)
