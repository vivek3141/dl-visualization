from PIL import Image
import numpy as np

ASPECT_RATIO = 16.0 / 9.0
FRAME_HEIGHT = 8.0
FRAME_WIDTH = FRAME_HEIGHT * ASPECT_RATIO

XRES = 1920
YRES = 1080

RED = (252, 98, 85)
YELLOW = (255, 255, 0)
GREEN = (131, 193, 103)
BLUE = (88, 196, 221)
PURPLE = (154, 114, 172)
colors = [RED, YELLOW, GREEN, BLUE, PURPLE]

x_min = -FRAME_WIDTH/2
x_max = FRAME_WIDTH/2
y_min = -FRAME_HEIGHT/2
y_max = FRAME_HEIGHT/2

x_values = np.linspace(x_min, x_max, XRES+1)
y_values = np.linspace(y_min, y_max, YRES+1)

pixels = []

vertices = [[-0.3003444,  1.51715155,  0.],
            [-0.7003444,  1.51715155,  0.],
            [-0.7003444,  1.11715155,  0.],
            [-0.3003444,  1.11715155,  0.]]

for i in range(len(y_values) - 1)[::-1]:
    pixels.append([])
    for j in range(len(x_values) - 1):
        x1, x2 = x_values[j:j + 2]
        y1, y2 = y_values[i:i + 2]

        inp = [(x1 + x2)/2, (y1 + y2)/2]

        if vertices[1][0] <= inp[0] <= vertices[0][0] and vertices[2][1] <= inp[1] <= vertices[0][1]:
            color = (0,0,0,0)
        else:
            color = (0,0,0,255)
        pixels[-1].append(color)


array = np.array(pixels, dtype=np.uint8)

# img = Image.new(mode="RGBA", size=(XRES, YRES))
# img.putdata(pixels)
# img.show()

new_image = Image.fromarray(array)
new_image.save("highlight_point.png")
new_image.show()
