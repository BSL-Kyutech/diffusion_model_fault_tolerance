from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
from tenpa_msgs.msg import Pressure12
import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

rclpy.init()
node = Node('publisher')
publisher_list = [node.create_publisher(Pressure12, '/tenpa/pressure/desired'+str(i), 10) for i in range(4)]

def press40_pub(input40:list[int]):
    for i in range(4):
        input12 = np.zeros(12)  # 12 input per 1 layer
        input12[0:10] = input40[0+10*i : 10+10*i]
        msg = Pressure12()
        msg.pressure = np.uint16(input12)
        publisher_list[i].publish(msg)

from robotdiffusion import Diffuser
import matplotlib.pyplot as plt
import matplotlib.widgets as wg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Slider

diffusion_model = Diffuser()
diffusion_model.load_dir('model')

outputs = []

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

X = []
Y = []
Z = []

t = 50

first = True

import torch
for i in range(t*3):
    output = torch.Tensor([0 for i in range(40)])
    z = 0.7
    l = 0.85
    x = round(float(l*np.cos(i/(t-1)*2*np.pi)),2)
    y = round(float(l*np.sin(i/(t-1)*2*np.pi)),2)
    X.append(x)
    Y.append(y)
    Z.append(z)
    print(x,y,z)
    if first:
        first = False
        output = diffusion_model.gen({
            '/mocap/rigidbody1/pos0':x,
            '/mocap/rigidbody1/pos1':y,
            '/mocap/rigidbody1/pos2':z
        },{})
    else:
        output = diffusion_model.gen(
            {
                '/mocap/rigidbody1/pos0':x,
                '/mocap/rigidbody1/pos1':y,
                '/mocap/rigidbody1/pos2':z
            },
            {}
        )
    outputs.append(output)
print(outputs)
#ax.scatter(X, Y, Z, color="red")
ax.plot(X, Y, Z, color="red")
print("plotted")
import pandas as pd

df = pd.read_csv('data.csv')[0:1000]
data_x = df['/mocap/rigidbody1/pos0'].to_numpy()
data_y = df['/mocap/rigidbody1/pos1'].to_numpy()
data_z = df['/mocap/rigidbody1/pos2'].to_numpy()

ax.scatter(data_x, data_y, data_z, color="blue", alpha=0.3)
print("plotted")
result_x = []
result_y = []
result_z = []

def callback(msg):
    global result_x, result_y, result_z
    result_x.append(msg.point.x)
    result_y.append(msg.point.y)
    result_z.append(msg.point.z)

subscriber= node.create_subscription(PointStamped, '/mocap/rigidbody1/marker0', callback, 10)
print("subscribed")
import time

sleep_time = 0.005

first = True

for i in range(len(outputs)):
    press40_pub(outputs[i].tolist()[0])
    if first:
        time.sleep(2)
        first = False
    else:
        time.sleep(sleep_time)
        
    received = False
    while rclpy.ok() and not received:
        rclpy.spin_once(node)
        received = True

    if i+1 < len(outputs):
        resolution = 100
        for j in range(resolution):
            middle = [0 for i in range(40)]
            for k in range(40):
                middle[k] = (outputs[i+1].tolist()[0][k]*j + outputs[i].tolist()[0][k]*(resolution-1-j)) / resolution
            press40_pub(middle)

            time.sleep(sleep_time)
print("pressed")
outputs = [150 for i in range(40)]
press40_pub(outputs)
print("pressed_150")
time.sleep(0.5)

outputs = [200 for i in range(40)]
press40_pub(outputs)
print("pressed_200")
time.sleep(0.5)

outputs = [255 for i in range(40)]
press40_pub(outputs)
print("pressed_250")
subscriber.destroy()
rclpy.shutdown()

ax.plot(result_x, result_y, result_z, color="green")

plt.savefig('circle.png')
plt.show()
print("plotted")
df = pd.DataFrame([result_x, result_y, result_z])
df.to_csv('result_circle_path.csv', index=False)

df = pd.DataFrame([X, Y, Z])
df.to_csv('target_circle_path.csv', index=False)
print("finished")