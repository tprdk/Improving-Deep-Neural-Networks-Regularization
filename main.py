import matplotlib.pyplot as plt
from reg_utils import load_2D_dataset


# set default size of plots
plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#If the dot is blue, it means the French player managed to hit the ball with his/her head
#If the dot is red, it means the other team's player hit the ball with their head
train_X, train_Y, test_X, test_Y = load_2D_dataset()
plt.show()

