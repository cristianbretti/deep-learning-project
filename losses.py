import numpy as np
import matplotlib.pyplot as plt

f = open('models/model_big/loss_raw.txt', 'r')

loss_lines = f.readlines()

losses = []  # 5600
epochs = []  # 28
e_count = 0
for line in loss_lines:
    loss_list = line.split()
    if loss_list[0].strip() == 'batch':
        losses.append(float(loss_list[-1].strip()))
    else:
        epochs.append(e_count)
        e_count += 1

avg = []
x_values = []
count = 0
for i in range(len(losses)):
    count += losses[i]
    if not i % (10000/50) and i != 0:
        avg.append(count / (10000/50))
        x_values.append(i - 100)
        count = 0
avg.append(count / (10000/50))
x_values.append(5600 - 100)


plt.title('Loss graph over 28 epochs')
plt.plot(losses)
plt.plot(x_values, avg)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xticks(np.arange(0, len(losses), step=200), epochs)
# plt.axis([0, 28, 0, 6])
plt.show()
