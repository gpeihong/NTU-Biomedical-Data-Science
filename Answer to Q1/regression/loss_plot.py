from pathlib import Path
from matplotlib import pyplot as plt

# read loss logs from txt file
f_loss = Path("./loss_value.txt")
s_loss = f_loss.read_text().splitlines() 
l_loss = [{'epoch': int(x.split(' ')[1]), 'loss': float(x.split(' ')[3])} for x in s_loss]

# plot style
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# draw lines on plot
ax.plot([x['epoch'] for x in l_loss], [x['loss'] for x in l_loss])

# plot beautify
x_0, x_1 = ax.get_xlim()
y_0, y_1 = ax.get_ylim()
ax.set_aspect((x_1 - x_0) / (y_1 - y_0))
ax.set_ylabel(r'Loss', fontsize=12)
ax.set_xlabel(r'Epoch', fontsize=12)

# show plot
plt.savefig('loss_value.png')
