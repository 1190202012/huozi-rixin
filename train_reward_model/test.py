import re

file = open("log.txt", "r", encoding="UTF-8")
text = file.read()

texts =  re.findall(r"epoch: \d, step: \d*, loss: [.\d]*", text)

step_list = []
loss_list = []

for i, t in enumerate(texts):
    epoch, step, loss = t.replace("epoch: ", "").replace("step: ", "").replace("loss: ", "").split(", ")
    epoch = int(epoch)
    step = int(step) + epoch * 440
    loss = float(loss)

    assert step == i, f"line {i}, epoch: {epoch}, step: {step}, loss: {loss}\ntext: {t}"
    step_list.append(step)
    loss_list.append(loss)

import matplotlib.pyplot as plt

plt.plot(step_list,loss_list,color = 'r')
plt.xlabel("step")#横坐标名字
plt.ylabel("loss")#纵坐标名字
plt.savefig("loss变化图")
