import random



rand = random.sample(range(33403), 3342)

f_train = open("train.txt", "a")
f_val = open("valid.txt", "a")

for i in range(1, 33403):
    print(str(i) + '.txt')
    if i in rand:
        f_val.write('data/custom/images/train/'+str(i)+'.png'+'\n')
    else:
        f_train.write('data/custom/images/train/'+str(i)+'.png'+'\n')


f_train.close()
f_val.close()



