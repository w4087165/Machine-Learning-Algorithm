def fun1():
    import random
    l = []
    for i in range(50):
        l.append([random.random()*20+30,random.random()*30+20,0])
    for i in range(50):
        l.append([random.random()*30+60,random.random()*30+60,1])
    with open('LogiReg_data.txt','w') as f:
        for i in l:
            for j in i:
                if i.index(j)!=2:
                    f.write(str(j)+',')
                else:
                    f.write(str(j)+'\n')

fun1()