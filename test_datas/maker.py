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

def fun2():
    l = [
        ['Budweiser',144, 15, 4.7, 0.43],
        ['Schiltz',151, 19, 4.9, 0.43],
        ['Beer1',157, 15, 0.9, 0.48],
        ['Beer2',170,7,5.2,0.73],
        ['Beer3',152,11,5.0,0.77],
        ['Beer4',145,23,4.6,0.28],
        ['Beer5',175,24,5.5,0.40],
        ['Beer6',149,27,4.7,0.42],
        ['Beer7',99,10,4.3,0.43],
        ['Beer8',113,8,3.7,0.40],
        ['Beer9',140,18,4.6,0.44],
        ['Beer10',102,15,4.1,0.46],
        ['Beer11',135,11,4.2,0.50],
        ['Beer12',150,19,4.7,0.76],
        ['Beer13',149,6,5.0,0.79],
        ['Beer14',68,15,2.3,0.38],
        ['Beer15',139,19,4.4,0.43],
        ['Beer16',144,24,4.9,0.43],
        ['Beer17',72,6,2.9,0.46],
        ['Beer18',97,7,4.2,0.47],
    ]
    with open('Beer_data.txt','w') as f:
        for i in l:
            for j in i:
                if i.index(j)!=len(i)-1:
                    f.write(str(j)+',')
                else:
                    f.write(str(j)+'\n')

fun2()