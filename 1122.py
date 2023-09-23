all = list()

with open('/home/ysk/data/Coral_bvt_MP6-3_2023-8-8(0229版本)/000040/NAV_PC_xx.log', 'r') as f:
    while(line := f.readline()):
        ss = line.split()
        all.append(ss[0]+ss[1]+ss[2])

print(' '.join(all))

