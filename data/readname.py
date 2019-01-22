import os
f=open('temp.txt','w')
for n in os.listdir('./bmp/'):
    l=n.split('_')[0]
    f.write(n+' '+l+'\n')
