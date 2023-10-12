import os

p1='/data/lulingxiao/murecom/MureCom'

p2_=os.listdir(p1)

for p2 in p2_:
    pp=p1+'/'+p2+'/fg'
    ppn=p1+'/'+p2+'/fg3'
    if os.path.exists(pp):
        os.rename(pp,ppn)
