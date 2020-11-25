



def toSingle(r:int,g:int,b:int)->int:
    retval=r+(g<<8 & (pow(2,8)-1)<<8)+(b<<16 & (pow(2,16)-1)<<16)
    return retval

def fromSingle(s:int)->tuple:
    return ((pow(2,8)-1)&s,((pow(2,16)-1)&(~255)&s)>>8,((pow(2,24)-1)&(~65535)&s)>>16)
"""
a,b,c
a=10010101
b=10010110
c=11011001

a=10
b=11
c=01

mask_a=(2^2-1)<<2
a<<2 & mask_a







"""




print(toSingle(97,126,39))
print(fromSingle(3574643))

print(pow(2,8)-1)