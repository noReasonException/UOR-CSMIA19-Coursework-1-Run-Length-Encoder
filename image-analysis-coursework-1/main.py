import cv2
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import sys
#curr_image = np.array(image.imread("img/1/IC1.png"))
import pickle
"""


a = imread('img/1/IC1-low.jpg')
% Compute the 2D fft.
r=a(:,:,1)
g=a(:,:,2)
b=a(:,:,3)
r_plane = fftshift(fft2(r));
g_plane = fftshift(fft2(g));
b_plane = fftshift(fft2(b));
%===============================mapper to freq domain




%===============================reverse mapper to spatial domain 
%recreate the image using inverse 2FFT 
inv_r = abs(ifft2(ifftshift(r_plane)));
inv_g = abs(ifft2(ifftshift(g_plane)));
inv_b = abs(ifft2(ifftshift(b_plane)));

ze=zeros(225,300)
w1=uint8(cat(3,inv_r,ze,ze))
w2=uint8(cat(3,ze,inv_g,ze))
w3=uint8(cat(3,ze,ze,inv_b))
w=uint8(cat(3,inv_r,inv_g,inv_b))



imshowpair(w,a, 'montage');
"""




def mapper(image):
    curr_image = image

    r=curr_image[:,:,0]                         #Red Channel as nxm
    g = curr_image[:, :, 1]                     #Green Channel as nxm
    b = curr_image[:, :, 2]                     #Blue Channel as nxm
    r_plane = np.fft.fftshift(np.fft.fft2(r))   #Red plane FFT2
    g_plane = np.fft.fftshift(np.fft.fft2(g))   #Green plane FFT2
    b_plane = np.fft.fftshift(np.fft.fft2(b))   #Blue plane FFT2
    return (r_plane,g_plane,b_plane)


def inverse_mapper(r_plane,g_plane,b_plane):
    inv_r = np.abs(np.fft.ifft2(np.fft.ifftshift(r_plane))) #Red plane IFFT2
    inv_g = np.abs(np.fft.ifft2(np.fft.ifftshift(g_plane))) #Green plane IFFT2
    inv_b = np.abs(np.fft.ifft2(np.fft.ifftshift(b_plane))) #Blue plane IFFT2

    #Here i added an additional axis in order to be able to concetenate the channels at 3rd
    #axis (the color)
    return np.concatenate((
        np.expand_dims(inv_r, axis=2),
        np.expand_dims(inv_g, axis=2),
        np.expand_dims(inv_b ,axis=2))
        ,axis=2)#.astype(np.int32) no need if loaded using plt.imread

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional


def run_length_coding_redundancy_mid(symbols,i,rep=1):

    if(i+1<len(symbols)):
        if(symbols[i]==symbols[i+1] and rep<900) : return run_length_coding_redundancy_mid(symbols,i+1,rep+1)
        else:return [rep,symbols[i]]
    else: return [rep,symbols[i]]


def run_length_coding_encode(channel):
    print("ENCODE")
    symbols=channel.flatten()
    assert len(symbols)==channel.shape[0]*channel.shape[1]
    #run_length_cod=np.ndarray((0,),np.float32)
    #run_length_cod=np.append(run_length_cod,[channel.shape[0],channel.shape[1]])

    run_length_cod_fast=[]
    run_length_cod_fast.extend([channel.shape[0],channel.shape[1]])

    i=0
    current_run=sys.maxsize
    while(i<len(symbols)):
        if((i/len(symbols)*100)%5==0 or (i/len(symbols)*100)%2==0):
            print(str(i/len(symbols))+"completed")
        curr=run_length_coding_redundancy_mid(symbols,i)
        if(current_run==abs(curr[0])):
            #run_length_cod = np.append(run_length_cod, [curr[1]])
            run_length_cod_fast.extend([curr[1]])
        else:
            #run_length_cod = np.append(run_length_cod, [-curr[0],curr[1]])
            run_length_cod_fast.extend([-curr[0],curr[1]])
            current_run=curr[0]
        i = i + current_run
        if(current_run==0):
            print("WARN")
    #return run_length_cod
    return np.array(run_length_cod_fast,np.int32)

def run_length_coding_decode(encoded):
    print("DECODE")
    symbols=[]
    shape=(encoded[0].astype(np.int32),encoded[1].astype(np.int32))
    encoded=encoded[2:] #first two contain the shape
    curr_step=sys.maxsize
    for i in range(len(encoded)):
        if(encoded[i]<0):
            curr_step=abs(encoded[i])
            continue
        a=curr_step.astype(np.int32) * [encoded[i]]
        symbols.extend(a)
    return np.array(symbols,np.int32)\
        .reshape(shape)


def run_length_cod(image):



    #np.ndArrays
    r=image[:,:,0]
    g=image[:,:,1]
    b=image[:, :, 2]






    #r = (255*r).astype(np.int32)
    #g = (255*g).astype(np.int32)
    #b = (255*b).astype(np.int32)



    print("RAW- "+str(r.shape)+str(g.shape)+str(b.shape))

    r=run_length_coding_encode(r)
    g = run_length_coding_encode(g)
    b = run_length_coding_encode(b)

    print("ENCODED- "+str(len(r))+"-"+str(len(r))+"-"+str(len(r)))



    r = run_length_coding_decode(r)
    g = run_length_coding_decode(g)
    b = run_length_coding_decode(b)

    print("DECODED- "+str(r.shape)+str(g.shape)+str(b.shape))

    return np.concatenate((
        np.expand_dims(r, axis=2),
        np.expand_dims(g, axis=2),
        np.expand_dims(b ,axis=2))
        ,axis=2)


def run_length_cod_single(image):
    image = (image * 255).astype(np.int32)
    print("RAW- " + str(image.shape))

    single=collapse_rgb_to_bitmap(image)


    print("SINGLE-LAYER- "+str(single.shape))

    single=run_length_coding_encode(single)

    print("ENCODED- "+str(len(single)))

    print(single)

    single = run_length_coding_decode(single).astype(np.int32)

    print("DECODED- "+str(single.shape))

    return expand_bitmap_to_rgb(single)



def collapse_rgb_to_bitmap(image:np.ndarray):
    def toSingle(r: int, g: int, b: int) -> int:
        retval = r + (g << 8 & (pow(2, 8) - 1) << 8) + (b << 16 & (pow(2, 16) - 1) << 16)
        return retval
    new_image=np.ndarray((image.shape[0],image.shape[1]),np.int32)
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i,j]=toSingle(image[i,j,0],image[i,j,1],image[i,j,2])
    return new_image

def expand_bitmap_to_rgb(bitmap:np.ndarray):
    def fromSingle(s: int) -> tuple:
        return ((pow(2, 8) - 1) & s, ((pow(2, 16) - 1) & (~255) & s) >> 8, ((pow(2, 24) - 1) & (~65535) & s) >> 16)
    new_image=np.ndarray((bitmap.shape[0],bitmap.shape[1],3),np.int32)
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            r,g,b=fromSingle(bitmap[i,j])
            new_image[i,j,0]=r
            new_image[i, j, 1] = g
            new_image[i, j, 2] = b
    return new_image


original=plt.imread("img/1/IC2.png")
#original=plt.imread("IC1-low.png")
a=mapper(original)
b=inverse_mapper(*a)

size=original.shape
a=np.array(original).flatten()
a=a.reshape(size)




dicta={
    "original":a,
    "mapped-inv_mapped":run_length_cod_single(original)
}

plot_figures(dicta,1,2)
plt.show()