import cv2
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
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
        else:return [symbols[i],rep]
    else: return [symbols[i],rep]


def run_length_coding_encode(channel):
    print("ENCODE")
    symbols=channel.flatten()
    assert len(symbols)==channel.shape[0]*channel.shape[1]
    run_length_cod=np.ndarray((0,),np.int32)
    run_length_cod=np.append(run_length_cod,[channel.shape[0],channel.shape[1]])
    i=0
    while(i<len(symbols)):
        curr=run_length_coding_redundancy_mid(symbols,i)
        run_length_cod=np.append(run_length_cod,[curr[0],curr[1]])
        i=i+curr[1]
    assert run_length_cod.shape[0]%2==0#should always insert in pairs
    return run_length_cod

def run_length_coding_decode(encoded):
    assert encoded.shape[0] % 2 == 0  # should always insert in pairs
    print("DECODE")
    symbols=[]
    shape=(encoded[0].astype(np.int32),encoded[1].astype(np.int32))
    encoded=encoded[2:] #first two contain the shape

    for i in range(0,len(encoded),2):
        a=encoded[i + 1].astype(np.int32) * [encoded[i]]
        symbols.extend(a)
    return np.array(symbols)\
        .reshape(shape)

def toBinaryFormat(r_encoded,g_encoded,b_encoded):
    print(type(r_encoded))
    return None

def run_length_cod(image):

    #np.ndArrays
    r=image[:,:,0]
    g=image[:,:,1]
    b=image[:, :, 2]





    print("RAW- "+str(r.shape)+str(g.shape)+str(b.shape))

    r=run_length_coding_encode(r)
    g = run_length_coding_encode(g)
    b = run_length_coding_encode(b)

    #list -> should be array
    toBinaryFormat(r, g, b)
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





#original=plt.imread("img/1/IC2.png")
original=plt.imread("IC1-low.jpg")
a=mapper(original)
b=inverse_mapper(*a)

size=original.shape
a=np.array(original).flatten()
a=a.reshape(size)




dicta={
    "original":a,
    "mapped-inv_mapped":run_length_cod(original)
}

plot_figures(dicta,1,2)
plt.show()