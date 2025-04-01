from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from scipy.interpolate import CubicSpline as spl
from scipy.interpolate import PchipInterpolator as pci
from scipy.ndimage import center_of_mass as com
from math import exp
import sys
from functools import cmp_to_key
    

def DerichePlus1D(arr,a):
    eArr = np.zeros(arr.size+2)
    eArr[2:] = arr
    eArr[1]  = arr[0]
    eArr[0]  = arr[0]
    gamma = np.zeros(eArr.size)
    for k in range(2,eArr.size):
        gamma[k] = eArr[k-1] + 2* exp(-a)*gamma[k-1]-exp(-2*a)*gamma[k-2]
    return gamma[2:]

def DericheMinus1D(arr,a):
    eArr = np.zeros(arr.size+2)
    eArr[-2] = arr[-1]
    eArr[-1] = arr[-1]
    gamma = np.zeros(eArr.size)
    for k in range(eArr.size-3,0,-1):
        gamma[k] = -eArr[k+1]+2*exp(-a)*gamma[k+1]-exp(-2*a)*gamma[k+2]
    return gamma[:eArr.size-2]

def directionalDeriche(arr,pt,v,extent,a):
    halfExtent = (extent-1)//2
    val = np.zeros(extent)
    for k in range(0,extent):
        currentPoint = (pt-(halfExtent-k)*v).astype(int)
        val[k] = arr[currentPoint[0], currentPoint[1]]
    gDp = DerichePlus1D(val,a)
    gDm = DericheMinus1D(val,a)
    gD = -(1-exp(-a))**2 * (gDp + gDm)
    return np.abs(gD), np.arange(-extent//2, extent//2)


def directionalDerivative(arr,pt,v, extent): #extent must be an odd integer
    halfExtent = (extent-1)//2
    val = np.zeros(extent)
    for k in range(0,extent):
        currentPoint = (pt-(halfExtent-k)*v).astype(int)
        val[k] = arr[currentPoint[0], currentPoint[1]]

    return val


def interpolateArray(arr):
    x = np.arange(-len(arr)//2,len(arr)//2)
    y = arr
    cs = spl(x,y)
    xs = np.arange(-len(arr)//2, len(arr)//2, 0.01)
    ys = cs(xs)
    return xs,ys,x,y



def dericheColsPlus(arr,a):
    eArr = np.zeros((arr.shape[0], arr.shape[1]+2))
    eArr[:,2:] = arr
    #eArr[:,1]  = arr[:,0]
    #eArr[:,0]  = arr[:,0]
    gamma = np.zeros(eArr.shape)
    ncols = eArr.shape[1]
    for k in range(2,ncols):
        gamma[:,k] = eArr[:,k-1]+2*exp(-a)*gamma[:,k-1]-exp(-2*a)*gamma[:,k-2]
    return gamma[:,2:]

def dericheColsMinus(arr,a):
    eArr = np.zeros((arr.shape[0], arr.shape[1]+2))
    eArr[:,-2] = arr[:,-1]
    eArr[:,-1] = arr[:,-1]
    eArr[:,:arr.shape[1]] = arr
    gamma = np.zeros(eArr.shape)
    ncols = eArr.shape[1]
    for k in range(ncols-3,0,-1):
        gamma[:,k] = -eArr[:,k+1] +2*exp(-a)*gamma[:,k+1]-exp(-2*a)*gamma[:,k+2]

    return gamma[:,0:ncols-2]

def dericheRowsPlus(arr,a):
    eArr = np.zeros((arr.shape[0]+2,arr.shape[1]))
    eArr[2:,:] = arr
    eArr[1,:] = arr[0,:]
    eArr[0,:] = arr[0,:]
    gamma = np.zeros(eArr.shape)
    nrows = eArr.shape[0]
    for k in range(2,nrows):
        gamma[k,:] = eArr[k-1,:]+ 2*exp(-a)*gamma[k-1,:]-exp(-2*a)*gamma[k-2,:]
    return gamma[2:,:]

def dericheRowsMinus(arr,a):
    eArr = np.zeros((arr.shape[0]+2, arr.shape[1]))
    eArr[-2,:] = arr[-1,:]
    eArr[-1,:] = arr[-1,:]
    eArr[:arr.shape[1],:] = arr
    gamma = np.zeros(eArr.shape)
    nrows = eArr.shape[0]
    for k in range(nrows-3,0,-1):
        gamma[k,:] = -eArr[k+1,:] +2*exp(-a)*gamma[k+1,:]-exp(-2*a)*gamma[k+2,:]
    return gamma[0:nrows-2,:]


if __name__ == "__main__":

    #create the square
    a = np.ones((51,51))
    a[:10,:] = 0
    a[:,-10:] = 0
    a[10,:-10] = 0.5
    a[10:,-10]=0.5
    a[10,-10]= 0.25

    #Sobel gradient map
    sobelx = cv.Sobel(a,cv.CV_64F,1,0,ksize=3)
    sobely = cv.Sobel(a,cv.CV_64F,0,1,ksize=3)
    b = np.sqrt(sobelx**2 + sobely**2)

    cdArr = np.array([0.25,0.5,0.75,1,2,3,4,5,10])#float(sys.argv[1])
    #cdArr = np.array([3])#float(sys.argv[1])
    allCoordsCG = np.zeros(cdArr.size)
    allCoordsInt = np.zeros(cdArr.size)
    k = 0
    for cd in cdArr:
        #Deriche gradient map
        gColsP = dericheColsPlus(a,cd)
        gColsM = dericheColsMinus(a,cd)
        thetaCols = -(1-exp(-cd))**2 * (gColsP+gColsM)

        gRowsP = dericheRowsPlus(a,cd)
        gRowsM = dericheRowsMinus(a,cd)
        thetaRows = -(1-exp(-cd))**2 * (gRowsP+gRowsM)

        theta = np.abs(thetaRows) + np.abs(thetaCols)
        window = 11
        #directional derivative in Deriche
        pt = np.array([10,41])
        ind = np.unravel_index(np.argmax(theta, axis=None), theta.shape)
        #pt = np.array([ind[0], ind[1]])
        v  = np.array([1,-1])
        gradientV = directionalDerivative(theta,pt,v,window)
        print(gradientV)
        xsD,ysD,x,y = interpolateArray(gradientV)
        
        plt.figure()
        plt.plot(xsD,ysD,'c')
        plt.plot(x,y,'*b')
        plt.title(pt)
        
    
        comD = com(gradientV)
        #plt.plot(xsD,ysD)

        #directional Deriche
        #dDG,x = directionalDeriche(a, pt, v, window, cd)
        #xsDd, ysDd = interpolateArray(dDG)
        
        allCoordsCG[k]= comD[0]-(window-1)//2
        allCoordsInt[k] = -xsD[np.argmax(ysD)]
        k = k+1

    plt.figure()
    plt.plot(cdArr,allCoordsCG,'g*')
    plt.plot(cdArr,allCoordsInt,'b*')
    plt.figure()
    plt.plot(cdArr, np.abs(allCoordsCG-allCoordsInt), 'r*')
    plt.show()

"""
plt.figure(1)
plt.imshow(a)
plt.title("Image originale")
plt.figure(2)
plt.imshow(b)
plt.title("Gradient Sobel")
plt.figure(3)
plt.imshow(theta)
plt.title("Gradient Deriche")
plt.figure(4)
plt.plot(xsD,ysD)
plt.title("Interpolation gradient directionnel Deriche")
plt.figure(5)
plt.plot(xsDd,ysDd)
plt.title("Interpolation Deriche directionnel")
plt.show()


v = [0,0,0,0,0,2.475,2.83,0.354,0,0,0,0,0]
v = np.array(v)
x = np.arange(-6,7)

cs = spl(x,v)
si = pci(x,v)
xs = np.arange(-6,7,0.05)

plt.figure()
plt.plot(xs,cs(xs),'c')
plt.plot(xs,si(xs),'r')
plt.plot(x,v,'*g')
plt.figure()
plt.imshow(a)
plt.figure()
plt.imshow(b)
plt.figure()
plt.imshow(sobelx)
plt.figure()
plt.imshow(sobely)
plt.show()
"""
