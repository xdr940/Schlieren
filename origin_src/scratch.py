import matplotlib.pyplot as plt
from path import  Path
import numpy as np
import  cv2
from numpy.ma import masked_array
import  argparse
import matplotlib.cm as cm
import matplotlib.colors as colors

from fft_tools import fft,ifft
dir = Path("./pics")
dir_rot = Path('./pics_rot')


def rotation():
    files = dir.files('*.bmp')

    dir_rot.mkdir_p()
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        rows, cols = img.shape  # cols-1 和 rows-1 是坐标限制
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -6.4, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        dst = dst[800:3200,2140:2840]
        cv2.imwrite(dir_rot/file.stem+'_rot.{}'.format('png'),dst)
def magnitude(x, y):
    x_m = x * x
    y_m = y * y
    z_m = x_m + y_m
    return np.sqrt(z_m)

def process():
    files = dir_rot.files('*.png')
    imgs=[]
    edges=[]


    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        imgs.append(img)
        edges.append(cv2.Canny(img, 2, 100))  # 其他的默认
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # 默认ksize=3
        #sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    #dft = cv2.dft(np.float32(imgs[0]), flags=cv2.DFT_COMPLEX_OUTPUT)
    #dft_shift = np.fft.fftshift(dft)
    #magnitude_spectrum2 = 20 * np.log10(magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    print('ok')


def mag2show(dft):
    magnitude_spectrum_show = 20 * np.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))#h,w
    return magnitude_spectrum_show
def fft_test():
    files = dir_rot.files('*.png')
    imgs = []
    edges = []

    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        imgs.append(img)

    img = imgs[6]
    img0 = imgs[0]


    dft = fft(img)
    dft0 = fft(img0)
    dft_show = mag2show(dft)
    dft0_show = mag2show(dft0)
    sub_dft = dft - dft0

    subdft_show = mag2show(sub_dft)


#mask
    # 创建低通滤波器掩模、使用掩模滤波、IDFT
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)#hw2
    spand = 10
    #mask[crow - spand:crow + spand, ccol - spand:ccol + spand] = 0
    #mask[:, ccol - spand:ccol + spand] = 0
    mask[crow - spand:crow + spand, :] = 0

    mask_show = mask[:,:,0]

#
    #dft = dft * mask

    img_bk = ifft(sub_dft)

    imgList = [img, dft0_show, dft_show, subdft_show, img-img0,img_bk]
    #imgName = ['img', 'magnitude_spectrum2', 'img_back', 'img', 'magnitude_spectrum2', 'img_back2']

    for i in range(len(imgList)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(imgList[i])
    #plt.imshow(img_back3)
    plt.show()


def filter1(img):
    kernel_ver = np.array([ 1,0,-1,
                            1,0,-1,
                            1,0,-1]).reshape([3,3])#向右卷积， src中暗处(小)点在src‘中值较大

    kernel_hon = np.array([ -1,-1,-1,
                            0,0,0,
                            1,1,1]).reshape([3,3])

    #dst = cv2.filter2D(src=img,kernel=kernel_hon,ddepth=-1)
    img = cv2.filter2D(src=img,kernel=kernel_ver,ddepth=-1)
   # img = cv2.filter2D(src=img,kernel=kernel_ver,ddepth=-1)


    pass
    print('ok')
    return img

def main():

    #read
    path = Path('./pics_rot/06.png')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #get shadow img
    south = np.array([-1, -2, -1,
                      0, 1, 0,
                      1, 2, 1]).reshape([3, 3])  # 向右卷积， src中暗处(小)点在src‘中值较大

    north = np.array([1, 2, 1,
                      0, 1, 0,
                      -1, -2, -1]).reshape([3, 3])  # 向右卷积， src中暗处(小)点在src‘中值较大

    img_n = cv2.filter2D(src=img, kernel=north, ddepth=-1)
    img_s = cv2.filter2D(src=img, kernel=south, ddepth=-1)



    #define show rigion
    img_n_show = img_n#[1333:1666,400:600]
    img_s_show = img_s#[1333:1666,400:600]
    img_show = img#[1333:1666,400:600]
    img_copy = img_show.copy()



    #
    rigid_M = img_show<3
    width=70 #0~255, 值越大， 波峰越明显， 但同时背景噪声也会越明显
    flow_bkg_M = True^rigid_M

    flow_M = ((img_s_show<width).__or__(img_n_show<width)) *flow_bkg_M
    flow = flow_M*img_show

    bkg_M = flow_bkg_M*(True^flow_M)
    img_show[bkg_M] = 0
    names = [
    'Src Img',
    'Rigid Mask',
    'Flow Mask',
    'BackGround Mask',
    ]
    shows= [img_copy,
            rigid_M,
            flow_M,
            bkg_M,
        ]
   #plt masks and src img
    i = 1
    lens = len(shows)
    cols = 4#lens/2
    if lens%2!=0:
        cols+=1

    while i <=lens:
        plt.subplot(1,cols,i)
        plt.title(names[i-1])
        plt.imshow(shows[i-1])
        i+=1

    #plt render img, 对不同区域的一张图通过mask 进行不同渲染
    rigid_ = masked_array(np.ones_like(rigid_M), rigid_M == False)
    flow_ = masked_array(flow_M.__xor__(bkg_M) * img0_show, flow_M.__xor__(bkg_M) == False)

    #ax = plt.subplot(1, 4, 4)
    fig, ax = plt.subplots()

    ax.imshow(rigid_, interpolation='nearest', cmap='bone')

    flow_p = ax.imshow(flow_, interpolation='nearest', cmap='jet',norm=colors.Normalize(vmin=0, vmax=255))
    plt.colorbar(flow_p, shrink=0.25)

    plt.show()
    print('ok')



def filter_with_mask(src,mask=None,ksize=9,step=3,mask_s=2):
    '''
    根据mask选择性的进行滤波
    :param src:
    :param mask:
    :param kernel:
    :return:
    '''
    dst = np.copy(src)
    h,w = dst.shape
    print(h,w)
    s = int(ksize/2)
    i,stop_h = s,h-s-1
    j,stop_w = s, w-s-1
    print(i,j,stop_h,stop_w)
    try:
        while i < h:
            while j<w:
                if mask[i-mask_s:i+mask_s,j-mask_s:j+mask_s].max()!=0:
                    dst[i,j] = src[i-s:i+s,j-s:j+s].max()
                j+=step
            j=s
            i+=step
    except:
        print('err')
        print(i,j)
            #print('ok')

    return dst

def test_shadow():
    path_n = Path('./pics_rot/03.png')
    img = cv2.imread(path_n)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    south = np.array([-1, -2, -1,
                           0, 1, 0,
                           1, 2, 1]).reshape([3, 3])  # 向右卷积， src中暗处(小)点在src‘中值较大

    north = np.array([1, 2, 1,
                      0, 1, 0,
                      -1, -2,-1]).reshape([3, 3])  # 向右卷积， src中暗处(小)点在src‘中值较大




    img = cv2.filter2D(src=img, kernel=north, ddepth=-1)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    #rotation()
    #process()
    main()
    #test_shadow()
    pass



