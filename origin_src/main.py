import matplotlib.pyplot as plt
from path import  Path
import numpy as np
import  cv2
import scipy.ndimage as ndimage
from numpy.ma import masked_array
import  argparse
import matplotlib.colors as colors
parser = argparse.ArgumentParser(description="Monodepthv2 training options")
parser.add_argument('--width',default=100,help='值越大， 波峰越明显， 但同时背景噪声也会越明显')
parser.add_argument('--rigid_bound',default=5,help='小于该值定义为rigid rigion')
parser.add_argument('--max_gray',default=255,help='主要是渲染到多大的灰度，越小层次越少，越大层次越明显，最大255')
parser.add_argument('--min_gray',default=0,help='主要是渲染到多大的灰度，越小层次越少，越大层次越明显，最大255')

parser.add_argument('--show',default=True,help='是否显示')
parser.add_argument('--save',default=True)
parser.add_argument('--src_p',default='./pics_rot',help='图像序列文件夹')
parser.add_argument('--dst_p',default='./masks')
parser.add_argument('--dst_p2',default='./results')

options = parser.parse_args()
def setting(img,mask,num = 0):

    snap = img*(1-mask)
    mask =np.ones_like(mask)*num*mask
    snap = snap+mask
    snap = np.clip(snap, 0, 255)  # 归一化也行
    snap = np.array(snap, np.uint8)
    return snap

def setting3(img,mask,num=0):
    if mask.shape ==img.shape:
        pass
    else:
        mask=mask.expand_dim(axis = 2)
        mask = np.concatenate([mask,mask,mask],axis=2)
    snap = img * (True ^ mask)
    mask = np.ones_like(mask) * mask*np.ones([1,3])*num
    snap = snap + mask
    snap = np.clip(snap, 0, 255)  # 归一化也行
    snap = np.array(snap, np.uint8)
    return snap

def main(options):

    #read
    path = Path(options.src_p+'/03.png')
    dst_p = Path(options.dst_p).mkdir_p()
    dst_p2 = Path(options.dst_p2).mkdir_p()

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
    snap = img_show.copy()


#
    rigid_M_ = img_show<options.rigid_bound
    rigid_M_ = rigid_M_.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    #先erosion 再 dilation 去除孤立点， 并将rigid 区域变大一些
    rigid_M = cv2.erode(rigid_M_,kernel,iterations=2)
    rigid_M = cv2.dilate(rigid_M, kernel, iterations=4)




    width=options.width
    flow_bkg_M = 1-rigid_M#
#
    flow_M = ((img_s_show<width).__or__(img_n_show<width)) *flow_bkg_M
    flow_M = flow_M.astype(np.uint8)
    flow = flow_M*img_show
#
    bkg_M = flow_bkg_M*(1-flow_M)
    #img_show[bkg_M] = 255
    snap = setting(img,bkg_M,255)
    names = [
    'Src Img',
    'Rigid Mask',
    'Rigid Mask dilation',
    'Flow Mask',
    'BackGround Mask',
    'mask sum'
    ]

    #img_copy = setting(img_copy,rigid_M,0)
    #snap = setting(snap,bkg_M,0)

    #snap = cv2.applyColorMap(snap, cv2.COLORMAP_JET)

    shows= [255-img_show*flow_M,
            rigid_M_,
            rigid_M,
            flow_M,
            bkg_M,
            np.array(rigid_M,np.uint8)+np.array(flow_M,np.uint8)+np.array(bkg_M,np.uint8)
        ]
   #plt masks and src img
    fig1 = plt.figure(figsize=[8, 4])

    i = 1
    lens = len(shows)
    cols = 6  # lens/2
    if lens % 2 != 0:
        cols += 1
    while i <=lens:
        plt.subplot(1,cols,i)
        plt.title(names[i-1])
        plt.imshow(shows[i-1])
        i+=1

    #plt render img, 对不同区域的一张图通过mask 进行不同渲染
    rigid_ = masked_array(np.ones_like(rigid_M), rigid_M == False)
    flow_ = masked_array(flow_M.__xor__(bkg_M) * snap, flow_M.__xor__(bkg_M) == False)

    fig = plt.figure(figsize=[6,8])


    ax = fig.add_subplot(1,1,1)
    ax.imshow(rigid_, interpolation='nearest', cmap='bone')

    #1 fig
    #flow = cv2.applyColorMap((255-flow_), cv2.COLORMAP_JET)
    #flow_p = ax.imshow(flow)
    #2
    flow_p = ax.imshow((255-flow_), interpolation='nearest', cmap='jet',norm=colors.Normalize(vmin=options.min_gray, vmax=options.max_gray))
    plt.colorbar(flow_p, shrink=0.25)
    plt.title('Rendering')

    #save and show
    if options.save:
        fig1.savefig(dst_p/(path.stem+'.png'))
        fig.savefig(dst_p2 / (path.stem + '.png'))
    if options.show:
        plt.show()
    print('ok')


if __name__ == '__main__':
    main(options)


