
import numpy as np
from imagepy.core.engine import Simple, Tool, Filter
from imagepy import IPy
import cv2
from skimage import color
from imagepy.core.manager import  ColorManager

class Undo(Simple):
    title = 'Undo'
    note = ['all']

    def run(self, ips, img, buf, para=None):
        ips.swap()
class Fill(Filter):
    title = 'Fill'
    note = ['req_roi', 'all', 'auto_snap', 'not_channel']

    def run(self, ips, snap, img, para=None):
        img[ips.get_msk()] = ColorManager.get_front(ips.channels!=3)

class Invert(Filter):
    title = 'Invert'
    note = ['all', 'auto_msk', 'auto_snap', 'preview']

    def run(self, ips, snap, img, para=None):
        np.subtract(ips.range[1], snap, out=img)
class RGB2Gray(Simple):
    title = 'RGB To Gray'
    note = ['rgb']

    def run(self, ips, imgs, para = None):
        gray = []
        for i in range(ips.get_nslices()):
            gray.append(color.rgb2gray(imgs[i])*255)
            self.progress(i, len(imgs))
        IPy.show_img(gray, ips.title+'-Gray')


# this is a Filter Sample, implements the Gaussian Blur
class Schlieren(Filter):
    # the title on the menu
    title = 'Schlieren'
    # the describe parameter
    note = ['all', 'auto_msk', 'auto_snap','preview']

    # parameter
    para = {'rigid_bound':5,'width':100}
    # how to interact with the para, it is in 0~30, and 1 decimal
    view = [(float, 'rigid_bound', (0,30), 1,  'rigid_bound', 'pix'),
            (float, 'width', (0,100), 1,  'width', 'pix')]

    # handle the image, img -> buf
    def setting(self,img, mask, num=0):

        snap = img * (True ^ mask)
        mask = np.ones_like(mask) * num * mask
        snap = snap + mask
        snap = np.clip(snap, 0, 255)  # 归一化也行
        snap = np.array(snap, np.uint8)
        return snap

    def setting3(self,img, mask, num):
        if mask.shape == img.shape:
            pass
        else:
            mask = np.expand_dims(mask, axis=2)
            mask = np.concatenate([mask, mask, mask], axis=2)
        num = np.array(num).reshape(1, 3)
        snap = img * (True ^ mask)
        mask = np.ones_like(mask) * mask * np.ones([1, 3]) * num
        snap = snap + mask

        return snap

    def run(self, ips, snap, img, para = None):


        rigid_bound = para['rigid_bound']
        width = para['width']

        snap = img.copy()
        snap = np.clip(snap, 0, 255)  # 归一化也行
        snap = np.array(snap, np.uint8)
        south = np.array([-1, -2, -1,
						  0, 1, 0,
						  1, 2, 1]).reshape([3, 3])  # 向右卷积， src中暗处(小)点在src‘中值较大

        north = np.array([1, 2, 1,
                          0, 1, 0,
                          -1, -2, -1]).reshape([3, 3])  # 向右卷积， src中暗处(小)点在src‘中值较大

        img_n = cv2.filter2D(src=img, kernel=north, ddepth=-1)
        img_s = cv2.filter2D(src=img, kernel=south, ddepth=-1)



        rigid_M_ = img<rigid_bound
        rigid_M_ = rigid_M_.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        rigid_M = cv2.erode(rigid_M_, kernel, iterations=2)
        rigid_M = cv2.dilate(rigid_M, kernel, iterations=4)




        flow_bkg_M = 1-rigid_M#
        #
        flow_M = ((img_s<width).__or__(img_n<width)) *flow_bkg_M
        flow_M = flow_M.astype(np.uint8)
        flow = flow_M*img
        #
        bkg_M = flow_bkg_M*(1- flow_M)
        snap = np.clip(snap, 0, 255)  # 归一化也行
        snap = np.array(snap, np.uint8)
        snap = cv2.applyColorMap(snap, cv2.COLORMAP_JET)
        snap = self.setting3(snap,np.array(rigid_M, np.uint8),[0,0,0])
        snap = self.setting3(snap,np.array(bkg_M, np.uint8),[0,0,128])
        IPy.show_img([snap],ips.title+'Schlieren')
        #IPy.show_img([rigid_M*255],ips.title+'_rigid dilation')
        #IPy.show_img([rigid_M_*255],ips.title+'_rigid')



        return #snap


class Crop(Simple):
    title = 'Crop'
    note = ['all', 'req_roi']

    def run(self, ips, imgs, para = None):
        sc, sr = ips.get_rect()
        if ips.is3d:
            imgs = imgs[:, sc, sr].copy()
        else:
            imgs = [i[sc,sr].copy() for i in imgs]
        ips.set_imgs(imgs)
        #if not ips.backimg is None:
        #    ips.backimg = ips.backimg[sc, sr]
        ips.roi = ips.roi.affine(np.eye(2), (-sr.start, -sc.start))




plgs = [Undo, Fill,'-',  Invert,RGB2Gray,Crop,Schlieren]
