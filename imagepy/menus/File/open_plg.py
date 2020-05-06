import wx,os,sys
from skimage.io import imread

from imagepy.core.util import fileio
from imagepy.core.manager import ReaderManager

class OpenFile(fileio.Reader):
    title = 'Open'

    def load(self):
        self.filt = sorted(ReaderManager.get(tag=None))
        return True


plgs = [OpenFile]
    
