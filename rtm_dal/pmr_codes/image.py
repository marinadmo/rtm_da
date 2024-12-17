
import os
import numpy as np
from PIL import Image as PILImage
from matplotlib import pylab as plt
import cmocean

def get_int(f):
    try:
        return int(f)
    except TypeError:
        return f.astype('int')


class BaseImage(object):

    def __init__(self, img, step):
        self.step  = step
        self.img   = img
        self.shape = img.shape

    def plot(self, ax=None, cmap=cmocean.cm.ice, vmin=0, vmax=1):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        if vmin == 'min':
            vmin = self.img.min()
        if vmax == 'max':
            vmax = self.img.max()
        im = ax.imshow(self.img, interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        return im

    def __sub__(self, other):
        if self.shape != other.shape:
            raise ValueError("Cannot compute similarity between two images (not the same shape)")
        if self.step  != other.step:
            raise ValueError("Cannot compute similarity between two images (not the same spacing)")

        return BaseImage(self.img - other.img, self.step)

    def min(self):
        return self.img.min()

    def max(self):
        return self.img.max()    

    def similarity(self, other, metric='var'):
        diff = self - other
        if metric == 'var':
            return diff.img.var()
        elif metric == 'msd':
            return ((diff.img)**2).mean()
        else:
            raise ValueError("Do not know about this image similarity metric: {}".format(metric))

class Image(BaseImage):
    # Image and its sub-classes are for high-resolution images that can be convoluted with an
    #    antenna pattern.

    def __init__(self, img, step,):

        # sanity check: the two elements of the shape must be odd numbers, to allow a
        #   central pixel to be at 0,0
        if (img.shape[0]%2 == 0) or ((img.shape[1]%2 == 0)):
            raise ValueError("image shape must be odd numbers, got {}".format(img.shape))
        
        super().__init__(img, step)

        # compute xs/ys (in km) with a central pixel at (0,0)
        self.xs   = np.arange(-self.step*(self.shape[1]-1)//2,+self.step*((self.shape[1]-1)//2 + 1),self.step)
        self.ys   = np.arange(-self.step*(self.shape[0]-1)//2,+self.step*((self.shape[0]-1)//2 + 1),self.step)

    def _get_xy_conv(self, spacing, in_img_pix=True, centered=False, stride=(1,1)):
        # compute the locations for convolution with ifov (at a given spacing) 
        if centered:
            # have a xy pos close to the center of the image
            imgc_yi = self.shape[0]//2
            imgc_xi = self.shape[1]//2
            hx_conv = np.arange(self.xs[imgc_xi+1],self.xs[-1]+1,spacing)[::stride[1]]
            hy_conv = np.arange(self.ys[imgc_yi+1],self.ys[-1]+1,spacing)[::stride[0]]
            x_conv  = np.append((-1.*hx_conv[1:])[::-1],hx_conv)
            y_conv  = np.append((-1.*hy_conv[1:])[::-1],hy_conv)
        else:
            step_conv = spacing//self.step
            x_conv = np.arange(self.xs[0]+spacing*0.5,self.xs[-1]-spacing*0.5+1,spacing)[::stride[1]]
            y_conv = np.arange(self.ys[0]+spacing*0.5,self.ys[-1]-spacing*0.5+1,spacing)[::stride[0]]
        
        if in_img_pix:
            x_conv, y_conv = self.get_pix(x_conv,y_conv)
            
        return x_conv, y_conv

    def get_pix(self, x_km, y_km, integer=True):
        xpix = x_km/self.step + self.shape[1]//2
        ypix = y_km/self.step + self.shape[0]//2
        if integer:
            xpix = get_int(xpix)
            ypix = get_int(ypix)
        return xpix, ypix

    def _scale_img(self,vmin,vmax):
        if vmin == 'min':
            vmin = self.img.min()
        elif vmin == '1p':
            vmin = np.percentile(self.img,(1,))[0]
        
        if vmax == 'max':
            vmax = self.img.max()
        elif vmax == '99p':
            vmax = np.percentile(self.img,(99,))[0]

        self.img = (self.img - vmin) / (vmax - vmin)
        self.img[self.img > 1] = 1
        self.img[self.img < 0] = 0

    

    def simulate_tb_images(self, ow, cice, kind_cice='mean'):

        # check ow and cice have the same channels
        if sorted(ow.channels) != sorted(cice.channels):
            raise ValueError("The OW and CICE signature objects do not have the same channels.")

        # prepare cice signature.
        if kind_cice == 'mean' or kind_cice == 'avg':
            cice_tp = cice.tp
        elif kind_cice == 'fyi':
            cice_tp = cice.fyi_tp
        elif kind_cice == 'myi':
            cice_tp = cice.myi_tp
        else:
            raise ValueError("Unsupported value for kind_cice= (got {})".format(kind_cice))

        # simulate the various tb channels by scaling the image by the 
        tb_imgs = dict()
        for ich, ch in enumerate(ow.channels):
            tb_imgs[ch] = Image(self.img.copy(), self.step)
            tb_imgs[ch].img *= ( cice_tp[ich] - ow.tp[ich] )
            tb_imgs[ch].img += ow.tp[ich]

        return tb_imgs

class SynthImage(Image):
    """ a square image covering [-size;+size] km with steps km (e.g. [-50;+50]km in steps of 0.1km)"""
    def __init__(self, kind, step, shape, vmin=0, vmax=1):
        
        if not hasattr(shape,"__len__"):
            shape = (int(shape),int(shape))
        elif len(shape) == 2:
            shape = (int(shape[0]),int(shape[1]))
        else:
            raise ValueError("shape can be a scalar or tuple of length 2")

        img = np.ones(shape)
        if kind == 'vertical':
            img[:,shape[1]//2:] = 0.
        elif kind == 'horizontal':
            img[shape[0]//2:,:] = 0.
        elif kind == 'box':
            img[shape[0]//4:3*shape[0]//4+1,shape[1]//4:3*shape[1]//4+1] = 0.
        else:
            raise ValuError("Know only kinds 'horizontal', 'vertical', and 'box'. ")

        super().__init__(img, step)
        self._scale_img(vmin,vmax)


class ModisImage(Image):
    def __init__(self, scene='antarctic', vmin=0, vmax=1,imgdir='.',binary=True):
        self.step = 0.25

        if scene == 'antarctic' or scene == 'antarctic-detail':
            im = PILImage.open( os.path.join(imgdir,'Antarctica.A2008055.0330.250m.jpg') ).convert('L')
            im = np.asarray(im)[0:2201,0:3201] / 255.
            if scene == 'antarctic-detail':
                lside = 351
                ydet, xdet = (400,1500)
                im = im[ydet:ydet+lside,xdet:xdet+lside]

        if binary:
            im = np.where(im>0.5,1.,0.)

        super().__init__(im,self.step)
        self._scale_img(vmin, vmax)

class S2Image(Image):
    def __init__(self, scene='barents20180503', vmin=0, vmax=1,imgdir='.',binary=True):
        self.step = 0.06 #km

        if scene == 'barents20180503':
            im = PILImage.open( os.path.join(imgdir,'Barents.20180503.60m.tif')).convert('L')
            im = np.asarray(im)[:-1,:] / 255.
        elif scene == 'barents20180502':
            im = PILImage.open( os.path.join(imgdir,'Barents.20180502.60m.tif')).convert('L')
            im = np.asarray(im)[:-1,:-1] / 255.

        if binary:
            im = np.where(im>0.5,1.,0.)    

        super().__init__(im,self.step)
        self._scale_img(vmin, vmax)


class ImageCombination(Image):

    def __init__(self,images,weights):

        # some checks before we allow this as a valid combination of channels
        # check same number of weigths and bells
        if len(images) != len(weights):
            raise ValueError("images and weights should have the same length")
        # check weights add to unity
        if sum(weights) != 1.:
            raise ValueError("sum of weights should be 1! (got {})".format(sum(weights)))
        # check all bells are on the same xs, ys
        if len(images) > 1:
            for iimage, image in enumerate(images[1:]):
                if image.shape != images[0].shape:
                    raise ValueError("all images should have the same shape!")
                if image.step != images[0].step:
                    raise ValueError("all images should have the same step size!")

        combined_image = np.zeros_like(images[0].img)
        for iimage, image in enumerate(images):
            combined_image += weights[iimage] * image.img

        super().__init__(combined_image,images[0].step)

        self.combi_images  = tuple(images)
        self.combi_weights = tuple(weights)

