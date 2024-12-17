
import numpy as np
from itertools import combinations
#import matplotlib._cntr as cntr


#from . import image
import image

# global definitions
tb_dict = {'tb01':'L','tb06':'C','tb10':'X','tb19':'Ku','tb37':'Ka','tb90':'W'}
rev_tb_dict = {v:k for k,v in tb_dict.items()}

# classes
class Bell(object):

    def __init__(self, xs, ys, bell):

        bell = np.asarray(bell)
        #check that bell has 1 as highest value
        if bell.max() > 1.:
            raise ValueError("the bell must peak at below 1! (got {})".format(bell.max()))

        #check that bell, xs, and ys have the same shape
        if bell.shape != xs.shape or bell.shape != ys.shape:
            raise ValueError("the bell shape must be the same as that of xs and ys")

        self.xs   = xs
        self.ys   = ys
        self.bell = bell
        self.shape = bell.shape
        self.norm = bell.sum()

    #def get_halfpower_contour(self):
    #    level = 0.5
    #    c = cntr.Cntr(self.xs, self.ys, self.bell)
    #    nlist = c.trace(level, level, 0)
    #    return zip(*(nlist[0]))    # x,y coords of contour points.

    def get_halfpower_diameters(self):
        cx, cy = self.get_halfpower_contour()
        cx = np.asarray(cx)
        cy = np.asarray(cy)
        # compute arithmetic mean for all the countour points (average by integration)
        vs = np.row_stack((cx,cy))
        ns = np.linalg.norm(vs,axis=0)
        avg_diam_1 = 2. * ns.mean()
        # compute arithmetic mean for a and b.
        a = cx.max() - cx.min()
        b = cy.max() - cy.min()
        avg_diam_2 = 0.5 * ( a + b )
        return (a, b), avg_diam_1, avg_diam_2

    def plot(self,ax,half=True,**kwargs):
        try:
            ax.plot_surface(self.xs, self.ys, self.bell,**kwargs)
        except:
            if not half:
                ax.contour(self.xs, self.ys, self.bell,**kwargs)
            else:
                cx, cy = self.get_halfpower_contour()
                ax.plot(cx,cy,**kwargs)

    def cropped_bell(self):
        mask = (self.bell == 0)
        si, se = np.where(~mask)
        c_bell = self.bell[si.min():si.max() + 1, se.min():se.max() + 1]
        c_xs   = self.xs[si.min():si.max() + 1, se.min():se.max() + 1]
        c_ys   = self.ys[si.min():si.max() + 1, se.min():se.max() + 1]
        return Bell(c_xs, c_ys, c_bell)

    def _check_ifov_vs_img(self, img):

        # check step of bell in inline with step of image
        step_x = self.xs[0,1] - self.xs[0,0]
        step_y = self.ys[1,0] - self.ys[0,0]
        if abs(step_x-img.step) > 1e-3 or abs(step_y-img.step) > 1e-3:
            raise ValueError("Cannot convolute Bell ({}) with image ({}) because steps not equal".format(step_x,img.step))

    def plot_bells_on_image(self, img, spacing, ax, stride=1, centered=False, filled=False, **plot_kwargs):

        # check we can use this bell and this image together
        self._check_ifov_vs_img(img)

        # handle the stride argument
        only_one = False
        if stride == 'one':
            only_one = True
            stride = (1,1)

        if hasattr(stride,"__len__"):
            if len(stride) == 2:
                stridey, stridex = int(stride[0]),int(stride[1])
            else:
                raise ValueError("the stride= parameter must be a scalar or a (stride_y, stride_x) tuple")
        else:
            stridey = int(stride)
            stridex = stridey
        
        # locations where to convolve image with ifov
        x_conv, y_conv = img._get_xy_conv(spacing,centered=centered,in_img_pix=False, stride=(stridex,stridey))

        if only_one:
            if centered:
                x_indx = len(x_conv)//2
                y_indx = len(y_conv)//2
            else:
                x_indx = 0
                y_indx = 0
            x_conv = np.array([x_conv[x_indx],])
            y_conv = np.array([y_conv[y_indx],])

        # the half-power contour of the fov
        cx, cy = self.get_halfpower_contour()
        cx = np.asarray(cx)
        cy = np.asarray(cy)

        # go through the locations of the convolved image
        for ix in range(0, len(x_conv), ):
            for iy in range(0, len(y_conv), ):
                xpix,ypix = x_conv[ix],y_conv[iy]

                ellx, elly = img.get_pix(cx + xpix, cy + ypix, integer=False)

                # plot the contour at the correction location
                if filled:
                    ax.fill(ellx, elly, **plot_kwargs, clip_on=True)
                ax.plot(ellx, elly, **plot_kwargs, clip_on=True)



    def image_convolution(self,img,spacing,centered=False,crop=False):

        # we use a cropped version of the bell to speed convolution
        c_bell = self.cropped_bell()

        # check we can use this bell and this image together
        c_bell._check_ifov_vs_img(img)

        # locations where to convolve image with ifov
        x_conv, y_conv = img._get_xy_conv(spacing,centered=centered,in_img_pix=True)

        # output convolved image
        conv_img = np.ma.zeros((y_conv.size, x_conv.size, ))

        # go through the locations of the convolved image
        for ix in range(len(x_conv)):
            for iy in range(len(y_conv)):
                
                xpix,ypix = x_conv[ix],y_conv[iy]
                #print(xpix,ypix)
                # extract sub-image
                x_sub_img = (xpix-c_bell.shape[1]//2,xpix+c_bell.shape[1]//2+1)
                y_sub_img = (ypix-c_bell.shape[0]//2,ypix+c_bell.shape[0]//2+1)

                if x_sub_img[0] < 0 or y_sub_img[0] < 0 or x_sub_img[1] > img.shape[1] or y_sub_img[1] > img.shape[0]:
                    #print('mask ', conv_img.shape,' at ',iy,ix)
                    conv_img[iy,ix] = np.ma.masked
                else:
                    sub_img = img.img[y_sub_img[0]:y_sub_img[1],x_sub_img[0]:x_sub_img[1]]
                    conv_sub_img = ( sub_img * c_bell.bell ).sum() / c_bell.norm
                    conv_img[iy,ix] = conv_sub_img

        if crop:
            # crop the image for masked border
            si, se = np.where(~conv_img.mask)
            conv_img = conv_img[si.min():si.max() + 1, se.min():se.max() + 1]

        # initialize and return an image object
        return image.BaseImage(conv_img, spacing)

class SquareFOV(Bell):

    def __init__(self, side, xs=None, ys=None):
        self.side = side
        self.bell = None
        if xs is not None and ys is not None:
            self.compute_bell(xs,ys,)

    def compute_bell(self,step,antialiasing=False):
        # store in object
        ls_1, xs_1, ys_1, gbell_1 = self._compute_bell(step, )
        if antialiasing:
            raise NotImplementedError("antialiasing not yet implemented for SquareFOV objects")
        else:
            gbell = gbell_1

        super().__init__(xs_1,ys_1,gbell)

    def _compute_bell(self, step, ):

        hside = self.side/2

        # compute xs and ys that cover the square
        ls = np.arange(0, 1.2*(hside+step/2), step)
        ls = np.append(-ls[::-1],ls[1:])
        xs,ys = np.meshgrid(ls,ls)

        # compute the tabulated square bell
        bell = np.ones_like(xs)
        bell[abs(xs) > hside] = 0.
        bell[abs(ys) > hside] = 0.

        return ls, xs, ys, bell

def get_ab(iFov_a, a_over_b=(35./62.)):
    return (iFov_a, iFov_a / a_over_b)

def _gauss_1d(t, sigt, mu=0., norm=False):
    bell = np.exp(-0.5*((t-mu)/sigt)**2)
    if norm:
        bell /= (sigt * np.sqrt(2*np.pi))
    return  bell

def _gauss_2d(x,y,sigx,sigy,mux=0,muy=0,norm=False):
    return _gauss_1d(x,sigx,mu=mux,norm=norm) * \
        _gauss_1d(y,sigy,mu=muy,norm=norm)

def _bup_1d(ls, bup):
    """
       Take a 1d array with equally spaced values, and interleave 'bup' values so
          that block-based operations can be performed.
    """
    if bup % 2 == 0:
        raise ValueError("only odd bup numbers are supported")

    if np.diff(ls).max() - np.diff(ls).min() > 1.e-3:
        raise ValueError("the input array does not seem equally spaced")
    step = np.diff(ls).mean()

    nls = np.empty((ls.size,bup))
    nls[:,0] = ls
    nstep = step / bup
    for i in range(1,bup,):
        nls[:,i]   = ls + ((i+1)//2)*nstep*(-1)**i
       
    nls = nls.reshape(-1)
    nls.sort()
    
    return nls

def _blockshaped(arr, nrows, ncols):
    """
    from: http://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    if (h % nrows != 0) or (w % ncols != 0):
        raise ValueError("unable to prepare a blockshaped view as dimensions do not match")
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

def _eFoV_integration_length(spacing, a_diam):
    # compute the integration length needed for
    #   having at least contiguous eFoVs
    return spacing - a_diam

class ChannelEFOV(Bell):

    @staticmethod
    def _gauss_fov(x,y,sigx,sigy,mux=0.,muy=0.,trunc=3.5):
        gauss = _gauss_2d(x, y, sigx, sigy, mux=mux, muy=muy)
        if trunc>0: 
            trunc = (abs(x-mux) < trunc*sigx) * (abs(y-muy) < trunc*sigy)
            trunc_gauss  = gauss * trunc
            return trunc_gauss
        else:
            return gauss

    @staticmethod
    def _sigmas(iFoV_a,iFoV_b):
        #https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        #   iFoV_a and iFoV_b are FWHM: Full Width at Half Maximum
        return tuple(np.array([iFoV_a,iFoV_b]) / (2. * np.sqrt(2. * np.log(2))))

    def _compute_bell(self, step, trunc=3.5, ):

        # compute xs and ys that cover the fov diameters
        (ifov_a, ifov_b) = self.iFoV
        Smax = max(ifov_a,ifov_b)/2. * (trunc + 1.)
        ls = np.arange(0, Smax+step/2, step)
        ls = np.append(-ls[::-1],ls[1:])
        xs,ys = np.meshgrid(ls,ls)

        # compute the tabulated bell

        # we use a loop to implement the integration during the eFoV
        if self.int_half_length > 0.:
            int_locs = np.linspace(-self.int_half_length,self.int_half_length,num=11)
        else:
            int_locs = np.array([0.,])

        # normalized sum over the integration period
        gauss_bell = np.zeros_like(xs)
        for int_step in int_locs:
            gauss_bell += self._gauss_fov(xs, ys, *self.sigmas, mux=int_step, trunc=False)
        gauss_bell /= len(int_locs)

        # apply truncation
        if trunc:
            sigx,sigy = self.sigmas
            trunc = (abs(xs) < trunc*sigx) * (abs(ys) < trunc*sigy)
            gauss_bell *= trunc

        # finally normalize to have 1 at 0,0
        gauss_bell /= gauss_bell[len(ls)//2,len(ls)//2]

        return ls, xs, ys, gauss_bell

    def compute_bell(self,step,trunc=3.5,antialiasing=False,crop=False):
        # store in object
        ls_1, xs_1,ys_1, gbell_1 = self._compute_bell(step, trunc=trunc)
        if antialiasing:
            bup = 5 #TODO: dynamically guess a good-enough bup factor
            nls = _bup_1d(ls_1,bup)
            nxs, nys = np.meshgrid(nls,nls)
            gauss_bell_bup = self._gauss_fov(nxs, nys, *self.sigmas, trunc=trunc)
            gbell = _blockshaped(gauss_bell_bup,bup,bup).mean(axis=1).mean(axis=1).reshape(*gbell_1.shape)
            gbell[gbell_1==0] = 0
        else:
            gbell = gbell_1

        super().__init__(xs_1,ys_1,gbell)
        if crop:
            cbell = self.cropped_bell()
            super().__init__(cbell.xs,cbell.ys,cbell.bell)

    def __init__(self, name, iFoV_a, iFoV_b, int_half_length=0., xs=None, ys=None):

        self.name = name
        self.iFoV = (iFoV_a, iFoV_b)
        self.iFoV_mean = 0.5 * (iFoV_a + iFoV_b)
        self.sigmas = self._sigmas(iFoV_a,iFoV_b)
        self.int_half_length = int_half_length
        self.bell = None
        if xs is not None and ys is not None:
            self.compute_bell(xs,ys,trunc=3.5)

    def __str__(self):
        return "{} ({:.0f}x{:.0f}km) s=({:.0f}x{:.0f}km)".format(self.name, *self.iFoV,
                                                                 *np.array(self.sigmas)*2)

class BellCombination(Bell):

    def __init__(self,bells,weights):
        # some checks before we allow this as a valid combination of channels
        # check same number of weigths and bells
        if len(bells) != len(weights):
            raise ValueError("bells and weights should have the same length")
        # check weights add to unity
        if sum(weights) != 1.:
            raise ValueError("sum of weights should be 1! (got {})".format(sum(weights)))
        # check all bells are on the same xs, ys
        if len(bells) > 1:
            for ibell, bell in enumerate(bells[1:]):
                if bell.xs.shape != bells[0].xs.shape or bell.ys.shape != bells[0].ys.shape:
                        raise ValueError("all the bells should be defined on the same xs and ys!")

        combined_bell = np.zeros_like(bells[0].bell)
        for ibell, bell in enumerate(bells):
            combined_bell += weights[ibell] * bell.bell

        super().__init__(bells[0].xs, bells[0].ys,combined_bell)
        self.combi_bells   = tuple(bells)
        self.combi_weights = tuple(weights)


class PMRChannel(object):

    def __init__(self, band, freq, oza, sampling, bell, pols=('h','v',), nedt=0.):
        self.band = band
        self.freq = freq
        self.oza  = oza
        self.sampling = sampling
        self.bell = bell
        self.pols = pols
        self.nedt = nedt

    def get_tb_shortname(self,pol=None):
        one_pol = False
        if pol is None:
            pol = self.pols
        else:
            pol = (pol,)
            one_pol = True

        ret = []
        for p in pol:
            ret.append(rev_tb_dict[self.band]+p)

        if one_pol:
            ret = ret[0]

        return ret

    def get_tb_longname(self,pol=None):
        one_pol = False
        if pol is None:
            pol = self.pols
        else:
            pol = (pol,)
            one_pol = True

        ret = []
        for p in pol:
            ret.append('Tb {:.1f} GHz {}-pol'.format(self.freq,p.upper()))

        if one_pol:
            ret = ret[0]

        return ret



class PMR(object):

    #@staticmethod
    #def get_band(tb):
    #    try:
    #        return tb_dict[tb[:4]]
    #    except KeyError as ke:
    #        raise ValueError('Unknown Tb shortname {}'.format(ke,))
    #
    #def get_freq(self,tb):
    #    try:
    #        band = self.get_band(tb)
    #        return getattr(self,band).freq
    #    except AttributeError as ae:
    #        raise ValueError('PMR instrument {} does not have band {}'.format(self,band))

    def get_PMRChannel(self,w):
        if w.startswith('tb'):
            return getattr(self,tb_dict[w[:4]])
        else:
            return getattr(self,w)

    def get_tb_longname(self,w,pols=None):
        if len(w) == 5:
            return getattr(self,tb_dict[w[:4]]).get_tb_longname(w[4])
        elif len(w) == 4:
            return getattr(self,tb_dict[w[:4]]).get_tb_longname()
        else:
            return getattr(self,w).get_tb_longname()

    def get_tb_shortname(self,w,):
        if len(w) == 5:
            return getattr(self,tb_dict[w[:4]]).get_tb_shortname(w[4])
        elif len(w) == 4:
            return getattr(self,tb_dict[w[:4]]).get_tb_shortname()
        else:
            return getattr(self,w).get_tb_shortname()

    def set_channels(self,):
        l = [ bnd.get_tb_shortname() for bnd in self.bands]
        self.channels = [item for sublist in l for item in sublist]

    def set_freqs(self,):
        self.freqs = [bnd.freq for bnd in self.bands]

    def set_bandn(self,):
        self.bandn = [bnd.band for bnd in self.bands]

    def set_fields(self,):
        self.set_freqs()
        self.set_bandn()
        self.set_channels()

    def get_nedt_C(self, channels=None):
        if channels is None:
            channels = self.channels
        nedt_C = np.identity(len(channels))
        for ich, ch in enumerate(channels):
            pmrch = self.get_PMRChannel(ch,)
            nedt_C[ich,ich] = pmrch.nedt**2

        return nedt_C

    def get_channel_combinations(self,n=1):
        if n < 1 or n > len(self.channels):
            raise ValueError("Cannot ask channel combinations with n={}for this instrument".format(n,))
        return list(combinations(self.channels,n))

    def __str__(self,):
        return self.__class__.__name__

    def simulate_image(self, tb_imgs, centered=False):
        # convolute a series of tb_imgs with the channels

        # check tb_imgs:
        #   o if a dict(), expect it has all needed channels
        #   o if not, duplicate in a dict()
        try:
            ch_imgs = tb_imgs.keys()
            for ch in self.channels:
                if ch not in ch_imgs:
                    raise ValueError("Tb channel {} is missing from tb_imgs".format(ch))
        except AttributeError:
            dict_tb_imgs = dict()
            for ch in self.channels:
                dict_tb_imgs[ch] = tb_imgs
            tb_imgs = dict_tb_imgs

        for ch in self.channels:
            if not isinstance(tb_imgs[ch], image.Image):
                raise ValueError("parameter tb_imgs does not result in Image objects")

        # simulate the image, one channel at a time
        simul_imgs = dict()
        for ch in self.channels:
            pmrch = self.get_PMRChannel(ch,)
            ifov = pmrch.bell
            # compute the bell shape
            ifov.compute_bell(step=tb_imgs[ch].step,trunc=2.5)
            # simulate and store image as observed by this iFoV
            simul_imgs[ch] = ifov.image_convolution(tb_imgs[ch],pmrch.sampling, centered=centered)

        return simul_imgs

class CIMR(PMR):

    def __init__(self, force_ifov=True):
        # define individual channels
        cimr_sampling = 5

        if force_ifov:
            int_hl = 0.
        else:
            # have the Ka-band channels contiguous
            int_hl = _eFoV_integration_length(cimr_sampling, 3.)

        cimr_oza = 55.
        self.C  = PMRChannel('C',  6.93,cimr_oza,cimr_sampling,
            ChannelEFOV('cimr_C', *get_ab(11.,), int_half_length=int_hl),nedt=0.2)
        self.X  = PMRChannel('X', 10.54,cimr_oza,cimr_sampling,
            ChannelEFOV('cimr_X', *get_ab( 7.,), int_half_length=int_hl),nedt=0.3)
        self.Ku = PMRChannel('Ku',18.70,cimr_oza,cimr_sampling,
            ChannelEFOV('cimr_Ku',*get_ab( 4.,), int_half_length=int_hl),nedt=0.3)
        self.Ka = PMRChannel('Ka',36.50,cimr_oza,cimr_sampling,
            ChannelEFOV('cimr_Ka',*get_ab( 3.,), int_half_length=int_hl),nedt=0.7)
        # collect them in a tuple
        self.bands = (self.C,self.X,self.Ku,self.Ka,)
        # extract channel information for quick referencing
        self.set_fields()

class TEST(PMR):

    def __init__(self,):
        # define individual channels
        cimr_sampling = 5
        cimr_oza = 55.
        self.C  = PMRChannel('C',  6.93,cimr_oza,cimr_sampling,ChannelEFOV('cimr_C', *get_ab( 4.,)))
        self.X  = PMRChannel('X', 10.54,cimr_oza,cimr_sampling,ChannelEFOV('cimr_X', *get_ab( 4.,)))
        self.Ku = PMRChannel('Ku',18.70,cimr_oza,cimr_sampling,ChannelEFOV('cimr_Ku',*get_ab( 4.,)))
        self.Ka = PMRChannel('Ka',36.50,cimr_oza,cimr_sampling,ChannelEFOV('cimr_Ka',*get_ab( 3.,)))
        # collect them in a tuple
        self.bands = (self.C,self.X,self.Ku,self.Ka,)
        # extract channel information for quick referencing
        self.set_fields()

class AMSR2(PMR):

    def __init__(self, force_ifov=True):

        samp_W = 5.
        samp_others = samp_W * 2.

        if force_ifov:
            int_hl = 0.
        else:
            # have the W-band channels contiguous
            int_hl = _eFoV_integration_length(samp_W, 3.)

        # define individual channels
        amsr2_oza = 55.
        a_over_b = 0.6
        self.C  = PMRChannel('C',  6.93,amsr2_oza,samp_others,ChannelEFOV('amsr2_C',
            *get_ab(35.,a_over_b=a_over_b),int_half_length=2.*int_hl))
        self.X  = PMRChannel('X', 10.54,amsr2_oza,samp_others,ChannelEFOV('amsr2_X',
            *get_ab(24.,a_over_b=a_over_b),int_half_length=2.*int_hl))
        self.Ku = PMRChannel('Ku',18.70,amsr2_oza,samp_others,ChannelEFOV('amsr2_Ku',
            *get_ab(14.,a_over_b=a_over_b),int_half_length=2.*int_hl))
        self.Ka = PMRChannel('Ka',36.50,amsr2_oza,samp_others,ChannelEFOV('amsr2_Ka',
            *get_ab( 7.,a_over_b=a_over_b),int_half_length=2.*int_hl))
        self.W  = PMRChannel('W', 89.00,amsr2_oza,samp_W, ChannelEFOV('amsr2_W', 
            *get_ab( 3.,a_over_b=a_over_b),int_half_length=int_hl))
        # collect them in a tuple
        self.bands = (self.C,self.X,self.Ku,self.Ka,self.W)
        # extract channel information for quick referencing
        self.set_fields()

class KuKa(PMR):

    def __init__(self,):
        # define individual channels
        KuKa_sampling = 25.
        KuKa_oza = 55.
        self.Ku = PMRChannel('Ku',18.70,KuKa_oza,KuKa_sampling,ChannelEFOV('Ku',*get_ab( 25.,)))
        self.Ka = PMRChannel('Ka',36.50,KuKa_oza,KuKa_sampling,ChannelEFOV('Ka',*get_ab( 25.,)))
        # collect them in a tuple
        self.bands = (self.Ku,self.Ka,)
        # extract channel information for quick referencing
        self.set_fields()

def get_PMR(instr_type):
    instr_type = instr_type.upper()
    if instr_type == 'CIMR':
        return CIMR()
    if instr_type == 'TEST':
        return TEST()
    elif instr_type == 'AMSR2':
        return AMSR2()
    else:
        raise NotImplementedError('Unknown instrument type {}'.format(instr_type,))

