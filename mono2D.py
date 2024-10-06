import torch
import numpy as np
import torch.nn as nn

from torch.nn.modules import Module

# Adopted from:
# Copyright (c) 1996-2009 Peter Kovesi
# School of Computer Science & Software Engineering
# The University of Western Australia
# pk at csse uwa edu au
# http://www.csse.uwa.edu.au/
# 
# Permission is hereby  granted, free of charge, to any  person obtaining a copy
# of this software and associated  documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# The software is provided "as is", without warranty of any kind.
#
#------------------------------------------------------------------------
#
# Modified by A. BELAID, 2013
#
# Description: This code implements a part of the paper:
# Ahror BELAID and Djamal BOUKERROUI. "A new generalised alpha scale 
# spaces quadrature filters." Pattern Recognition 47.10 (2014): 3209-3224.
#
# Ahror BELAID and Djamal BOUKERROUI. "Alpha scale space filters for 
# phase based edge detection in ultrasound images." ISBI (2014): 1247-1250.
#
# Copyright (c), Heudiasyc laboratory, Compiègne, France.
#
#------------------------------------------------------------------------

class Mono2D(Module):
    def __init__(
        self, nscale: int = 1, sigmaonf: float = None, wls: list = None, trainable: bool = True,
        return_phase: bool = True, return_phase_asym: bool = False, return_ori: bool = False,
        return_input: bool = False
        ):
        super(Mono2D, self).__init__()

        # Hyperparameters - these can be tuned
        self.return_phase = return_phase
        self.return_phase_asym = return_phase_asym
        self.return_input = return_input
        self.return_ori = return_ori
        self.trainable = True if trainable is None else trainable

        assert nscale > 0
        self.nscale = nn.Parameter(torch.tensor(nscale, dtype=torch.int), requires_grad=False)

        # Fixed parameters
        # According to Nyquist theorem, the smallest wavelength should be 2 pixels to avoid aliasing.
        # Pick 3 pixels to be totally certain.
        self.min_wl = 3.0
        # Heuristically, at least for knee cartilage, the max wavelength won't reach 128 pixels.
        # This is a temporary fix that will be investigated in future work
        self.max_wl = 128.0

        # Learned parameters
        self.wls = nn.Parameter(self.initialize_wls(wls), requires_grad=self.trainable)
        self.sigmaonf = nn.Parameter(self.initialize_sigmaonf(sigmaonf), requires_grad=self.trainable)
        
        # Parameters below are used to construct the largest possible low-pass filter
        # that quickly falls to zero at the boundaries. Cut-off frequency (normalized)
        # should be between 0 and 0.5 according to Nyquist theorem.
        # The larger the value of g, the sharper the transition to zero.
        self.cut_off = 0.4
        self.g = 10
        # This implementation does not apply any noise compensation, hence, noise compensation
        # threshold is set to zero.
        self.T = 0
        # Set a small value used throughout the layer to avoid division by zero
        self.episilon = 0.0001


    def forward(self, x):
        _, _, rows, cols = x.size()
        # Transform the input image to frequency domain
        IM = torch.fft.fft2(x).to(self.get_device())

        # Obtain quadrature filters and low-pass filter
        H, lgf = self.get_filters(rows, cols)

        # Bandpassed image in the frequency domain
        IMF = IM * lgf

        # Bandpassed image in the spatial domain
        f = torch.fft.ifft2(IMF).real

        # Bandpassed monogenic filtering, real part of h contains convolution result with h1, 
        # imaginary part contains convolution result with h2
        h = torch.fft.ifft2(IMF * H)
        h1 = h.real
        h2 = h.imag
        h_Amp2 = h1 ** 2 + h2 ** 2          # Amplitude of the bandpassed monogenic signal
        An = torch.sqrt(f ** 2 + h_Amp2)    # Magnitude of Energy (Amplitude)

        # Compute the phase asymmetry (odd - even)
        symmetry_energy = torch.sqrt(h_Amp2) - torch.abs(f)
        
        # Sum all responses across all scales
        f = torch.sum(f, dim=1)
        h1 = torch.sum(h1, dim=1)
        h2 = torch.sum(h2, dim=1)
        h_Amp2 = torch.sum(h_Amp2, dim=1)
        An = torch.sum(An, dim=1)
        symmetry_energy = torch.sum(symmetry_energy, dim=1)

        # Compute the phase asymmetry
        phase_asym = torch.clamp(symmetry_energy - self.T, min=0) / (An + self.episilon)

        # Orientation - this varies +/- pi
        ori = torch.atan2(-h2,h1)
        ori = self.scale_max_min(ori)

        # Feature type - a phase angle +/- pi.
        ft = torch.atan(f/torch.sqrt(h1 ** 2 + h2 ** 2))
        ft = self.scale_max_min(ft)

        out = []
        if self.return_input:
            out.append(x)
        if self.return_phase:
            out.append(ft)
        if self.return_ori:
            out.append(ori)
        if self.return_phase_asym:
            out.append(phase_asym)

        return torch.stack(out, dim=1)    

    def get_filters(self, rows, cols):
        u1, u2, radius = self.mesh_range((rows, cols))
        # Get rid of the 0 radius value in the middle (at top left corner after
        # fftshifting) so that taking the log of the radius, or dividing by the
        # radius, will not cause trouble.
        radius[0,0] = 1.

        # Construct the monogenic filters in the frequency domain.  The two
        # filters would normally be constructed as follows
        #    H1 = i*u1./radius; 
        #    H2 = i*u2./radius;
        # However the two filters can be packed together as a complex valued
        # matrix, one in the real part and one in the imaginary part.
        # When the convolution is performed via the fft the real part of the result
        # will correspond to the convolution with H1 and the imaginary part with H2.
        H = (1j*u1 - u2) / radius

        # Construct a low-pass filter that is as large as possible, yet falls
        # away to zero at the boundaries.  All filters are multiplied by
        # this to ensure no extra frequencies at the 'corners' of the FFT are
        # incorporated as this can upset the normalisation process when
        # calculating phase symmetry
        lp = self.lowpassfilter([rows, cols], self.cut_off, self.g)
        
        # Compute the log-Gabor filter
        lgf = self.compute_logGabor(radius)
        # Apply low-pass filter
        lgf = lgf * lp
        # Set the value at the 0 frequency point of the filter back to zero (undo the radius fudge).
        lgf[0,0] = 0

        return H, lgf

            
    def compute_logGabor(self, radius):
        # Obtain the different scales wavelengths
        wls = self.get_wls()
        # Obtain the center frequencies
        fo = 1.0 / wls
        # Reshape fo to be broadcastable with radius
        fo = fo.view(-1, 1, 1)
        # The parameter sigmaonf is in the range -inf to inf. Rescale it to 0-1
        sigmaonf = torch.nn.functional.sigmoid(self.sigmaonf)
        # Construct the filter
        filter = torch.exp((-(torch.log(radius/fo)) ** 2) / (2 * torch.log(sigmaonf) ** 2))
        return filter.to(self.get_device())


    def lowpassfilter(self, sze, cutoff, n):
        # LOWPASSFILTER - Constructs a low-pass butterworth filter.

        # usage: f = lowpassfilter(sze, cutoff, n)

        # where: sze    is a two element vector specifying the size of filter 
        #             to construct [rows cols].
        #     cutoff is the cutoff frequency of the filter 0 - 0.5
        #     n      is the order of the filter, the higher n is the sharper
        #             the transition is. (n must be an integer >= 1).
        #             Note that n is doubled so that it is always an even integer.

        #                     1
        #     f =    --------------------
        #                             2n
        #             1.0 + (w/cutoff)

        # The frequency origin of the returned filter is at the corners.

        # See also: HIGHPASSFILTER, HIGHBOOSTFILTER, BANDPASSFILTER


        # Copyright (c) 1999 Peter Kovesi
        # School of Computer Science & Software Engineering
        # The University of Western Australia
        # http://www.csse.uwa.edu.au/

        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, subject to the following conditions:

        # The above copyright notice and this permission notice shall be included in 
        # all copies or substantial portions of the Software.

        # The Software is provided "as is", without warranty of any kind.

        # October 1999
        # August  2005 - Fixed up frequency ranges for odd and even sized filters
        #             (previous code was a bit approximate)
        if cutoff < 0 or cutoff > 0.5:
            raise('cutoff frequency must be between 0 and 0.5')
            
        if n % 1 != 0 or n < 1:
            raise('n must be an integer >= 1')
        
        if len(sze) == 1:
            sze = (sze, sze)
        else:
            rows = sze[0]
            cols = sze[1]

        _, _, radius = self.mesh_range((rows, cols))
        # Compute the filter
        f = ( 1.0 / (1.0 + (radius / cutoff) ** (2*n)) )

        return f
    

    def mesh_range(self, size):
        rows, cols = size
        # Set up u1 and u2 matrices with ranges normalized to +/- 0.5
        # The following code adjusts things appropriately for odd and even values of rows and columns.
        if cols % 2:
            xrange = torch.arange(-(cols - 1) / 2, (cols) / 2) / (cols - 1)
        else:
            xrange = torch.arange(-cols / 2, cols/2) / cols
        
        if rows % 2:
            yrange = torch.arange(-(rows - 1) / 2, (rows) / 2) / (rows - 1)
        else:
            yrange = torch.arange(-rows / 2, rows/2) / rows

        # print("xrange: ", xrange.shape, "yrange: ", yrange.shape)
        
        # print("xrange: ", xrange)
        # # print("yrange: ", yrange)

        u1, u2 = torch.meshgrid(xrange, yrange, indexing='xy')

        # Quadrant shift to put 0 frequency at the corners
        u1 = torch.fft.ifftshift(u1).to(self.get_device())
        u2 = torch.fft.ifftshift(u2).to(self.get_device())

        # print("\n")
        # print("u1: ", u1)
        # print("u2: ", u2)

        # Matrix values contain frequency values as a radius from center (but quandrant shifted)
        radius = torch.sqrt(u1**2 + u2**2).to(self.get_device())

        return u1, u2, radius
    

    def initialize_sigmaonf(self, sigmaonf):
        if sigmaonf is None:
            # Choose a random value very close to zero (akin to choosing a sigmaonf very close to 0.5)
            return torch.randn(1) * 0.05
        else:
            assert sigmaonf > 0 and sigmaonf < 1
            # Transform sigmaonf to between -inf and inf by applying the inverse sigmoid function
            sigmaonf = torch.tensor(sigmaonf)
            return torch.log(sigmaonf / (1 - sigmaonf))
    
    def initialize_wls(self, wls):
        if wls is None:
            return torch.randn(self.nscale)
        else:
            assert np.all(wls > 0)  # Cannot have a negative wavelength
            # Rescale the wavelengths to be between 0 and 1 for faster training
            wls = torch.tensor((wls - self.min_wl) / (self.max_wl - self.min_wl))
            return torch.log(wls / (1 - wls))

    def rescale_wls(self, wls):
        return self.min_wl + wls * (self.max_wl - self.min_wl)
    
    def get_wls(self):
        return self.rescale_wls(torch.nn.functional.sigmoid(self.wls))
    
    def get_sigmaonf(self):
        return torch.nn.functional.sigmoid(self.sigmaonf)

    def get_device(self):
        return self.parameters().__next__().device
    
    def scale_max_min(self, x):
        x_min = torch.amin(x, dim=(-2, -1), keepdim=True)
        x_max = torch.amax(x, dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min)
    
    def get_params(self):
        # return a dictionary of the parameters
        return {
            "nscale": self.nscale.item(),
            "self.max_wl": self.max_wl,
            "wls": self.get_wls().tolist(),
            "sigmaonf": self.get_sigmaonf().item(),
            "trainable": self.wls.requires_grad,
            "return_phase": self.return_phase,
            "return_phase_asym": self.return_phase_asym,
            "return_ori": self.return_ori,
            "return_input": self.return_input,
            "self.cut_off": self.cut_off,
            "self.g": self.g,
            "self.T": self.T,
        }
