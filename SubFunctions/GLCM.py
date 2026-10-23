import numpy as np
import cv2

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
class GLCM_Features:
    @staticmethod
    def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
        mi, ma = vmin, vmax
        ks = kernel_size
        h, w = img.shape

        # digitize
        bins = np.linspace(mi, ma + 1, nbit + 1)
        gl1 = np.digitize(img, bins) - 1
        gl2 = np.append(gl1[:, 1:], gl1[:, -1:], axis=1)

        # make glcm
        glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
        for i in range(nbit):
            for j in range(nbit):
                mask = ((gl1 == i) & (gl2 == j))
                glcm[i, j, mask] = 1

        kernel = np.ones((ks, ks), dtype=np.uint8)
        for i in range(nbit):
            for j in range(nbit):
                glcm[i, j] = cv2.filter2D(glcm[i, j], -1, kernel)

        glcm = glcm.astype(np.float32)
        return glcm

    def fast_glcm_mean(self, img, vmin=0, vmax=255, nbit=8, ks=5):
        '''
        calc glcm mean
        '''
        h, w = img.shape
        glcm = self.fast_glcm(img, vmin, vmax, nbit, ks)
        mean = np.zeros((h, w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                mean += glcm[i, j] * i / (nbit) ** 2

        return mean

    def fast_glcm_std(self, img, vmin=0, vmax=255, nbit=8, ks=5):
        '''
        calc glcm std
        '''
        h, w = img.shape
        glcm = self.fast_glcm(img, vmin, vmax, nbit, ks)
        mean = np.zeros((h, w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                mean += glcm[i, j] * i / (nbit) ** 2

        std2 = np.zeros((h, w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                std2 += (glcm[i, j] * i - mean) ** 2

        std = np.sqrt(std2)
        return std

    def fast_glcm_contrast(self, img, vmin=0, vmax=255, nbit=8, ks=5):
        '''
        calc glcm contrast
        '''
        h, w = img.shape
        glcm = self.fast_glcm(img, vmin, vmax, nbit, ks)
        cont = np.zeros((h, w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                cont += glcm[i, j] * (i - j) ** 2

        return cont

    def fast_glcm_dissimilarity(self, img, vmin=0, vmax=255, nbit=8, ks=5):
        '''
        calc glcm dissimilarity
        '''
        h, w = img.shape
        glcm = self.fast_glcm(img, vmin, vmax, nbit, ks)
        diss = np.zeros((h, w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                diss += glcm[i, j] * np.abs(i - j)

        return diss

    def fast_glcm_homogeneity(self, img, vmin=0, vmax=255, nbit=8, ks=5):
        '''
        calc glcm homogeneity
        '''
        h, w = img.shape
        glcm = self.fast_glcm(img, vmin, vmax, nbit, ks)
        homo = np.zeros((h, w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                homo += glcm[i, j] / (1. + (i - j) ** 2)

        return homo

    def fast_glcm_ASM(self, img, vmin=0, vmax=255, nbit=8, ks=5):
        '''
        calc glcm asm, energy
        '''
        h, w = img.shape
        glcm = self.fast_glcm(img, vmin, vmax, nbit, ks)
        asm = np.zeros((h, w), dtype=np.float32)
        for i in range(nbit):
            for j in range(nbit):
                asm += glcm[i, j] ** 2

        ene = np.sqrt(asm)
        return asm, ene

    def fast_glcm_max(self, img, vmin=0, vmax=255, nbit=8, ks=5):
        '''
        calc glcm max
        '''
        glcm = self.fast_glcm(img, vmin, vmax, nbit, ks)
        max_ = np.max(glcm, axis=(0, 1))
        return max_

    def fast_glcm_entropy(self, img, vmin=0, vmax=255, nbit=8, ks=5):
        '''
        calc glcm entropy
        '''
        glcm = self.fast_glcm(img, vmin, vmax, nbit, ks)
        pnorm = glcm / np.sum(glcm, axis=(0, 1)) + 1. / ks ** 2
        ent = np.sum(-pnorm * np.log(pnorm), axis=(0, 1))
        return ent
