import numpy as np
import cv2
from math import pi,sqrt
import matplotlib.pyplot as plt
from numpy.ma.core import multiply
import scipy.signal
import scipy.fftpack

def main():
    def get_fourier(img):
        f =  np.fft.fft2(img)
        return f
    def get_padded(img):
        n,m = img.shape
        skel = np.zeros((2*n,2*m))
        skel[:m,:n] = img
        return skel
    def show_image(image):
        plt.imshow(image, cmap = 'gray')
        plt.show()
    def DFT_matrix(N):
        i, j = np.meshgrid(np.arange(N), np.arange(N))
        omega = np.exp( - 2 * pi * 1J / N )
        W = np.power( omega, i * j ) / sqrt(N)
        return W
    def butterLow(cutoff, critical):
        normal_cutoff = float(cutoff) / critical
        b, a = signal.butter(2, normal_cutoff, btype='lowpass')
        print(b)
        return b, a
    def create_notch_filter(image):
        n,m = image.shape
        notch_filter = np.ones((n,m))
        notch_filter[184:196,:184] = np.zeros((12,184))
        notch_filter[250:262,:240] = np.zeros((12,240))
        notch_filter[316:328,:184] = np.zeros((12,184))
        notch_filter[184:196,328:] = np.zeros((12,184))
        notch_filter[250:262,272:] = np.zeros((12,240))
        notch_filter[316:328,328:] = np.zeros((12,184))

        notch_filter[:184,184:196] = np.zeros((184,12))
        notch_filter[:240,250:262] = np.zeros((240,12))
        notch_filter[:184,316:328] = np.zeros((184,12))
        notch_filter[328:,184:196] = np.zeros((184,12))
        notch_filter[272:,250:262] = np.zeros((240,12))
        notch_filter[328:,316:328] = np.zeros((184,12))
        return notch_filter
    def create_new_notch(image):
        n,m = image.shape
        filter = np.ones((n,m))
        filter[188:192,188:192] = np.zeros((4,4))
        filter[320:324,320:324] = np.zeros((4,4))
        return filter
    def get_1(image):
        n,m = image.shape
        skel = np.ones((n,m))
        for i in range(n):
            for j in range(m):
                skel[i][j] = image[i][j]*((-1)**(i + j))
        return skel
    def butterFilter(data, cutoff_freq, nyq_freq, order):
        b, a = butterLow(cutoff_freq, nyq_freq, order)
        y = signal.filtfilt(b, a, data)
        return y
    def create_butterworth(cutoff, image):
        m,n = image.shape
        filter = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                term = (get_d(i,j)/cutoff)**4
                filter[i][j] = 1/(1 + term)
        return filter
    def get_d(k,l):
        return sqrt((k - 256)**2 + (l - 256)**2)

    def q3():
        camera_img = cv2.imread("input_image.jpg", 0)
        camera_img = np.array(camera_img)
        m,n = camera_img.shape
        box_filter = 1/81* np.ones((9,9))
        print(box_filter)
        skel1 = np.zeros((m + 9, n + 9))
        skel2 = np.zeros((m + 9, n + 9))
        skel1[:m][:,:n] = camera_img
        skel2[:9][:,:9] = box_filter
        dft_1 = np.fft.fft2(skel1)
        dft_2 = np.fft.fft2(skel2)
        magnitude_spectrum =  np.log(np.abs(dft_1))
        magnitude_spectrum = np.asarray(magnitude_spectrum, dtype = np.uint8)
        plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.show()
        #dft_1 = np.fft.fft2(skel1)
        #dft_2 = np.fft.fft2(skel2)
        mul_mat = np.multiply(dft_1, dft_2)
        inverse_mat = np.fft.ifft2(mul_mat).real
        plt.imshow(inverse_mat, cmap = 'gray')
        plt.show()

    def q1():
        #creating a low pass filter first
        
        cutoff_values = [10,30,60]
        camera_img = np.array(cv2.imread("input_image.jpg", 0))
        padded_camera_img = get_padded(camera_img)
        centred_camera_img = get_1(padded_camera_img)
        filter_list = []
        for cutoff in cutoff_values:
            filter_list.append(create_butterworth(cutoff, centred_camera_img))
        for filter in filter_list:
            show_image(filter)
        dft_image = np.fft.fft2(centred_camera_img)
        magnitude_spectrum =  np.array(np.log(np.abs(dft_image)), dtype=np.uint8)
        show_image(magnitude_spectrum)
        for filter in filter_list:
            elementwise_mult = np.multiply(dft_image, filter)
            inverse_real = np.fft.ifft2(elementwise_mult).real
            centred_inv = get_1(inverse_real)
            n,m = centred_inv.shape
            show_image(centred_inv)
            show_image(centred_inv[:n//2,:m//2])
    def q4():
        noisy_img = np.array(cv2.imread("noiseIm.jpg", 0))
        
        padded_in = get_padded(noisy_img)
        centre_noisy = get_1(padded_in)
        fourier_noisy = get_fourier(centre_noisy)
        magnitude_spectrum =  np.log(np.abs(fourier_noisy))
        magnitude_spectrum = np.asarray(magnitude_spectrum, dtype = np.uint8)
        show_image(magnitude_spectrum)
        notch_filter = create_new_notch(magnitude_spectrum)
        show_image(notch_filter)
        multiply_matrix = np.multiply(fourier_noisy, notch_filter)
        show_image(multiply_matrix.real)
        inv = np.fft.ifft2(multiply_matrix)
        inv = inv.real
        final_im = get_1(inv)
        show_image(final_im)
        #magnitude spectrum of final image
        spec = np.asarray(np.log(np.abs(np.fft.fft2(final_im))), dtype = np.uint8)
        show_image(spec)
        #print(magnitude_spectrum[0])
        denoisy_img = np.array(cv2.imread("denoiseIm.jpg", 0))
        
        padded_denoise = get_padded(denoisy_img)
        centre_denoisy = get_1(padded_denoise)
        fourier_denoisy = get_fourier(centre_denoisy)
        magnitude_spectrum_denoise =  np.log(np.abs(fourier_denoisy))
        magnitude_spectrum_denoise = np.asarray(magnitude_spectrum_denoise, dtype = np.uint8)
        show_image(magnitude_spectrum_denoise)
        fig = plt.figure(figsize=(10,7))
        fig.add_subplot(2,1,1)
        plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.axis('off')
        plt.title("noisy")
        fig.add_subplot(2,1,2)
        plt.imshow(magnitude_spectrum_denoise, cmap='gray')
        plt.axis('off')
        plt.title("denoise")
        plt.show()

    q_n = input("enter question number:")
    exec("q{}()".format(q_n))

if( __name__ == "__main__"):
    main()