import cv2
import numpy as np

class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img
    
    def image_histogram_equalization(self, channel, number_bins=256):
        image_histogram, bins = np.histogram(channel.flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize
    
        # use linear interpolation of cdf to find new pixel values
        channel_equalized = np.interp(channel.flatten(), bins[:-1], cdf)
    
        return channel_equalized.reshape(channel.shape), cdf

    # Add your own preprocessing techniques here.