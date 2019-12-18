import matplotlib.pyplot as mat_plot_library

from skimage.feature import hog
from skimage import data, exposure


image = data.astronaut()

fd, get_hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

figure, (img1, img2) = mat_plot_library.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

img1.axis('off')
img1.imshow(image, cmap='gray')
img1.set_title('Original Image')

# Resizing the histogram to display a better image
new_hog_image = exposure.rescale_intensity(get_hog_image, in_range=(0, 10))

img2.axis('off')
img2.imshow(new_hog_image, cmap='gray')
img2.set_title('Histogram of Oriented Gradients Image')
mat_plot_library.show()
