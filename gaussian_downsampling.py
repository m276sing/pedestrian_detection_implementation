import imutils


def gaussian_reduction(image, scale_factor, minimum_height_required, minimum_width_required):
    yield image
    while True:
        width_downscaling = int(round(image.shape[1] / scale_factor))
        height_downscaling = int(round(image.shape[0] / scale_factor))
        image = imutils.resize(image, width_downscaling, height_downscaling)
        if (image.shape[1] >= minimum_width_required) and (image.shape[0] >= minimum_height_required):
            yield image
        else:
            break
