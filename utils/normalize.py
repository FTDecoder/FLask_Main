
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    '''
    convert rgb image to gray scale, it uses formula
    gray_img = 0.21 R + 0.72 G + 0.07 B
    '''

    return np.dot(rgb[..., :3], [0.21, 0.72, 0.07])


# read image
im = plt.imread("img/photo.jpg").astype(float)
gray = rgb2gray(im)
gray /= 255
plt.imshow(gray, interpolation='none', cmap=plt.cm.gray)

# emmboss filter
kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

em_img = scipy.signal.convolve2d(gray, kernel)
em_img *= 255

plt.subplot(1, 3, 1)
plt.imshow(im, interpolation='none', cmap=plt.cm.gray)
plt.subplot(1, 3, 2)
plt.imshow(gray, interpolation='none', cmap=plt.cm.gray)
plt.show()


def normalize(image, x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil, rad_pixels, angular_div):
    """
    Description:
        Normalize iris region by unwraping the circular region into a rectangular
        block of constant dimensions.
    Input:
        image		- Input iris image.
        x_iris		- x-coordinate of the circle defining the iris boundary.
        y_iris		- y-coordinate of the circle defining the iris boundary.
        r_iris		- Radius of the circle defining the iris boundary.
        x_pupil		- x-coordinate of the circle defining the pupil boundary.
        y_pupil		- y-coordinate of the circle defining the pupil boundary.
        r_pupil		- Radius of the circle defining the pupil boundary.
        radpixels	- Radial resolution (vertical dimension).
        angulardiv	- Angular resolution (horizontal dimension).
    Output:
        polar_array	- Normalized form of the iris region.
        polar_noise	- Normalized form of the noise region.
    """
    radius_pixels = rad_pixels + 2
    angle_divisions = angular_div - 1

    r = np.arange(radius_pixels)
    theta = np.linspace(0, 2 * np.pi, angle_divisions + 1)

    # Calculate displacement of pupil center from the iris center
    ox = x_pupil - x_iris
    oy = y_pupil - y_iris

    if ox <= 0:
        sgn = -1
    elif ox > 0:
        sgn = 1

    if ox == 0 and oy > 0:
        sgn = 1

    a = np.ones(angle_divisions + 1) * (ox ** 2 + oy ** 2)

    # Need to do something for ox = 0
    if ox == 0:
        phi = np.pi / 2
    else:
        phi = np.arctan(oy / ox)

    b = sgn * np.cos(np.pi - phi - theta)

    # Calculate radius around the iris as a function of the angle
    r = np.sqrt(a) * b + np.sqrt(a * b ** 2 - (a - r_iris ** 2))
    r = np.array([r - r_pupil])

    rmat = np.dot(np.ones([radius_pixels, 1]), r)

    rmat = rmat * np.dot(np.ones([angle_divisions + 1, 1]),
                         np.array([np.linspace(0, 1, radius_pixels)])).transpose()
    rmat = rmat + r_pupil

    # Exclude values at the boundary of the pupil iris border, and the iris scelra border
    # as these may not correspond to areas in the iris region and will introduce noise.
    # ie don't take the outside rings as iris data.
    rmat = rmat[1: radius_pixels - 1, :]

    # Calculate cartesian location of each data point around the circular iris region
    xcosmat = np.dot(np.ones([radius_pixels - 2, 1]), np.array([np.cos(theta)]))
    xsinmat = np.dot(np.ones([radius_pixels - 2, 1]), np.array([np.sin(theta)]))

    xo = rmat * xcosmat
    yo = rmat * xsinmat

    xo = x_pupil + xo
    xo = np.round(xo).astype(int)
    coords = np.where(xo >= image.shape[1])
    xo[coords] = image.shape[1] - 1
    coords = np.where(xo < 0)
    xo[coords] = 0

    yo = y_pupil - yo
    yo = np.round(yo).astype(int)
    coords = np.where(yo >= image.shape[0])
    yo[coords] = image.shape[0] - 1
    coords = np.where(yo < 0)
    yo[coords] = 0

    # Extract intensity values into the normalised polar representation through
    # interpolation
    # x,y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    # f = interpolate.interp2d(x, y, image, kind='linear')
    # polar_array = f(xo, yo)
    # polar_array = polar_array / 255

    polar_array = image[yo, xo]
    polar_array = polar_array / 255

    # Create noise array with location of NaNs in polar_array
    polar_noise = np.zeros(polar_array.shape)
    coords = np.where(np.isnan(polar_array))
    polar_noise[coords] = 1

    # Get rid of outling points in order to write out the circular pattern
    image[yo, xo] = 255

    # Get pixel coords for circle around iris
    x, y = circlecoords([x_iris, y_iris], r_iris, image.shape)
    image[y, x] = 255

    # Get pixel coords for circle around pupil
    xp, yp = circlecoords([x_pupil, y_pupil], r_pupil, image.shape)
    image[yp, xp] = 255

    # Replace NaNs before performing feature encoding
    coords = np.where((np.isnan(polar_array)))
    polar_array2 = polar_array
    polar_array2[coords] = 0.5
    avg = np.sum(polar_array2) / (polar_array.shape[0] * polar_array.shape[1])
    polar_array[coords] = avg

    return polar_array, polar_noise.astype(bool)


# ------------------------------------------------------------------------------
def circlecoords(c, r, imgsize, nsides=600):
    """
    Description:
        Find the coordinates of a circle based on its centre and radius.
    Input:
        c   	- Centre of the circle.
        r  		- Radius of the circle.
        imgsize - Size of the image that the circle will be plotted onto.
        nsides 	- Number of sides of the convex-hull bodering the circle
                  (default as 600).
    Output:
        x,y     - Circle coordinates.
    """
    a = np.linspace(0, 2 * np.pi, 2 * nsides + 1)
    xd = np.round(r * np.cos(a) + c[0])
    yd = np.round(r * np.sin(a) + c[1])

    #  Get rid of values larger than image
    xd2 = xd
    coords = np.where(xd >= imgsize[1])
    xd2[coords[0]] = imgsize[1] - 1
    coords = np.where(xd < 0)
    xd2[coords[0]] = 0

    yd2 = yd
    coords = np.where(yd >= imgsize[0])
    yd2[coords[0]] = imgsize[0] - 1
    coords = np.where(yd < 0)
    yd2[coords[0]] = 0

    x = np.round(xd2).astype(int)
    y = np.round(yd2).astype(int)
    return x, y