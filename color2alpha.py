'''
Application for testing the GIMP color to alpha algorithm

Maverick Reynolds
07.18.2023

'''

from PIL import Image
from io import BytesIO
import numpy as np
import sys


# Different scalar interpolation functions
def interpolate(x, interpolation=None):
    if interpolation == 'power':
        return x**2
    elif interpolation == 'root':
        return np.sqrt(x)
    elif interpolation == 'smooth':
        return (np.sin(np.pi/2*x))**2
    elif interpolation == 'inverse-sin':
        return np.arcsin(2*x-1)/np.pi + 0.5
    else:
        return x


def rgb_distance(pixels: np.array, color: np.array, shape='cube'):
    '''
    If shape is 'cube', the radius is the maximum orthogonal distance between the two colors
    If shape is 'sphere', the radius is the distance between the two colors in 3D space

    Returned value should always be between 0 and 255
    '''

    # Ensure parameters are RGB (three channels)
    pixels = pixels[:,:,:3]

    # Take advantage of numpy's vectorization here
    if shape == 'cube':
        return np.amax(abs(pixels - color), axis=2)
    elif shape == 'sphere':
        return np.linalg.norm(pixels - color, axis=2)


def color_to_alpha(pixels, color, transparency_threshold, opacity_threshold, shape='cube', interpolation=None):
    '''
    this function takes in the image and performs the GIMP color to alpha algorithm
    Colors within the transparency_threshold are converted to transparent
    Colors within the opacity_threshold are unchanged
    Colors between the two thresholds smoothly transition between transparent and opaque

    Takes advantage of np vectorization
    '''
    color = np.array(color)

    # Make new pixels and th channel for RGBA
    pixels = pixels[:,:,:3]
    new_pixels = np.copy(pixels)
    new_pixels = np.append(new_pixels, np.zeros((new_pixels.shape[0], new_pixels.shape[1], 1), dtype=np.uint8), axis=2)

    # Get the distance matrix
    distances = rgb_distance(pixels, color, shape=shape)

    # Create masks for pixels that are transparent and opaque
    transparency_mask = distances <= transparency_threshold
    opacity_mask = distances >= opacity_threshold

    # Calculate alpha values for pixels between the thresholds
    threshold_difference = opacity_threshold - transparency_threshold
    alpha = (distances - transparency_threshold) / threshold_difference
    alpha = np.clip(alpha, 0, 1)

    # Interpolate based on method provided
    alpha = interpolate(alpha, interpolation=interpolation)

    # Extrapolate along line passing through color and pixel onto the opacity threshold
    # This is the RGB value that will be used for the pixel
    proportion_to_opacity = distances / opacity_threshold
    extrapolated_colors = (pixels - color) / proportion_to_opacity[:, :, np.newaxis] + color

    extrapolated_colors = np.nan_to_num(extrapolated_colors, nan=0)
    extrapolated_colors = np.clip(np.around(extrapolated_colors), 0, 255).astype(np.uint8)

    # Reassign color values of intermediate pixels
    new_pixels[~transparency_mask & ~opacity_mask, :3] = extrapolated_colors[~transparency_mask & ~opacity_mask]
    # Reassign the alpha values of intermediate pixels
    new_pixels[:, :, 3] = alpha * 255

    return new_pixels


# For downloading the new image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return byte_im


# Color conversion functions
def tuple_to_hex(tup):
    return f'#{tup[0]:02x}{tup[1]:02x}{tup[2]:02x}'.upper()

def hex_to_tuple(hex: str):
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))


# To get the top colors in the image
def get_pixel_distribution(img):
    width, height = img.size
    counter = dict()

    # Count the number of pixels of each color
    for h in range(height):
        for w in range(width):
            pix = img.getpixel((w, h))
            if pix not in counter:
                counter[pix] = 1
            else:
                counter[pix] += 1

    # Sort the dictionary by value, highest to lowest
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return counter



def color_to_alpha_command(input_filename, output_filename):
    img = Image.open(input_filename)
    # User settings
    #shape = st.sidebar.selectbox('Shape (used for calculating distance in RGB-space)', ['sphere', 'cube'])
    shape = 'sphere'
    #interpolation = st.sidebar.selectbox('Interpolation', ['linear', 'power', 'root', 'smooth', 'inverse-sin'])
    interpolation = 'linear'
    top_threshold_bound = 255 if shape == 'cube' else 442
    #transparency_threshold = st.sidebar.slider('Transparency Threshold', 0, top_threshold_bound, 18)
    transparency_threshold = 18
    #opacity_threshold = st.sidebar.slider('Opacity Threshold', 0, top_threshold_bound, 193)
    opacity_threshold = 193
    # Background replacement option
    #if (use_background := st.sidebar.checkbox('Background Replacement')):
    #    background_color = st.sidebar.color_picker('Background Color', '#10EAEC')
    #    bg = Image.new('RGBA', img.size, background_color)
    # Color selection
    #st.sidebar.title('Color Selection')
    #color = '#FFFFFF'
    # Show the top colors in the image
    #if st.sidebar.checkbox('Use top color in image'):
    top_colors = get_pixel_distribution(img)[0][0]
    color = tuple_to_hex(top_colors)
    #color = st.sidebar.color_picker('Color', color)
#     if st.sidebar.button('Get top colors from image'):
#         NUM = 8
#         top_colors = get_pixel_distribution(img)[0:NUM]
#         colsb1, colsb2 = st.sidebar.columns(2)
#         colsb2.write('\n')
#
#         for i in range(0, NUM):
#             colsb1.color_picker(tuple_to_hex(top_colors[i][0]), tuple_to_hex(top_colors[i][0]))
#             colsb2.write(top_colors[i][1])
#             colsb2.write('\n')
#             colsb2.write('\n')
    # Main content on the page
#     col1, col2 = st.columns(2)
#
#     col1.subheader("Original Image 🖼️")
#     col1.image(img)
#
#     col2.subheader("Color to Alpha applied :wrench:")
    cta_arr = color_to_alpha(np.array(img), hex_to_tuple(color), transparency_threshold, opacity_threshold, shape=shape, interpolation=interpolation)
    cta_img = Image.fromarray(cta_arr, 'RGBA')
#     if use_background:
#         cta_img = Image.alpha_composite(bg, cta_img)
    cta_img.save(output_filename, dpi=img.info.get('dpi'), exif=img.info.get('exif'))
#     col2.image(cta_img)
#
#     # Add download button
#     col2.download_button('Download Image', convert_image(cta_img), file_name='color_to_alpha.png', mime='image/png')
#
# if __name__ == '__main__':
#     main()

color_to_alpha_command(sys.argv[1], sys.argv[2])
