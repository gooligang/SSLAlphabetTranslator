import cv2
import numpy as np
from scipy import ndimage
import os


def save_image(input_image_path, filename, outputDir):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    output_image_path = os.path.join(outputDir, filename)
    cv2.imwrite(output_image_path, image)


def flip_and_save_image(input_image_path, filename, outputDir, added_file_name='flipped'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    flipped_image = cv2.flip(image, 1)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, flipped_image)


def rotate_and_save_image(input_image_path, filename, outputDir, target_angle=20, added_file_name='rotated'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    angle = np.random.uniform(target_angle - 5, target_angle + 5)
    rotated_image = ndimage.rotate(image, angle, reshape=False)

    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)


def median_blur_and_save_image(input_image_path, filename, outputDir, blur_strength=5, added_file_name='median_blur'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    augmented_image = cv2.medianBlur(image, blur_strength)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)


def contrast_and_save_image(input_image_path, filename, outputDir, contrast=2, added_file_name='contrast'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    augmented_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)


def gamma_and_save_image(input_image_path, filename, outputDir, gamma=2, added_file_name='gamma'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    augmented_image = cv2.LUT(image, table)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)


def saturate_and_save_image(input_image_path, filename, outputDir, saturation=1.0, added_file_name='saturation'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    augmented_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    addedFilenameText = "saturation"
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)


def sharpen_and_save_image(input_image_path, filename, outputDir, scale=1.0, added_file_name='sharpen'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    augmented_image = cv2.Laplacian(image, cv2.CV_64F)
    augmented_image = np.uint8(np.clip(image + scale * augmented_image, 0, 255))
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)




def blue_color_enhance(input_image_path, filename, outputDir,added_blue_value = 1.5,added_file_name='_blue_enhance'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    blue_channel = image[:, :, 0]
    enhanced_blue_channel = np.clip(blue_channel * added_blue_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([image[:, :, 2], image[:, :, 1], enhanced_blue_channel])
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)

def blue_color_enhance_rotate(input_image_path, filename, outputDir,added_blue_value = 1.5,added_file_name='_blue_enhance',target_angle = 10):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    blue_channel = image[:, :, 0]
    enhanced_blue_channel = np.clip(blue_channel * added_blue_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([image[:, :, 2], image[:, :, 1], enhanced_blue_channel])
    angle = np.random.uniform(target_angle - 5, target_angle + 5)
    rotated_image = ndimage.rotate(augmented_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)


def red_color_enhance(input_image_path, filename, outputDir,added_red_value = 1.5,added_file_name='_red_enhance'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    red_channel = image[:, :, 2]
    enhanced_red_channel = np.clip(red_channel * added_red_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([enhanced_red_channel, image[:, :, 1], image[:, :, 0]])
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)

def red_color_enhance_rotate(input_image_path, filename, outputDir,added_red_value = 1.5,added_file_name='_red_enhance_rotate',target_angle = 10):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    red_channel = image[:, :, 2]
    enhanced_red_channel = np.clip(red_channel * added_red_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([enhanced_red_channel, image[:, :, 1], image[:, :, 0]])
    angle = np.random.uniform(target_angle - 5, target_angle + 5)
    rotated_image = ndimage.rotate(augmented_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)



def green_color_enhance(input_image_path, filename, outputDir,added_green_value = 1.5,added_file_name='_green_enhance'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    green_channel = image[:, :, 1]
    enhanced_green_channel = np.clip(green_channel * added_green_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([image[:, :, 2], enhanced_green_channel, image[:, :, 0]])
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)


def green_color_enhance_rotate(input_image_path, filename, outputDir,added_green_value = 1.5,added_file_name='_green_enhance_rotate', target_angle = 10):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    green_channel = image[:, :, 1]
    enhanced_green_channel = np.clip(green_channel * added_green_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([image[:, :, 2], enhanced_green_channel, image[:, :, 0]])
    angle = np.random.uniform(target_angle - 5, target_angle + 5)
    rotated_image = ndimage.rotate(augmented_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)


def green_blue_color_enhance(input_image_path, filename, outputDir,added_value = 1.5,added_file_name='_green_blue_enhance'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    enhanced_blue_channel = np.clip(blue_channel * added_value, 0, 255).astype(np.uint8)
    enhanced_green_channel = np.clip(green_channel * added_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([enhanced_blue_channel, enhanced_green_channel, image[:, :, 2]])
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)


def green_blue_color_enhance_rotate(input_image_path, filename, outputDir,added_value = 1.5,added_file_name='_green_blue_enhance_rotate', target_angle = 10):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)  # Read input image
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    enhanced_blue_channel = np.clip(blue_channel * added_value, 0, 255).astype(np.uint8)
    enhanced_green_channel = np.clip(green_channel * added_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([enhanced_blue_channel, enhanced_green_channel, image[:, :, 2]])
    angle = np.random.uniform(target_angle - 5, target_angle + 5)  # get random angle
    rotated_image = ndimage.rotate(augmented_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)



def green_red_color_enhance(input_image_path, filename, outputDir,added_value = 1.5,added_file_name='_green_red_enhance'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    enhanced_red_channel = np.clip(red_channel * added_value, 0, 255).astype(np.uint8)
    enhanced_green_channel = np.clip(green_channel * added_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([enhanced_red_channel, enhanced_green_channel, image[:, :, 0]])
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)

def green_red_color_enhance_rotate(input_image_path, filename, outputDir,added_value = 1.5,added_file_name='_green_red_enhance_rotate',target_angle = 10):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    enhanced_red_channel = np.clip(red_channel * added_value, 0, 255).astype(np.uint8)
    enhanced_green_channel = np.clip(green_channel * added_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([enhanced_red_channel, enhanced_green_channel, image[:, :, 0]])
    angle = np.random.uniform(target_angle - 5, target_angle + 5)  # get random angle
    rotated_image = ndimage.rotate(augmented_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)


def blue_red_color_enhance(input_image_path, filename, outputDir,added_value = 1.5,added_file_name='_blue_red_enhance'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    blue_channel = image[:, :, 0]
    red_channel = image[:, :, 2]
    enhanced_blue_channel = np.clip(blue_channel * added_value, 0, 255).astype(np.uint8)
    enhanced_red_channel = np.clip(red_channel * added_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([enhanced_blue_channel, image[:, :, 1], enhanced_red_channel])
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)


def blue_red_color_enhance_and_rotate(input_image_path, filename, outputDir,added_value = 1.5,added_file_name='_blue_red_enhance_rotate', target_angle = 10):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    blue_channel = image[:, :, 0]
    red_channel = image[:, :, 2]
    enhanced_blue_channel = np.clip(blue_channel * added_value, 0, 255).astype(np.uint8)
    enhanced_red_channel = np.clip(red_channel * added_value, 0, 255).astype(np.uint8)
    augmented_image = cv2.merge([enhanced_blue_channel, image[:, :, 1], enhanced_red_channel])
    angle = np.random.uniform(target_angle - 5, target_angle + 5)
    rotated_image = ndimage.rotate(augmented_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)

def gray_scale(input_image_path, filename, outputDir,added_file_name='_gray_scaled'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    augmented_image = cv2.merge([gray_image] * 3)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, augmented_image)

def gray_scale_and_rotate(input_image_path, filename, outputDir,added_file_name='_gray_scaled',target_angle = 10):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    augmented_image = cv2.merge([gray_image] * 3)
    angle = np.random.uniform(target_angle - 5, target_angle + 5)
    rotated_image = ndimage.rotate(augmented_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)


def flip_rotate_and_save_image(input_image_path, filename, outputDir, target_angle=20, added_file_name='rotated_blur'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    angle = np.random.uniform(target_angle - 5, target_angle + 5)
    flipped_image = cv2.flip(image, 1)
    rotated_image = ndimage.rotate(flipped_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)


def median_blur_rotate_and_save_image(input_image_path, filename, outputDir, target_angle=20, blur_strength=5,
                                      added_file_name='flipped_rotated'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    augmented_image = cv2.medianBlur(image, blur_strength)
    angle = np.random.uniform(target_angle - 5, target_angle + 5)
    rotated_image = ndimage.rotate(augmented_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)


def contrast_rotate_and_save_image(input_image_path, filename, outputDir, target_angle=20, contrast=2,
                                   added_file_name='contrast_rotated'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    augmented_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    angle = np.random.uniform(target_angle - 5, target_angle + 5)
    rotated_image = ndimage.rotate(augmented_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)


def gamma_rotate_and_save_image(input_image_path, filename, outputDir, target_angle=20, gamma=2,
                                added_file_name='gamma'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    augmented_image = cv2.LUT(image, table)
    angle = np.random.uniform(target_angle - 5, target_angle + 5)
    rotated_image = ndimage.rotate(augmented_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)


def saturate_rotate_and_save_image(input_image_path, filename, outputDir, target_angle=20, saturation=1.0,
                                   added_file_name='saturation'):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    augmented_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    angle = np.random.uniform(target_angle - 5, target_angle + 5)
    rotated_image = ndimage.rotate(augmented_image, angle, reshape=False)
    addedFilenameText = added_file_name
    newFileName = filename.replace('.', f'_{addedFilenameText}.')
    output_image_path = os.path.join(outputDir, newFileName)
    cv2.imwrite(output_image_path, rotated_image)





def one_dim_grayscale(input_image_path, filename, outputDir):
    image_path = os.path.join(input_image_path, filename)
    image = cv2.imread(image_path)  # Read input image

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_image_path = os.path.join(outputDir, filename)
    cv2.imwrite(output_image_path, gray_image)
# input files and folder path



directory = "Input folder path"
outputDir = "output folder path"


for foldername in os.listdir(directory):
    letterFolder = os.path.join(directory, foldername)
    augLetterFolder = os.path.join(outputDir, foldername)
    index = 0
    print("new letter")
    for filename in os.listdir(letterFolder):
        #call on augmentation methods
        # Example
        rotate_and_save_image(letterFolder,filename,outputDir,10,"_rotated_10")
