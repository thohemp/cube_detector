import argparse
import cv2
import os
import numpy as np
import random
import math
import sys
import time

NAMES = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z']

cube_list = []
cube_names = []
background_list = []
background_names = []

NUM_TOTAL_VARIATIONS = 7500
SAMPLES_PER_IMAGE = 15
SAMPLE = True


def load_images_from_folder(folder, images, names):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            images.append(img)
            names.append(filename)
    return images

def rotation(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, 360-angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg

def writeYOLOAnnotation(textfile, idx, bb, angle):
    name_index = NAMES.index(os.path.splitext(cube_names[idx])[0])          
    with open(textfile, 'a') as the_file:
        the_file.write(str(name_index))
        the_file.write(' ')
        the_file.write(str(bb[0]))
        the_file.write(' ')
        the_file.write(str(bb[1]))
        the_file.write(' ')
        the_file.write(str(bb[2]))
        the_file.write(' ')
        the_file.write(str(bb[3]))
        the_file.write(' ')
        the_file.write(str(angle))
        the_file.write('\n')

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def updt(total, progress):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()

def main(opt):


    bg_rect = [100, 255, 1700, 700]
    background_dir = os.getcwd() + "/"+ opt.bg
    backgrounds = load_images_from_folder(background_dir, background_list, background_names)
    
    object_dir = os.getcwd() + "/"+ opt.obj
    objects = load_images_from_folder(object_dir, cube_list, cube_names)

    print("Found %d backgrounds in image folder." % len(backgrounds))
    print("Found %d objects in image folder." % len(objects))
    print("Generating file count: %d" % NUM_TOTAL_VARIATIONS)
    for x in range(NUM_TOTAL_VARIATIONS):
        image_name = str(x).zfill(5)
        textfile_name = str(x).zfill(5) +".txt"
        imagefile_name = str(x).zfill(5) +".jpg"

        num_of_samples = random.randint(1,15)

        bg_idx = random.randint(0, len(backgrounds))
        background = backgrounds[bg_idx-1]
        if len(background.shape) == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
        
        if background.shape is not (1920,1080):
            background = cv2.resize(background, (1920, 1080), interpolation = cv2.INTER_AREA)

        result_img = background.copy()

        for y in range(num_of_samples):
            idx = random.randrange(len(objects))
            object = objects[idx]

            ### RADNOM SCALING ###
            scale_percent = random.randint(80,250)

            #mu, sigma = 1, 0.1
            #w = np.random.normal(mu, sigma, 1)
            #h = np.random.normal(mu, sigma, 1)

            width = int(object.shape[1] * scale_percent  / 100)
            height = int(object.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            object = cv2.resize(object, dim, interpolation = cv2.INTER_AREA)



            ### RANDOM BLUR ###
            if random.randint(0,1) == 1:
                ksize = random.randint(1,10 )
                if(ksize % 2 != 1):
                    ksize = ksize+1
                object = cv2.GaussianBlur(object,(ksize,ksize),0)


            ### RANDOM ROTATING ###
            # mu, sigma = 0, 90
            # rot_degree = int(np.random.normal(mu, sigma, 1))
            rot_degree = random.randint(0,360)

            #if(rot_degree < 0):
            #   rot_degree = 360 - rot_degree
            # grab the dimensions of the image and calculate the center of the
            # image
        
            (h, w) = object.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            # rotate our image around the center of the image
            rotated = rotation(object, rot_degree)
            rad = math.radians(rot_degree)
            sin = math.sin(rad)
            cos = math.cos(rad)
            
            #cv2.imshow("Rotated", rotated)
            #cv2.waitKey(0)

            b_w = int((h * abs(sin)) + (w * abs(cos)))
            b_h = int((h * abs(cos)) + (w * abs(sin)))

            rand_pos = [random.randint(bg_rect[0], bg_rect[0]+ bg_rect[2]),random.randint(bg_rect[1], bg_rect[1]+ bg_rect[3])]

            y1, y2 = int(rand_pos[1]-b_h/2),  int(rand_pos[1]+b_h/2)
            x1, x2 = int(rand_pos[0]-b_w/2), int(rand_pos[0]+b_w/2)
            y1_, y2_ = int(rand_pos[1]-h/2),  int(rand_pos[1]+h/2)
            x1_, x2_ = int(rand_pos[0]-w/2), int(rand_pos[0]+w/2)



            trans_indices = rotated[...,3] != 0 # Where not transparent
            alpha_s = rotated[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                result_img[y1:y2, x1:x2, c] = (alpha_s * rotated[:, :, c] +
                            alpha_l * result_img[y1:y2, x1:x2, c])

            # cv2.imshow("Rotated", target_im)
            


            ### CREATING BOUNDING BOX ###

            bb = [(x1+(x2-x1)/2)/background.shape[1], (y1+(y2-y1)/2)/background.shape[0], (x2_-x1_)/background.shape[1], (y2_-y1_)/background.shape[0]]
            #result_img = cv2.rectangle(result_img,(x1, y1),(x2, y2), (255,0,0), 2)

            #cv2.imshow("Rotated", result_img)
            #cv2.waitKey(0)


            rot_degree = rot_degree%90
            ### WRITE ANNOTATION TO FILE ###
            textfile = os.getcwd() + "/labels/" + textfile_name
            writeYOLOAnnotation(textfile, idx, bb, rot_degree)
        
        
        # Create annotation txt even though there are no objects
        if(num_of_samples == 0):
            textfile = os.getcwd() + "/labels/" + textfile_name
            with open(textfile, 'a') as the_file:
                the_file.write(' ')
         
                   
        file = "images/" + imagefile_name

        res = adjust_gamma(result_img, random.uniform(0.5,1.5))

        cv2.imwrite(file, res)
        updt(NUM_TOTAL_VARIATIONS, x)

    print("\n")

        
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg', type=str, help='initial weights path')
    parser.add_argument('--obj', type=str, help='directory for object images')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
