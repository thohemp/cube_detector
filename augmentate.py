
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

images = []
image_names = []

NUM_TOTAL_VARIATIONS = 500
SAMPLES_PER_IMAGE = 15
SAMPLE = True


def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            images.append(img)
            image_names.append(filename)
    return images

def rotation(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg

def writeYOLOAnnotation(textfile, idx, bb):
    name_index = NAMES.index(os.path.splitext(image_names[idx])[0])          
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
        the_file.write('\n')

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


    bg_rect = [530, 255, 1000, 450]
    background = cv2.imread(opt.bg)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

    object_dir = os.getcwd() + "/"+ opt.objects
    objects = load_images_from_folder(object_dir)
    print("Found %d objects in image folder." % len(objects))
    print("Generating files: %d" % NUM_TOTAL_VARIATIONS)
    for x in range(NUM_TOTAL_VARIATIONS):
        image_name = str(x).zfill(5)
        textfile_name = str(x).zfill(5) +".txt"
        imagefile_name = str(x).zfill(5) +".jpg"

        result_img = background.copy()

        if SAMPLE:
            for y in range(SAMPLES_PER_IMAGE):
                idx = random.randrange(len(objects))
                object = objects[idx]
                rot_degree = random.randint(0,360)
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

                trans_indices = rotated[...,3] != 0 # Where not transparent
                alpha_s = rotated[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    result_img[y1:y2, x1:x2, c] = (alpha_s * rotated[:, :, c] +
                                alpha_l * result_img[y1:y2, x1:x2, c])

                # cv2.imshow("Rotated", target_im)

                bb = [(x1+(x2-x1)/2)/background.shape[1], (y1+(y2-y1)/2)/background.shape[0], (x2-x1)/background.shape[1], (y2-y1)/background.shape[0]]

                #result_img = cv2.rectangle(result_img,(x1, y1),(x2, y2), (255,0,0), 2)

                #cv2.imshow("Rotated", result_img)
                #cv2.waitKey(0)
                textfile = os.getcwd() + "/aug_labels/" + textfile_name
                writeYOLOAnnotation(textfile, idx, bb)
            
        else:
            for idx,object in enumerate(objects):
                rot_degree = random.randint(0,360)
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

                trans_indices = rotated[...,3] != 0 # Where not transparent
                alpha_s = rotated[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    result_img[y1:y2, x1:x2, c] = (alpha_s * rotated[:, :, c] +
                                alpha_l * result_img[y1:y2, x1:x2, c])

                # cv2.imshow("Rotated", target_im)

                bb = [(x1+(x2-x1)/2)/background.shape[1], (y1+(y2-y1)/2)/background.shape[0], (x2-x1)/background.shape[1], (y2-y1)/background.shape[0]]

                #result_img = cv2.rectangle(result_img,(x1, y1),(x2, y2), (255,0,0), 2)

                #cv2.imshow("Rotated", result_img)
                #cv2.waitKey(0)

                textfile = os.getcwd() + "/aug_labels/" + textfile_name
                writeYOLOAnnotation(textfile, idx, bb)
            
            
        file = "aug_images/" + imagefile_name
        cv2.imwrite(file, result_img)
        updt(NUM_TOTAL_VARIATIONS, x)

    print("\n")

        
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg', type=str, help='initial weights path')
    parser.add_argument('--objects', type=str, help='directory for object images')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)