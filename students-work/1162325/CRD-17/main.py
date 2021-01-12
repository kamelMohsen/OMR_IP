import argparse
import os
import datetime
from commonfunctions import *
import numpy as np
import cv2
import math
from scipy import ndimage
import math
import imutils
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import binary_erosion, binary_dilation
import pandas as pd
import os

def digital_or_not(input_image):
    
    if '.png' in input_image.lower():
        return  1
    else:
        return  0



def show(img, factor=1,name="image"):
    """ 
    show an image until the escape key is pressed
    :param factor: scale factor (default 1, half size)
    """
    if factor != 1.0:
        img = cv2.resize(img, (0,0), fx=factor, fy=factor) 

    cv2.imshow(name,img)
    while(1):
        k = cv2.waitKey(0)
        if k==27:    # Esc key to stop
            break
    cv2.destroyAllWindows()



def gaussian(img,gaussian_kernel=5):
    
    if(len(img.shape)>2):
        img = cv2.GaussianBlur(img,(gaussian_kernel,gaussian_kernel),0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mid = 0.5
        mean = np.mean(img)
        gamma = math.log(mid*255)/math.log(mean)
        img = np.power(img, gamma).clip(0,255).astype(np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.GaussianBlur(img,(gaussian_kernel,gaussian_kernel),0)
    
    
    return img




def binarize(img,block_size=25,neighbours=7):
    if(len(img.shape)>2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,neighbours)
    else:
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,neighbours)
    return img



def deskew(im):
    max_skew=50
    height, width = im.shape
    if(height>width):
        width = im.shape[0]
        height = im.shape[1]
        
    im_bw = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    lines = cv2.HoughLinesP(
        im_bw, 1, np.pi / 180, 200, minLineLength=width / 3, maxLineGap=width / 100
    )
    line_image = np.copy(im) * 0 
    # Collect the angles of these lines (in radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)    
        angles.append(np.arctan2(y2 - y1, x2 - x1))
    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2

    # Filter the angles to remove outliers based on max_skew
    if landscape:
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]
    
    if len(angles) < 5:
        # Insufficient data to deskew
        return im

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))

    # If this is landscape image, rotate the entire canvas appropriately
    if landscape:
        if angle_deg < 0:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90
    # Rotate the image by the residual offset
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    im = cv2.warpAffine(im, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return im


def morphology(img,kernel_size=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    if(len(img.shape)>2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = closing
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)      
    else:
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = closing
    return img


def preprocess_img(img,show_steps=0,show_size=1,digital=1):
    
    if(digital != 1):
        gaussian_kernel = 9
        block_size = 95
        neighbours = 4
    else:
        gaussian_kernel = 1
        block_size = 95
        neighbours = 1
        

    # Gaussian
    if(digital != 1):
        img = gaussian(img,gaussian_kernel)
        if(show_steps !=0 ):
            show(img,show_size,"Gaussian Filtered")

    # local thresh
    img = binarize(img,block_size,neighbours)
    if(show_steps !=0 ):
        show(img,show_size,"Binarized")
        
    # rotate
    img = deskew(img)
    if(show_steps !=0 ):
        show(img,show_size,"Rotated")

    # morphology to remove noise
    if(digital != 1):
        img = morphology(img,4)
        if(show_steps !=0 ):
            show(img,show_size,"Morphology")

    return img



def run_length_encoding(img):
        
    rows = img.shape[0]
    cols = img.shape[1]

    black_runs = []
    white_runs = []
    for i in range(cols):        
        black_run = 0
        white_run = 0

        for j in range(rows):
            if (img[j][i] == 255 and black_run == 0):
                white_run += 1
            if (img[j][i] == 0 and white_run != 0):
                white_runs.append(white_run)
                white_run = 0
            if (img[j][i] == 0 and white_run == 0):
                black_run += 1
            if (img[j][i] == 255 and black_run != 0):
                black_runs.append(black_run)
                black_run = 0

    return white_runs,black_runs


def get_common_run(rle):
    
    counter = np.zeros(5000)
    for i in range(len(rle)):
        counter[rle[i]] += 1
        
    return np.argmax(counter)

def remove_lines_non_digital(img,staff_thickness,staff_space):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(cols):
        j = 0
        while j < rows:
            if img[j][i] == 0 and j + staff_thickness < rows:
                if np.sum(img[j:j+staff_thickness,i]) != 0:
                    img[j:j+staff_thickness,i] = 255
            j+=1
    kernel = np.ones((staff_thickness+1,2),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)
    return img


def get_horizontal_projection(img,show_steps=1,show_size=1):
    if(len(img.shape)>2):
        img = rgb2gray(img)
    proj = np.sum(img,axis=1).astype(int)
    
    max_line = np.amax(proj)
    staffs = proj == (max_line)
    m = np.max(proj)
    w = 500
    result = np.zeros((proj.shape[0],500))

    if show_steps == 1 :
        for row in range(img.shape[0]):
            cv2.line(result, (0,row), (int(proj[row]*w/m),row), (255,255,255), 1)
        show(result,show_size,"Horizontal Projection")
    return proj



def identify_lines(img,show_steps=1,show_size=1):
    """ 
    Gets horizontal projection and extracts five lines from it
    :param
    show_steps: wether show steps or not 0 dont show , 1 show
    show_size : the size of the shown image a value to factor in x and y
    """
    if(len(img.shape)>2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(img)    
    rows_sums = get_horizontal_projection(inverted,show_steps,show_size)
    
    lines_pixels = []
    for i in range(len(rows_sums)):
        if rows_sums[i] > img.shape[1]*0.4*255:
            lines_pixels.append(i)

    lines = []
    max_thickness = 0
    i = 0

    while i < len(lines_pixels):
        current = lines_pixels[i] - 1
        lines.append(lines_pixels[i])
        counter = 0
        while(lines_pixels[i] == current + 1):
            current = lines_pixels[i]
            counter += 1
            i += 1
            if i >= len(lines_pixels):
                break
        if counter > max_thickness:
            max_thickness = counter

    return img,lines,max_thickness




def remove_lines_digital(img,show_steps=1,show_size=1):
    _,lines,thickness = identify_lines(img,show_steps,show_size)
    test_value = 255*2
    pixels_tested = thickness
    
    for line in lines:
        for i in range(2,img.shape[1]-2):
            sum_down = 0
            for j in range(pixels_tested):
                sum_down += img[line+j][i]
            if(sum_down < test_value):
                img[line][i] = 255
                for j in range(pixels_tested):
                    img[line+j][i] = 255
    
    kernel = np.ones((thickness+1,1),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)
    return img,lines



def remove_staff_lines(img,show_steps=1,show_size=1,digital=1):

    white_rle,black_rle = run_length_encoding(img)
    staff_thickness = get_common_run(black_rle) + 4
    staff_space = get_common_run(white_rle) 
    _,lines,_ = identify_lines(img,0)
    rows = []
    i = 0

    if(digital ==1):
        img,lines = remove_lines_digital(img,show_steps,show_size)
        while i < len(lines):
            if lines[i]-staff_space*3 < 0 and lines[(4+i)] + staff_space*3 > img.shape[0]:
                print("A")
                rows.append(img[0:img.shape[0]-1,:])
            elif lines[(4+i)] + staff_space*3 > img.shape[0]:
                print("B")
                print(lines[i]-staff_space*3)
                rows.append(img[int(lines[i]-staff_space*3):img.shape[0]-1,:])
            elif  lines[i]-staff_space*3 < 0 :
                print("C")
                rows.append(img[0:lines[(4+i)]+staff_space*3,:])
            else:
                rows.append(img[int(lines[i]-staff_space*3):int(lines[(4+i)]+staff_space*3),:])
            i += 5
            
    else:
        img = remove_lines_non_digital(img,staff_thickness,staff_space)
        rows.append(img)
    if show_steps == 1 :
        for row in rows:
            show(row,show_size,"Removed Lines")
        
    return rows,staff_space,staff_thickness,lines



def get_notes(img):
    """ 
    Segements the notes
    :param
    img: the img to extract notes from
    """

    if(len(img.shape)>2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(img)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    proj = np.sum(thresh,axis=0)
    avg = np.mean(proj)
    start = 0
    end = 0
    started = False
    segments = []
    for i in range(len(proj)):
        if proj[i] > int(avg/50) and started == False:
            started = True
        if proj[i] < int(avg/50) and started == True:
            started = False
            end = i
            if img[:,start:end] is not None:
                segments.append(img[:,start:end])
            start = i

    return segments



def segment_img(img,digital=1,show_steps=1,show_size=1):
    if show_steps == 1 :
        show(img,show_size,"Image Row")
    if digital == 1:
        segments = get_notes(img)
    else:
        segments = better_segment(img)
    return segments


def better_segment(img):

    original = img.copy()
    blurred = cv2.GaussianBlur(img, (1, 1), 0)
    canny = cv2.Canny(blurred, 50, 255, 1)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=2)

    
    # Find contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and filter for ROI
    image_number = 0
    order_list = []
    images = []
    area = 0 
    for c in cnts:
        area += cv2.contourArea(c)
    area /= (len(cnts)+5)  
    for c in cnts:
        if(cv2.contourArea(c) > area):
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
            ROI = original[y:y+h, x:x+w]
            images.append(ROI)
            order_list.append(x+y/1000)
            image_number += 1

    segments = [x for _,x in sorted(zip(order_list,images))]
    return segments

def check_full_symmetry(image):
    notImg = cv2.bitwise_not(image)
    if notImg is None:
        return True
    avg = np.sum(notImg)/4
    thresh = avg - int(avg/10)
    rows = notImg.shape[0]
    cols = notImg.shape[1]
    q1 = np.sum(notImg[0:int(rows/2),0:int(cols/2)])
    q2 = np.sum(notImg[0:int(rows/2),int(cols/2):cols])
    q3 = np.sum(notImg[int(rows/2):rows,0:int(cols/2)])
    q4 = np.sum(notImg[int(rows/2):rows,int(cols/2):cols])
    #print("Q1 is " ,q1, " and Q2 is ", q2, " and Q3 is ", q3, " and Q4 is ", q4)
    #print("Avg is ", avg)
    if q1  >= thresh and q2 >= thresh and q3 >= thresh and q4 >= thresh:
        return True
    else: 
        return False


def get_v_lines(img): #This function is for use in classifications only
    
    gray = img

    blur_gray = gray
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = img.shape[0]-int(img.shape[0]/8)  # minimum number of pixels making up a line
    max_line_gap = img.shape[0]-int(img.shape[0]/8)  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    if lines is None:
        return 0
    return len(lines)


def check_half_symmetry(image):
    notImg = cv2.bitwise_not(image)
    rows = notImg.shape[0]
    cols = notImg.shape[1]
    topSum =  np.sum(notImg[0:int(rows/2),:])
    bottomSum = np.sum(notImg[int(rows/2):rows,: ])
    leftSum =np.sum(notImg[:, 0:int(cols/2)])
    rightSum = np.sum(notImg[:, int(cols/2): cols]) 
    topSum = np.int64(topSum)
    bottomSum = np.int64(bottomSum)
    leftSum = np.int64(leftSum)
    rightSum = np.int64(rightSum)
    if abs(leftSum - rightSum) <= int(leftSum/10):
        
        if abs(topSum - bottomSum) > int(bottomSum/5):
            return "double flat"
        else:
            return "natural"
    
    else:
        return "flat"


def crop_segment(segment):
    mask_inv = cv2.bitwise_not(np.copy(segment))
    ver_sum = np.sum(mask_inv,axis=1)
    v_start = 0
    v_end = 0
    for i in range(len(ver_sum)):
        if(ver_sum[i] > 0 and v_start ==0):
            v_start = i
        if(ver_sum[i] == 0 and v_start != 0):
            v_end = i
            break
    if(v_end == 0):
        v_end = len(ver_sum) - 1
    
    hor_sum = np.sum(mask_inv,axis=0)
    h_start = 0
    h_end = 0
    for i in range(len(hor_sum)):
        if(hor_sum[i] > 0 and h_start ==0):
            h_start = i
        if(hor_sum[i] == 0 and h_start != 0):
            h_end = i
            break
    if(h_end == 0):
        h_end = len(hor_sum) - 1

    return segment[v_start:v_end,h_start:h_end]


def getNoteHead(segment):
    rows, cols = segment.shape
    for row in rows:
        for col in cols:
            if (segment[row, shape] == 1):
                return row,shape



def classifyNotes(segment, staff_space, staff_thickness,lines):
    croppedSegment = crop_segment(np.copy(segment))
    croppedSegment = cv2.bitwise_not(np.copy(croppedSegment)) 
    mask_inv = cv2.bitwise_not(np.copy(segment))
    rows, cols = croppedSegment.shape
    
    #show(croppedSegment)
    topHalf = np.sum(croppedSegment[0:int(rows/2), :])
    bottomHalf = np.sum(croppedSegment[int(rows/2):rows, :])
    topHalf = np.int64(topHalf)
    bottomHalf = np.int64(bottomHalf)
    if (topHalf - bottomHalf) > 0:
        lines.sort(reverse = True)
        for index, line in enumerate(lines):
            distance = 0
            found = False
            if index == 0:
                for i in range(line, mask_inv.shape[0] , 1):
                    su = np.sum(mask_inv[i])
                    if su > 0:
                        found = True
                        distance += 1
                    else:
                        break
                    if found == False:
                        continue
                    if abs(distance - staff_space) <= int(staff_space/3):
                        return "b"
            if index == 1:
                return None
        return None
    else:
        where = 0
        for index, line in enumerate(lines):
            if index == 1:
                distance = 0
                found = False
                for i in range(line, lines[index-1] , -1):
                    su = np.sum(mask_inv[i])
                    if su > 0:
                        found = True
                        distance += 1
                    else:
                        break
                if found == False:
                    continue
                if distance < staff_space and distance > int(staff_space/3):
                    #print("line is ", line)
                    #print("distance is ",distance," and space ", staff_space)
                    return "e"
                elif distance <2 and distance > -2:
                    return "d"
                print("line is ", line)
            
            if index == 2:
                distance = 0
                found = False
                for i in range(line, lines[index-1] , -1):
                    su = np.sum(mask_inv[i])
                    if su > 0:
                        found = True
                        distance += 1
                    else:
                        break
                if found == False:
                    continue
                if distance < staff_space and distance > int(staff_space/3):
                    #print("line is ", line)
                    #print("distance is ",distance," and space ", staff_space)
                    return "c"
                
            if index == 0:
                #print("index is now 0")
                distance = 0
                found = False
                for i in range(line, 0, -1):
                    su = np.sum(mask_inv[i])
                    if su > 0:
                        found = True
                        distance += 1
                    else:
                        
                        break
                if found == False:
                    continue
                if distance < staff_space and distance > int(staff_space/3):
                    #print("line is ", line)
                    #print("distance is ",distance," and space ", staff_space)
                    return "g"
                elif abs(distance - staff_space) <= int(staff_space/2):
                    #print("line is ", line)
                    #print("distance is ",distance," and space ", staff_space)
                    return "a"
                elif distance <2 and distance > -2:
                    return "f"



def classify_old(segment, staff_space, staff_thickness,lines):
        rows = segment.shape[0]
        mask_inv = cv2.bitwise_not(np.copy(segment))
        if(np.sum(mask_inv) < 1020):
            return None
        sum_ver = np.sum(mask_inv,axis=1)
        if (np.count_nonzero(sum_ver) >= staff_space*4):
            
            #TODO: classify note here
            note = classifyNotes(segment, staff_space, staff_thickness,lines)
            if note == None:
                return None
            else:
                return note

        else:
            segment = crop_segment(segment)
            if (check_full_symmetry(segment)):
                if get_v_lines(segment) > 2 :
                    return "#"
                else:
                    return "##"
            else:
                classification = check_half_symmetry(segment)
                if (classification  == None):
                    return None
                else:
                    return classification



def get_DU(segment):
    segment = cv2.bitwise_not(np.copy(segment))
    ver_sum = np.sum(segment,axis=1)
    counter = 0
    for i in range(len(ver_sum)):
        if ver_sum[i] == 0:
            counter+=1
        else :
            break
    return counter
def get_ADAU(segment):
    segment = crop_segment(segment)
    segment = cv2.bitwise_not(np.copy(segment))
    AD = np.sum(segment[0:int(segment.shape[0]/2),:])
    AU = np.sum(segment[int(segment.shape[0]/2):-1,:])
    if AU == 0: 
        ADAU = 0
    else:
        ADAU = (AD/AU)
    return ADAU
def get_ALAR(segment):
    segment = crop_segment(segment)
    segment = cv2.bitwise_not(np.copy(segment))
    AL = np.sum(segment[:,0:int(segment.shape[1]/2)])
    AR = np.sum(segment[:,int(segment.shape[1]/2):-1])
    if AR == 0: 
        ALAR = 0
    else:
        ALAR = (AL/AR)
    return ALAR



def classify(segments,f,classifier,show_steps=1,show_size=1):
    f.write("[ ")
    
    flat = False
    double_flat = False
    hashtag = False
    double_hashtag = False
    
    for i in range(len(segments)):
        mask_inv = cv2.bitwise_not(crop_segment(segments[i]))
        if(mask_inv is None):
            continue
        if show_steps == 1 :
            show(segments[i],show_size,"Segment "+ str(i+1))
        sum_ver = np.sum(mask_inv,axis=1)
        sum_hor = np.sum(mask_inv,axis=0)
        X_test = np.array([(np.sum(np.count_nonzero(sum_ver))/staff_space)*100,
                           (np.sum(np.count_nonzero(sum_hor))/staff_space)*100,
                           (get_ADAU(segments[i])/staff_space)*100 ,
                           (get_ALAR(segments[i])/staff_space)*100,
                           (get_DU(segments[i])/staff_space)*100])
        y_pred = classifier.predict([X_test])
        if y_pred[0] != "CLEF" and y_pred[0] != "BARLINE":
            if y_pred[0] == "&":
                flat = True
                double_flat = False
                hashtag = False
                double_hashtag = False
            elif y_pred[0] == "&&":
                flat = False
                double_flat = True
                hashtag = False
                double_hashtag = False                
            elif y_pred[0] == "#":
                flat = False
                double_flat = False
                hashtag = True
                double_hashtag = False
            elif y_pred[0] == "##":
                flat = False
                double_flat = False
                hashtag = False
                double_hashtag = True                
            elif y_pred[0] == "NATURAL":
                pass
            elif flat == True:
                flat = False
                double_flat = False
                hashtag = False
                double_hashtag = False
                temp_str =""
                for char in (y_pred[0])[1:]:
                    temp_str += char
                f.write((y_pred[0])[0]+"&"+temp_str+" ")
            elif double_flat == True:
                flat = False
                double_flat = False
                hashtag = False
                double_hashtag = False
                temp_str =""
                for char in (y_pred[0])[1:]:
                    temp_str += char
                f.write((y_pred[0])[0]+"&&"+temp_str+" ")
            elif hashtag == True:
                flat = False
                double_flat = False
                hashtag = False
                double_hashtag = False
                temp_str =""
                for char in (y_pred[0])[1:]:
                    temp_str += char
                f.write((y_pred[0])[0]+"#"+temp_str+" ")
            elif double_hashtag == True:
                flat = False
                double_flat = False
                hashtag = False
                double_hashtag = False
                temp_str =""
                for char in (y_pred[0])[1:]:
                    temp_str += char
                f.write((y_pred[0])[0]+"##"+temp_str+" ")
            else:
                f.write(y_pred[0]+" ")
    f.write("]\n")
    f.flush()
        
def threshold_img(img):
    img[img<180] = 0
    img[img>=180] = 255
    return img


def initialize_KNN():
    url = "train.csv"
    # Assign colum names to the dataset
    names = ['H', 'W', 'ADAU', 'ALAR', 'DU', "CLASS"]

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(url, names=names)
    dataset.head()
    X = dataset.iloc[1:-1,0:5].values
    y = dataset.iloc[1:-1, 5].values
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(X, y)
    
    return classifier

def read_image(input_path,input_img):
    img = cv2.imread(input_path+input_img)
    digital = digital_or_not(input_img)
    return img,digital

# Initialize parser

parser = argparse.ArgumentParser()

parser.add_argument("inputfolder", help = "Input File")
parser.add_argument("outputfolder", help = "Output File")

args = parser.parse_args()


show_steps = 0
show_size = 1


input_path = args.inputfolder
output_path = args.outputfolder
classifier = initialize_KNN()
images = []
for file in os.listdir(input_path):
    images.append(file)
images.sort()
for input_img in images:
    output_txt = input_img.split(".")[0] + ".txt"
    img, digital = read_image(input_path,input_img)
    f = open(output_path+output_txt,"w")
    img = preprocess_img(img,show_steps,show_size,digital)
    img = threshold_img(img)
    rows,staff_space,staff_thickness,lines = remove_staff_lines(img,show_steps,show_size,digital)
    for row in rows:
        segments = segment_img(row,digital,show_steps=1,show_size=1)
        classify(segments,f,classifier,show_steps,show_size)
    f.close()
