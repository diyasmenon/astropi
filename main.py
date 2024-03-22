#--------------------------------
#       Astro Pi Competition    #
#                               #
#         Team: CodeByte        #
#                               #
#           18/02/24            #
#                               #
#     Written by: Diya Menon    #
#--------------------------------


from exif import Image
from datetime import datetime, timedelta
from time import sleep
import cv2
import math #work out hyp between points
from picamera import PiCamera
import numpy #to work out outliers
import os #to find an img

start_time = datetime.now() #starting time of program

#functions------------------------------------------------------------------------------------------
def get_time(image): #passes an image to get the time it was taken at
    with open(image, 'rb') as image_file: #closes the image after use
        img = Image(image_file) #creates an Image instance
        #we need datetime_original - saved as string - converted to datetime object for calcuations
        time_string = img.get("datetime_original") #gets the time string
        time = datetime.strptime(time_string, '%Y:%m:%d %H:%M:%S') #formats the info as a date time object
    return time
            
def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1) #gets the time of the first pic passed in
    time_2 = get_time(image_2) #gets the time of the second pic passed in
    time_difference = time_2 - time_1 #calcs time diff
    return time_difference.seconds #returns time diff in seconds
    
def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0) #makes an image object so it can be processed
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv #returns the objects

def calculate_features(image_1, image_2, feature_number): #3rd param is max features to look out for
    orb = cv2.ORB_create(nfeatures = feature_number) #makes sure the keypoints/descriptors are w the same max feature number
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None) #keypoints/descriptors for img 1
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None) #keypoints/descriptors for img 2
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2 #returns these values to be used in another function

def calculate_matches(descriptors_1, descriptors_2): #takes the two sets of descriptors and tries to find matches by brute force
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches): #creates a window w lines showing matching features
    #draw lines between the keypoints where the descriptors match
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    #images can be resized and shown, side by side the screen, with the lines drawn between the matches
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize) #makes a new window w the image and lines
    #wait until a key is pressed, and then close the image
    cv2.waitKey(0)
    cv2.destroyWindow('matches')

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = [] #store the coordinates of each matching feature in each of the images
    coordinates_2 = [] #same as above but for img 2
    # list of matches contains many OpenCV match objects - iterate through the list to find the coordinates of each match on each image
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt #gets coords for img 1
        (x2,y2) = keypoints_2[image_2_idx].pt #gets coords for img 2
        coordinates_1.append((x1,y1)) #adds coords to list for img 1
        coordinates_2.append((x2,y2)) #adds coords to list for img 2
    return coordinates_1, coordinates_2 #returns the 2 lists

def calculate_mean_distance(coordinates_1, coordinates_2): #calculate the average distance between matching coordinates
    all_distances = [] #store the sum of all the distances between coordinates
    #zip function will take items from two lists and join them together
    merged_coordinates = list(zip(coordinates_1, coordinates_2)) #zip the two lists, and then converts the zipped object back to a list
    #iterate over the merged_coordinates and calculate the differences between the x and y coordinates in each image.
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference) #calculate the distance between the two points
        all_distances.append(distance) #adds distance to the all_distances variable  
    
    all_distances = remove_outliers(all_distances, 2)
    return sum(all_distances) / len(all_distances) #dividing all_distances by the number of feature matches

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    #The Ground Sample Distance (GSD) is given in centimeters/pixels
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

def remove_outliers(arr, n): #[og code]
    elements = numpy.array(arr)
    mean = numpy.mean(elements, axis=0)
    sd = numpy.std(elements, axis=0)
    final_list = [x for x in arr if (x > mean - n * sd)]
    final_list = [x for x in final_list if (x < mean + n * sd)]
    return final_list

def write_to_file(speed):
    estimate_kmps_formatted = "{:.4f}".format(speed) #makes it 5sf
    estimate_kmps_formatted += ' km/s'
    file_path = "result.txt"  #correct file name
    with open(file_path, 'w') as file: #closes file after opening
        file.truncate(0) #removes everything from the file
        file.write(str(estimate_kmps_formatted)) #writes resulting speed to file as req.

def delete_photo(img_num):
    os.remove(f'image{img_num - 2}.jpg') #deletes img w that name

#main code---------------------------------------------------------------------------------------
cam = PiCamera() #creates an instance of the camera
cam.resolution = (4096, 3040) #resolution of cam in pixels
total_speeds = [] #list of all the calculated speeds

now_time = datetime.now()

cam.capture("image0.jpg") #starts image sequencing on 0
pic_num = 1

while (now_time < start_time + timedelta(minutes=9.25)): #while the program has been running for < 9.25 mins
   
    try:   
        image_1 = f"image{(pic_num - 1)}.jpg" #uses the prev image taken as img 1
        
        cam.capture(f"image{pic_num}.jpg") #takes a new img for the img 2
        image_2 = f"image{pic_num}.jpg" #passes through the img for img 2
        pic_num += 1 #indexes

        time_difference = get_time_difference(image_1, image_2) # Get time difference between images
        image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects
        keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # Get keypoints and descriptors
        matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors
        #display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) # Display matches
        coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
        average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
        speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference) #for 2 images
        total_speeds.append(speed) #adds this speed to the list

        write_to_file(sum(total_speeds)/len(total_speeds)) #rewrites new avg speed to txt file
        delete_photo(pic_num) #deletes oldest img
        
        now_time = datetime.now() #gets the current time now
        
    except:
        now_time = datetime.now() #gets the current time now

total_speeds_new = remove_outliers(total_speeds, 1) #makes a new list of speeds without outliers

if len(total_speeds_new) == 0:
    write_to_file(sum(total_speeds)/len(total_speeds)) #creates final text file with all the data, as there is not enough to find outliers
else:
    write_to_file(sum(total_speeds_new)/len(total_speeds_new)) #creates final text file, without outliers


#print("completed successfully")
#print((datetime.now() - start_time).seconds) #how long the program ran for (in seconds)
cam.close() #closes cam