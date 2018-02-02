#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 23:28:29 2017
On my honor, I have not given, nor recieved, nor witnessed any unauthorized 
assistance on this work.
I worked alone, and consulted the following resources:
    https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html
    http://uberhip.com/python/image-processing/opencv/2014/10/26/warping-brien/
    https://www.learnopencv.com/homography-examples-using-opencv-python-c/
    https://gist.github.com/nikgens/1a129d620978a4abc6a9a30f5f66e0d3
    
@author: Hannah Holman (hholman@rollins.edu)
"""
import cv2
import numpy as np

refPt = []
destPt = []
 
def get_src_coordinates(event, x, y, flags, param):
    """ Method to work with mouse callback in order to collect a set of 
        user specified click points on source image.
        Appends points to global array variable, refPt
        
        Args:
            event: mouse event
            x, y: mouse click coordinates
            flags: None
            param (np.ndarray): input image    
    """
    #instanitate global variables in local context
    global refPt
    img = param.copy()
    
    #When user clicks:
    if event == cv2.EVENT_LBUTTONDOWN:
        #Append click coordinates to global array variable
        refPt.append((x, y))
        print(len(refPt), "source points selected")
        
        #Draw circle to show click point
        img = cv2.circle(img, refPt[-1], 2, (255, 255, 0), 2)
        cv2.imshow('image', img)
        temp = refPt    

        #Once four points have been selected, organize array
        if len(refPt) == 4:
            temp = np.array(temp)

            #top left and bottom right
            pt_sum = temp.sum(axis = 1)
            refPt[0] = temp[np.argmin(pt_sum)]
            refPt[2] = temp[np.argmax(pt_sum)]
            
            #top right and bottom left
            diff = np.diff(temp, axis = 1)
            refPt[1] = temp[np.argmin(diff)]
            refPt[3] = temp[np.argmax(diff)]
            return
        
def get_dest_coordinates(event, x, y, flags, param):
    """ Method to work with mouse callback in order to collect a set of 
        user specified click points on destination image.
        Appends points to global array variable, destPt
        
        Args:
            event: mouse event
            x, y: mouse click coordinates
            flags: None
            param (np.ndarray): input image    
    """
    #Instantiate global variables in local context
    global destPt
    img = param.copy()
    
    #When user clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        #Append click coordinates to global array variable
        destPt.append((x, y))
        print(len(destPt), "destination points selected")

        #Draw circle to show click point
        img = cv2.circle(img, destPt[-1], 2, (255, 255, 0), 2)
        cv2.imshow('image 2', img)
        temp = destPt
        
        #When four points have been selected, organize array and return
        if len(destPt) == 4:
            temp = np.array(temp)

            #top left and bottom right
            pt_sum = temp.sum(axis = 1)
            destPt[0] = temp[np.argmin(pt_sum)]
            destPt[2] = temp[np.argmax(pt_sum)]
            
            #top right and bottom left
            diff = np.diff(temp, axis = 1)
            destPt[1] = temp[np.argmin(diff)]
            destPt[3] = temp[np.argmax(diff)]
            return
    
def transform_upright(image, pts):
    """Transforms the perspective of an image to a "face-on" view
    
       Args: 
           image (numpy.ndarray): an input image to be transformed
       
           pts (array): an array of points on an input image to be used as source points
               for a perspective transform
            
       Returns:
           cpy (numpy.ndarray): a transformed image
    """
    #Instantiate global variables in local context
    global refPt
    
    #Retrieve the min and max values in the reference points array
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(refPt)
    
    cpy = image.copy()
    pts = np.float32(pts)

    #Use max value to create a large enough desination image with square shape    
    dest_pts = np.float32([[0, 0], [maxVal, 0], [maxVal, maxVal], [0, maxVal]])

    M = cv2.getPerspectiveTransform(pts, dest_pts)
    
    maxVal = np.int32(maxVal)
    cpy = cv2.warpPerspective(image, M, (maxVal, maxVal))

    return cpy
    
def transform_homography(img_src, img_dst, pts1, pts2):
    """Warps a source image to a destination image based on a set of source
       and destination points
       
       Args:
           img_src (numpy.ndarray): source image to be warped
           img_dest (numpy.ndarray): destination image to be warped to
           pts1 (array): array of coordinates to be used to compute homography
           pts2 (array): array of coordinates to be used to compute homography
          
       Returns:
           image_dst (numpy.ndarray): the final image with warped image on top
    """
    image_src = img_src.copy()
    image_dst = img_dst.copy()
    mask = image_dst.copy()

    h, status = cv2.findHomography(pts1, pts2)
    
    #create masks of both images around the portion of the image to be merged
    mask = cv2.warpPerspective(image_src, h, (image_dst.shape[1], image_dst.shape[0]))  
    cv2.fillConvexPoly(image_dst, np.int32(pts2), 0, cv2.LINE_AA)
    
    image_dst = image_dst + mask

    #return image_dst
    return image_dst

def foreground_warp(img):
    """Takes an input image and transforms the perspective to a face-on view, 
       and fills an array of points with the four corners fo the transformed 
       image
       
       Args:
           img (numpy.ndarray): image to be transformed
         
       Returns:
           img_transform (numpy.ndarray): transformed image
           src_pts_transform (array of points): array containign the four 
               corners of the output image
    """
    global refPt
    src_pts = []
    
    img_src = img.copy()
    src_pts = refPt
    src_pts = np.float32(src_pts)
    refPt = np.array(refPt)
    
    img_transform = transform_upright(img_src, src_pts)
    
    #build an array with four corner points of transformed image to be used
    #when computing homography with final dest image later
    h, w = img_transform.shape[:2]
    src_pts_transform = np.empty((0,2),dtype=np.float32)
    src_pts_transform = np.append(src_pts_transform, [(0,0)], axis=0)
    src_pts_transform = np.append(src_pts_transform, [(w - 1, 0)], axis=0)
    src_pts_transform = np.append(src_pts_transform, [(w - 1, h - 1)], axis=0)
    src_pts_transform = np.append(src_pts_transform, [(0, h - 1)], axis=0)

    return img_transform, src_pts_transform

def warp_to_background(img):
    """Creates an array of destination points and copies an input image in
       order to preserve original for later manipulation
       
       Args 
           img (numpy.ndarray): image to be preserved
       
       Returns:
           img_dest (numpy.ndarray): copied image to be used as destination for
               perspective transformation
            
           dest_pts (np.float32): list of destination points for computing
               homography          
    """
    dest_pts = []

    if len(dest_pts) == 0:
        img_dest = img.copy()
        dest_pts = destPt
        dest_pts = np.float32(dest_pts)
        cv2.destroyAllWindows()
        return img_dest, dest_pts
    else:
        print("invalid")
        cv2.destroyAllWindows()
        return None, None
    
################################################################
 
#process first image
while (1):

    src = cv2.imread('comic1.jpg')
    #src = cv2.resize(src, (1500,900))
    dst = cv2.imread('billboard.jpg')

    cv2.namedWindow('image')
    cv2.imshow('image', src)
    
    #set the "on click" to get the source coordinates, param is the source image
    cv2.setMouseCallback('image', get_src_coordinates, param = src)
    
    #Wait for user to select points and press a key. If keypress = 'b', go back
    #and erase selected points
    if cv2.waitKey(0) == 98:
        refPt = []
        print("Source point selection restarted")
    elif cv2.waitKey(0) and len(refPt) == 4:
        img_transform, src_pts_transform = foreground_warp(src)
        cv2.destroyAllWindows()
        break
    elif len(refPt) > 4:
        print("Error: Too many points. Points set to 0, please start again")
        refPt = []
    else:
        print("Source points selected: ", len(refPt), 
              "- Please select", 4 - len(refPt), "more points")

#process second image
while (1):

    cv2.namedWindow('image 2')
    cv2.imshow('image 2', dst)
    
    #set the "on click" to get the destination coordinates, param is the dest image
    cv2.setMouseCallback('image 2', get_dest_coordinates, param = dst)
    
    #Wait for user to select points and press a key. If keypress = 'b', go back
    #and erase selected points
    if cv2.waitKey(0) == 98:
        destPt = []
        print("Destination point selection restarted")
    elif cv2.waitKey(0) and len(destPt) == 4:
        img_dest, dest_pts = warp_to_background(dst)
        break
    elif len(destPt) > 4:
        print("Error: Too many points. Points set to 0, please start again")
        destPt = []
    else:
        print("Destination points selected:", len(destPt), 
              "- Please select", 4 - len(destPt), "more points")
        

#find the homography and transform images
output = transform_homography(img_transform, img_dest, src_pts_transform, dest_pts)
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
 