# USAGE
# python text_detection_video.py --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import os
import argparse
import imutils
import time
import cv2

# Credit to Adrian Rosebrock @ pyimagesearch.com, for the text detection tutorial
# https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/

################################################################
################################################################
################################################################

def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects 		= []
	confidences = []

	for y in range(0, numRows):
		scoresData 	= scores[0, 0, y]
		xData0 		= geometry[0, 0, y]
		xData1 		= geometry[0, 1, y]
		xData2 		= geometry[0, 2, y]
		xData3 		= geometry[0, 3, y]
		anglesData 	= geometry[0, 4, y]

		for x in range(0, numCols):
			if scoresData[x] < 0.6:
				continue

			(offsetX, offsetY) 	= (x * 4.0, y * 4.0)
			angle 				= anglesData[x]
			cos 				= np.cos(angle)
			sin 				= np.sin(angle)
			h 					= xData0[x] + xData2[x]
			w 					= xData1[x] + xData3[x]
			endX 				= int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY 				= int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX 				= int(endX - w)
			startY 				= int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return (rects, confidences)

################################################################
################################################################
################################################################

def make_dir():
	for a in os.listdir("_vid"):
		dir_name = str(a).split(".mp4")
		dir_name = str(dir_name[0])
		dir_name = dir_name.lower().replace("'", "").replace('"', '').replace("?", "").replace("@", "").replace(".", "").replace("!", "").replace("#", "").replace("$", "").replace("+", "").replace("=", "").replace("%", "").replace("^", "").replace("(", "").replace(")", "").replace("&", "").replace("*", "").replace(";", "").replace(":", "").replace("/", "").replace(",", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("\\", "").replace("|", "").replace("`", "").replace("~", "").replace(" ", "_").replace("-----", "").replace("----", "").replace("---", "").replace("--", "").replace("-", "_").replace("___", "_").replace("__", "_")
		print(dir_name)
		os.system("mkdir _exports/" + str(dir_name))

################################################################
################################################################
################################################################

def init_frame_grab():
	text_display 	= False
	new_shot		= True

	for a in os.listdir("_vid"):
		filename 		= str(a).split(".mp4")
		filename 		= str(filename[0])
		filename 		= filename.lower().replace("'", "").replace('"', '').replace("?", "").replace("@", "").replace(".", "").replace("!", "").replace("#", "").replace("$", "").replace("+", "").replace("=", "").replace("%", "").replace("^", "").replace("(", "").replace(")", "").replace("&", "").replace("*", "").replace(";", "").replace(":", "").replace("/", "").replace(",", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("\\", "").replace("|", "").replace("`", "").replace("~", "").replace(" ", "_").replace("-----", "").replace("----", "").replace("---", "").replace("--", "").replace("-", "_").replace("___", "_").replace("__", "_")
		(W, H) 			= (None, None)
		(newW, newH) 	= (320, 320)
		(rW, rH) 		= (None, None)
		layerNames 		= ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

		print("[INFO] loading EAST text detector...")
		print(a)
		net 				= cv2.dnn.readNet("frozen_east_text_detection.pb")
		vs 					= cv2.VideoCapture("./_vid/" + str(a))
		z, current_frame 	= vs.read()
		previous_frame 		= current_frame
		fps 				= FPS().start()
		frame_count 		= 0

		while True:
			if frame_count%100000 == 0:
				frame = vs.read()
				frame = frame[1]

				if frame is None:
					break

				frame 				= imutils.resize(frame, width=1000)
				orig 				= frame.copy()
				current_frame_gray 	= cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
				previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
				curr_sum 			= sum(sum(current_frame_gray))
				prev_sum			= sum(sum(previous_frame_gray))
				avg_diff			= curr_sum/prev_sum
				frame_diff 			= cv2.absdiff(current_frame_gray,previous_frame_gray)

				if W is None or H is None:
					(H, W) 	= frame.shape[:2]
					rW 		= W / float(newW)
					rH 		= H / float(newH)

				frame 					= cv2.resize(frame, (newW, newH))
				blob 					= cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
				net.setInput(blob)
				(scores, geometry) 		= net.forward(layerNames)
				(rects, confidences) 	= decode_predictions(scores, geometry)
				boxes 					= non_max_suppression(np.array(rects), probs=confidences)
				try:
					if sum(confidences)/len(confidences) > 0.98:
						text_display 	= True
					else:
						text_display 	= False
				except:
					text_display 		= False
				if text_display == True and new_shot == True:
					print("-"*50)
					print("-"*50)
					print(sum(confidences)/len(confidences))
					print(avg_diff)
					for (startX, startY, endX, endY) in boxes:
						startX 	= int(startX * rW)
						startY 	= int(startY * rH)
						endX 	= int(endX * rW)
						endY 	= int(endY * rH)

						cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

					fps.update()
					cv2.imwrite("_exports/" + str(filename) + "/" + str(filename) + "_" + str(frame_count) + ".jpg", orig)
					cv2.imshow("Text Detection", orig)
					new_shot 	= False
					key 		= cv2.waitKey(1) & 0xFF

					if key == ord("q"):
						break
				if avg_diff > 1.04 or avg_diff < 0.94:
					new_shot 		= False
				else:
					new_shot 		= True
				previous_frame 		= current_frame.copy()
				z, current_frame 	= vs.read()
			frame_count+=1
		fps.stop()
		os.system("mv _vid/" + str(a) + " ./__scanned_vids/")
		print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		vs.release()
		cv2.destroyAllWindows()

################################################################
################################################################
################################################################

init_frame_grab()