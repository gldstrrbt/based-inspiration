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

def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			# if scoresData[x] < args["min_confidence"]:
			if scoresData[x] < 0.6:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-east", "--east", type=str, required=True,
# 	help="path to input EAST text detector")
# ap.add_argument("-v", "--video", type=str,
# 	help="path to optinal input video file")
# ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
# 	help="minimum probability required to inspect a region")
# ap.add_argument("-w", "--width", type=int, default=320,
# 	help="resized image width (should be multiple of 32)")
# ap.add_argument("-e", "--height", type=int, default=320,
# 	help="resized image height (should be multiple of 32)")
# args = vars(ap.parse_args())


def make_dir():
	for a in os.listdir("_vid"):
		dir_name = str(a).split(".mp4")
		dir_name = str(dir_name[0])
		dir_name = dir_name.lower().replace("'", "").replace('"', '').replace("?", "").replace("@", "").replace(".", "").replace("!", "").replace("#", "").replace("$", "").replace("+", "").replace("=", "").replace("%", "").replace("^", "").replace("(", "").replace(")", "").replace("&", "").replace("*", "").replace(";", "").replace(":", "").replace("/", "").replace(",", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("\\", "").replace("|", "").replace("`", "").replace("~", "").replace(" ", "_").replace("-----", "").replace("----", "").replace("---", "").replace("--", "").replace("-", "_").replace("___", "_").replace("__", "_")
		print(dir_name)
		os.system("mkdir _exports/" + str(dir_name))

# make_dir()

text_display 	= False
new_shot		= True

for a in os.listdir("_vid"):

	filename = str(a).split(".mp4")
	filename = str(filename[0])
	filename = filename.lower().replace("'", "").replace('"', '').replace("?", "").replace("@", "").replace(".", "").replace("!", "").replace("#", "").replace("$", "").replace("+", "").replace("=", "").replace("%", "").replace("^", "").replace("(", "").replace(")", "").replace("&", "").replace("*", "").replace(";", "").replace(":", "").replace("/", "").replace(",", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("\\", "").replace("|", "").replace("`", "").replace("~", "").replace(" ", "_").replace("-----", "").replace("----", "").replace("---", "").replace("--", "").replace("-", "_").replace("___", "_").replace("__", "_")
	# initialize the original frame dimensions, new frame dimensions,
	# and ratio between the dimensions
	(W, H) = (None, None)
	# (newW, newH) = (args["width"], args["height"])
	(newW, newH) = (320, 320)
	(rW, rH) = (None, None)

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	print("[INFO] loading EAST text detector...")
	print(a)
	# net = cv2.dnn.readNet(args["east"])
	net = cv2.dnn.readNet("frozen_east_text_detection.pb")

	# if a video path was not supplied, grab the reference to the web cam
	# if not args.get("video", False):
	# 	print("[INFO] starting video stream...")
	# 	vs = VideoStream(src=0).start()
	# 	time.sleep(1.0)

	# otherwise, grab a reference to the video file
	# else:
	# vs = cv2.VideoCapture(args["video"])
		# vs = cv2.VideoCapture(args["video"])
	vs = cv2.VideoCapture("./_vid/" + str(a))

	z, current_frame = vs.read()
	previous_frame 		= current_frame

	# start the FPS throughput estimator
	fps = FPS().start()

	frame_count = 0

	# loop over frames from the video stream
	while True:
		# print(vs.get(cv2.CAP_PROP_POS_FRAMES))
		# grab the current frame, then handle if we are using a
		# VideoStream or VideoCapture object
		if frame_count%100000 == 0:
			frame = vs.read()
			frame = frame[1]
			# frame = frame[1] if args.get("video", False) else frame

			# check to see if we have reached the end of the stream
			if frame is None:
				break

			# resize the frame, maintaining the aspect ratio
			frame = imutils.resize(frame, width=1000)
			orig = frame.copy()

			current_frame_gray 	= cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
			previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    

			curr_sum 	= sum(sum(current_frame_gray))
			prev_sum	= sum(sum(previous_frame_gray))
			# avg_f		= (curr_sum+prev_sum)/2
			avg_diff	= curr_sum/prev_sum
			# print(avg_diff)
			
			frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)

			# if our frame dimensions are None, we still need to compute the
			# ratio of old frame dimensions to new frame dimensions
			if W is None or H is None:
				(H, W) = frame.shape[:2]
				rW = W / float(newW)
				rH = H / float(newH)

			# resize the frame, this time ignoring aspect ratio
			frame = cv2.resize(frame, (newW, newH))

			# construct a blob from the frame and then perform a forward pass
			# of the model to obtain the two output layer sets
			blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
				(123.68, 116.78, 103.94), swapRB=True, crop=False)
			net.setInput(blob)
			(scores, geometry) = net.forward(layerNames)

			# decode the predictions, then  apply non-maxima suppression to
			# suppress weak, overlapping bounding boxes
			(rects, confidences) = decode_predictions(scores, geometry)
			boxes = non_max_suppression(np.array(rects), probs=confidences)
			try:
				if sum(confidences)/len(confidences) > 0.98:
					text_display = True
				else:
					text_display = False
			except:
				text_display = False
				# pass
			# loop over the bounding boxes
			if text_display == True and new_shot == True:
				print("-"*50)
				print("-"*50)
				print(sum(confidences)/len(confidences))
				print(avg_diff)
				for (startX, startY, endX, endY) in boxes:
					# scale the bounding box coordinates based on the respective
					# ratios
					startX = int(startX * rW)
					startY = int(startY * rH)
					endX = int(endX * rW)
					endY = int(endY * rH)

					# draw the bounding box on the frame
					cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

				# update the FPS counter
				fps.update()
				cv2.imwrite("_exports/" + str(filename) + "/" + str(filename) + "_" + str(frame_count) + ".jpg", orig)
				# show the output frame
				cv2.imshow("Text Detection", orig)
				# cv2.imshow('frame diff ',frame_diff)
				# print(sum(sum(frame_diff)))
				new_shot = False
				key = cv2.waitKey(1) & 0xFF


				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break
			if avg_diff > 1.04 or avg_diff < 0.94:
				new_shot = False
			else:
				new_shot = True
			previous_frame = current_frame.copy()
			z, current_frame = vs.read()
		frame_count+=1
	# stop the timer and display FPS information
	fps.stop()
	os.system("mv _vid/" + str(a) + " ./__scanned_vids/")
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# if we are using a webcam, release the pointer
	# if not args.get("video", False):
	# 	vs.stop()

	# otherwise, release the file pointer
	# else:
	vs.release()

	# close all windows
	cv2.destroyAllWindows()