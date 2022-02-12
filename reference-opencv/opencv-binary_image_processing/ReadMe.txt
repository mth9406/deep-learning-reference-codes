We usually binarize 'gray scale image'

(1) Binarization

threshold.py, threshold_trackbar.py

cv2.threshold()
using cv2.threshold with trakbar
you can find the optimal threshold using the 
trackbar in the threshold_trackbar.py

(2) Otsu algorithm: automatically determining threshold 
(in binarization)

otsu.py

automatic binarization
when the distribution is 'bimodal'.
say, we havye 'background' and 'object'!
you can visualize the patten by cv2.histogram.
refer to histogram1.py or histogram2.py in '../opencv-basics-brightness'
objective function is to minimize within variance,
thus, maximizing between variance

fast and stable algorithm using a recursive formula.

(3) local binarization

sudoku.py, local_threshold.py,
adaptiveThreshold.py (cv2.adaptiveThreshold())

refer to sudoku.py to why we need local binarization.
in short, uneven brightness matters too much.

to solve the above issue,
we split the images into patches and apply Otsu or threshold 
algorithm locally (patch by patch).
refer to local_threshold.p

* notice how we split the images, which is similar to
* making a batch iterator.

refer to adaptiveThreshold.py if you want
example usage of cv2.adaptiveThreshold()

(4) morphology operation
we want to analyze how the image looks 

(4)-1) erosion and dilation (of an object (color dim = 255)) 

erode_dilate.py

cv2.erode(), 
cv2.getStructuringElement() -> kernel

cv2.dilate()

(4)-2) Opening and closing

we usually use opening to erase noise in an image.

* Opening = Erosion -> dilation
    : to erase small objects (noise)
    : to disconnect some weak lines

* Closing = dilation -> Erosion
    : to make weak lines stronger

refer to opening_closing.py for an example usage of cv2.morphologyEx()

(5) (Object) Labeling 

Terminology:
    * A pixel (value >= 1) is an object
    * zero: background, nonzero: object
    * label map: int32 2dArray

labeling.py

cv2.connectedComponents()
cv2.connectedComponentsWithStats() 
-> retval, labels, stats, centroids