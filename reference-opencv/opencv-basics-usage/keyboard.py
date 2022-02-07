import cv2

img = cv2.imread('./data/cat.bmp', cv2.IMREAD_COLOR)

# cv2.namedWindow('Image')
cv2.imshow('Image', img)

# reflects the image if press 'i' or 'I'
while True:
    query = cv2.waitKey()
    if query == ord('i') or query == ord('I'):
        img = ~img
        cv2.imshow('Image', img) 
    elif query == 27:
        break
       
cv2.destroyAllWindows()