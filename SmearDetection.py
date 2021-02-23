import cv2
import numpy as np
import os

# os.walk() -> [0] path, [1] folders, [2] files
path = "C://Users//Programmer//Downloads//Thermal image Test 7.21.18-20200609T193601Z-001//Thermal image Test 7.21.18"
folders = [i for i in os.walk(path) if i[0] == path][0][1]

for d in folders:
    print(d)
    sub_dir = os.path.join(path,d)
    cv2.destroyAllWindows()
    files = [i for i in os.walk(sub_dir) if i[0] == sub_dir][0][2]
    masks = []

    count = 0
    avg = []
    if not os.path.exists(d+'_average.jpg'):

        # Sum all the images. 
        for i,f in enumerate(files):
            
            # Every fifth image, so that similar objects in sequential images, 
            # don't get confused with smear.
            # This helps in faster code execution too.
             
            #if i % 1 == 0:
            img = cv2.imread(sub_dir+'/'+f,0)
            h,w = img.shape[:2]
            ratio = w/float(h)
            img = cv2.resize(img, (int(ratio*720), 720))

            if i == 0:
                avg = np.zeros(img.shape, dtype=np.float64)
            avg_img = np.mean(img)
                # Disregard any image that is too bright or too dark.
            if avg_img > 15 and avg_img < 239:
                count += 1
                avg += np.float64(img)                     

        avg /= count
        avg = np.uint8(avg)
        cv2.imwrite(d+'_average.jpg',avg)

    # Apply threshold 
    img = cv2.imread(d+'_average.jpg',0)
    img[np.where(img < 100)] = 0
    img[np.where(img >= 100)] = 255
    cv2.imwrite(d+'_thresh.jpg',img)

    # Delete the middle part in the mask, since it's the noise in all the dataset.
    img[int(img.shape[0]/3) : 2 * int(img.shape[0] / 3), :] = 255 
    mask = 255 * np.ones(img.shape, dtype=img.dtype) - img
    
    kernel = np.ones((3,3), np.uint8)
    kernel = np.asarray([[0, 1, 0],[1, 1, 1],[0, 1, 0]], dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
 
    cv2.imwrite(d+'_smear.jpg',mask)