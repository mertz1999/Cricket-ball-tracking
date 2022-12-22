import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob
import re
import pickle

#listing down all the file names
frames = os.listdir('frames/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

# Outpust video creator
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
out_vid = cv2.VideoWriter('./resultt.mp4', fourcc, 30, (1024, 576),True)

# Classify model loading
classify = pickle.load(open("./model/classification.sav", 'rb'))

# Remove pre. files
files = glob.glob('./patch/*')
for f in files:
    os.remove(f)

# Itrate on each frame image
cnt=0
for i in range(1,len(frames)):
    print(i,len(frames))
    # Read frames (Current and prev. frames)
    img = cv2.imread('frames/' + frames[i])
    back = cv2.imread('frames/' + frames[i-1])

    # Make motion frame and remove noises
    diff = cv2.absdiff(img, back)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(11,11),0)

    # Set thresh. and countours
    _ , mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    gray = cv2.bitwise_and(gray, mask)

    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_copy = np.copy(img)
    # cv2.drawContours(img_copy, contours, -1, (0,0,255), 3)

    # extract condidates box
    num=5*2
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        
        numer=min([w,h])
        denom=max([w,h])
        ratio=numer/denom

        xmin, ymin=x+w//2 - num, y+h//2 - num
        xmax, ymax=x+w//2 + num, y+h//2 + num

        if(ratio>=0.5 and ((w<=20*2) and (h<=20*2)) ):    
            cnt=cnt+1
            # try:
            #     cv2.imwrite("patch/"+str(cnt)+".png",img[ymin:ymax,xmin:xmax])
            # except:
            #     pass
    
            # Classification boxes
            try:
                selected_box  = cv2.cvtColor(img[ymin:ymax, xmin:xmax, :], cv2.COLOR_BGR2GRAY)
                selected_feat = np.array([selected_box]).reshape(1,-1)

                if classify.predict_proba(selected_feat)[0][1] > 0.7:
                    cv2.rectangle(img_copy, (xmin,ymin), (xmax,ymax),(0,255,0), 2)
                else:
                    cv2.rectangle(img_copy, (xmin,ymin), (xmax,ymax),(0,0,255), 2)
        
            except:
                pass


    # cv2.imshow("image", img_copy)

    # key = cv2.waitKey(0)
    # while key not in [ord('q'), ord('k')]:
    #     key = cv2.waitKey(0)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    out_vid.write(cv2.resize(img_copy,(1024,576)))
    # out_vid.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

out_vid.release()


