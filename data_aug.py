import cv2 
import numpy as np  
import os
import matplotlib.pyplot as plt
import random

# Define functions
def rand_rotate():
    rotate = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    return np.random.choice(rotate)
    
def rand_sp_noise(image):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    choises = [0.01,0.02,0.03,0.04,0.05]
    prob = np.random.choice(choises)
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def rand_resize(image):
    s     = np.random.choice([10,13,15,17])
    image = cv2.resize(image, (s,s))

    mean_B = int(np.mean(image[:,:,0]))
    mean_G = int(np.mean(image[:,:,1]))
    mean_R = int(np.mean(image[:,:,2]))

    output = np.ones((20,20,3), dtype=np.uint8)
    output[:,:,0] *= mean_B
    output[:,:,1] *= mean_G
    output[:,:,2] *= mean_R    

    if s%2 == 0:
        output[10-s//2:10+s//2, 10-s//2:10+s//2, :] = image
    else:
        output[10-s//2:10+s//2+1, 10-s//2:10+s//2+1, :] = image

    return output


# Read folders and save labels with images
folders=os.listdir('dataset/')

images=[]
labels= []
for folder in folders:
    files=os.listdir('dataset/'+folder)
    
    for file in files:
        img=cv2.imread('dataset/'+folder+'/'+file)
        img=cv2.resize(img,(20,20))

        for i in range(3):
            # random rotate image
            img = cv2.rotate(img, rand_rotate())

            # Adding random noise
            img = rand_sp_noise(img)

            # Adding random line
            
            # plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            # plt.show()
            cv2.imwrite('./dataset/'+folder+'/'+str(i)+'_'+file, img)

            # exit()
        print(file)