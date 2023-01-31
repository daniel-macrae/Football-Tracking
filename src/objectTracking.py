import cv2
import numpy as np


def get_colour(original_image, masks, cnnBoxes, labels):
    #cnnBoxes = cnnBoxes[:-2]
    colour_vectors = []
    for bb in cnnBoxes:
        # crop the image to the player
        [(x1,y1),(x2,y2)] = bb
        img = original_image[y1:y2, x1:x2]
        #plt.imshow(img), plt.show()

        # find a vector that represents the colours here
        # convert to LAB colour space (easier to filter out the green pitch)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        a_channel = lab[:,:,1]
        
        th = cv2.threshold(a_channel,100,255,cv2.THRESH_BINARY)[1]
        masked = cv2.bitwise_and(img, img, mask = th)    # contains dark background
        m1 = masked.copy()
        m1[th==0]=(255,255,255)  
        #plt.imshow(m1), plt.show()
        
        pixels = np.float32(img.reshape(-1, 3))

        n_colors = 10
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 20, flags)
        _, counts = np.unique(labels, return_counts=True)
        
        indices = np.argsort(counts)[::-1]   
        freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
        rows = np.int_(img.shape[0]*freqs)

        dom_patch = np.zeros(shape=img.shape, dtype=np.float32)
        for i in range(len(rows) - 1):
            dom_patch[rows[i]:rows[i + 1], :, :] += np.float32(palette[indices[i]])

        img_colour_vector = dom_patch[:,0,:].mean(axis=0)

        img_colour_vector = palette.mean(axis=0)
        colour_vectors.append(img_colour_vector)
        #print(img_colour_vector)
    
    return np.array(colour_vectors)



def clustering(colour_vectors, k=2):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(colour_vectors, k, None, criteria, 10, flags)
    return labels


def convertBoxes(cnnBoxes):
    newBoxes = []
    for bb in cnnBoxes:
        h = bb[1][0] - bb[0][0]
        w = bb[1][1] - bb[0][1]
        x = bb[0][0]
        y = bb[0][1]
        newBoxes.append((x,y,h,w))
    return newBoxes

def makeTrackers(cnnBoxes, frame):
    TrDict = {
	"csrt": cv2.legacy.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.legacy.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.legacy.TrackerTLD_create,
	"medianflow": cv2.legacy.TrackerMedianFlow_create,
	"mosse": cv2.legacy.TrackerMOSSE_create
    }
    multi_trackers = cv2.legacy.MultiTracker_create()
    boxes = convertBoxes(cnnBoxes)

    for bb_i in boxes:
        tracker_i = TrDict['csrt']()
        multi_trackers.add(tracker_i, frame, bb_i)

    return multi_trackers, boxes