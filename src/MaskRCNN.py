from torchvision.transforms import transforms as transforms
import torch
import cv2
import numpy as np
import random
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights


class MaskRCNN():
    def __init__(self) -> None:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
            'cpu')

        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']

    def predict(self, image, threshold):
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # forward pass of the image through the modle
            outputs = self.model(image)

        # get all the scores
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        # index of those scores which are above a certain threshold
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
        
        # get the masks, bounding boxes and labels
        masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
        labels = [self.class_names[i] for i in outputs[0]['labels']]

        # discard those below the threshold
        masks, boxes, labels = masks[:thresholded_preds_count], boxes[:thresholded_preds_count], labels[:thresholded_preds_count]

        return masks, boxes, labels
    
    def draw_segmentation_map(image, masks, boxes, labels):
        class_names = ["A", "M"]
        COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

        for i in range(len(masks)):
        
            # apply a random color mask to each object
            color = COLORS[random.randrange(0, len(COLORS))]
            #convert the original PIL image into NumPy format
            image = np.array(image)
            # convert from RGN to OpenCV BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # draw the bounding boxes around the objects
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                        thickness=2)
            # put the label text above the objects
            cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                        thickness=2, lineType=cv2.LINE_AA)
        
        return image



