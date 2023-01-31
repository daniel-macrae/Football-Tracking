import numpy as np




def people_only(labels, masks, boxes):
    labels, masks, boxes = np.array(labels), np.array(masks), np.array(boxes)

    people = np.where(labels == 'person', True, False)

    labels = labels[people]
    masks = masks[people]
    boxes = boxes[people]

    return labels, masks, boxes