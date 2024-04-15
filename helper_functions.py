import cv2

#### Draw Transparent Rectangle BBOX ####
def DrawOpac(image, alpha=0.3, bbox=None, color=(255, 255, 255), DrawShape="rect", thickness=-1):
    if bbox is not None:
        overlay = image.copy()
        if DrawShape == "rect":
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, thickness)
        elif DrawShape == "circle":
            cx, cy = get_centroid(bbox)
            radius = min(bbox[2], bbox[3]) // 8  # Adjust the radius as needed
            # Draw the circle at the center of the bounding box
            cv2.circle(image, (cx, cy), radius, color, thickness)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image

def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    # Calculate the intersection area
    x_intersect = max(x1, x2)
    y_intersect = max(y1, y2)
    width_intersect = max(0, min(x1 + w1, x2 + w2) - x_intersect)
    height_intersect = max(0, min(y1 + h1, y2 + h2) - y_intersect)
    intersection_area = width_intersect * height_intersect
    # Calculate the union area
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area
    # Calculate the IoU
    iou = intersection_area / union_area
    return iou

def get_centroid(bbox):
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

def draw_line(image, roi_list, color=(0, 255, 0)):
    x, y, w, h = roi_list
    image = cv2.line(image, (0, y), (image.shape[1], y), color, 2)
    image = cv2.line(image, (x, 0), (x, image.shape[0]), color, 2)
    return image

def drawError(image, actualROI, predROI):
    x, y, w, h = actualROI
    cv2.rectangle(image, (x, y), (x + w, y + h), (220, 90, 90), 3)
    draw_line(image, actualROI, (220, 90, 90))
    img = DrawOpac(image, alpha=0.4, bbox=actualROI, color=(10, 255, 15), DrawShape="circle",
                     thickness=-1)
    final_img = DrawOpac(img, alpha=0.2, bbox=predROI, color=(0, 0, 250), DrawShape="circle",
                     thickness=2)
    return final_img
