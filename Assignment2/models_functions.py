import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer


def plot_bbox(img_id,x_min,x_max,y_min,y_max, pts_id):
    """ plotting bounding box of swimming pool image """

    # creating the path
    my_path = 'images/' + img_id

    # Load images and convert to array
    img = load_img(my_path, target_size=(512, 512))
    img = img_to_array(img)

    plt.imshow(img.astype(np.uint32))
    height, width, channel = img.shape
    print(f"Image: {img.shape}")

    isClosed = True


    #    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), -1)
    cv2.polylines(img, np.int64([pts_id]), isClosed, (255, 0, 0), 2)
    cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(img, "swimmingpool", (int(x_min), int(y_min)-10), font, 3, (0,255,0), -1)

    plt.figure(figsize=(15, 10))
    plt.title('Image with Bounding Box')
    plt.imshow(img.astype(np.uint32))
    plt.axis("off")
    plt.show()


def Shape(pts_id):
    """ return the polygon corner points coordinates """
    return np.int64([pts_id])


def DrawShape(Img, Polygon, Color):
    """Draw polygon on image """

    

    try:
        cv2.fillPoly(Img, Polygon, Color)  # Only using this function may cause errors, I don’t know why
    except:
        try:
            cv2.fillConvexPoly(Img, Polygon, Color)
        except:
            print('cant fill\n')

    return Img


def Drawrect(Img,x_min,x_max,y_min,y_max):
    """Draw Rectangle on image """
    cv2.rectangle(Img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), -1)

    return Img


def Intersectarea_OverlapIm_BoxArea_ShapeArea(ImShape, Polygon1, xmin, ymin, xmax, ymax):
    """ Calculates the intersecting area of ​​two polygons in an image and the individuals areas of both shapes.
      Return the three areas and the intersection area drawn on the image."""

    # Polygon area is filled with 122

    Img1 = np.zeros((512, 512), dtype=np.uint8)

    Im1 = DrawShape(Img1, Polygon1, 122)
    Img = np.zeros((512, 512), dtype=np.uint8)

    # Polygon area is filled with 133
    Im2 = cv2.rectangle(Img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), 133, -1)  # Polygon area is filled with 133
    Im = Im1 + Im2

    # According to the above filling value, so the pixel value in the new image is 255 as the overlapping place
    ret, OverlapIm = cv2.threshold(Im, 200, 255, cv2.THRESH_BINARY)

    # Find the area of ​​the overlapping area of ​​two polygons
    IntersectArea = np.sum(np.greater(OverlapIm, 0))

    # Below, use the function that comes with opencv to find out, the most contrast
    #     print( IntersectArea)
    #     print("OverlapImg")
    if IntersectArea != 0:
        contours, hierarchy = cv2.findContours(OverlapIm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         contourArea=cv2.contourArea(contours[0])
    #         print('contourArea={}\n'.format(contourArea))
    #         perimeter = cv2.arcLength(contours[0], True)
    #         print('contourPerimeter={}\n'.format(perimeter))
    #         RealContourArea=contourArea+perimeter
    #         print('RealContourArea={}\n'.format(RealContourArea))

    #     print(f"the area of bounding box is {(xmax-xmin)*(ymax-ymin)}")

    contours, hierarchy = cv2.findContours(Im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    BoxArea = cv2.contourArea(contours[0])
    #     print('contourArea={}\n'.format(BoxArea))
    #     perimeter = cv2.arcLength(contours[0], True)
    #     print('contourPerimeter={}\n'.format(perimeter))
    #     RealContourArea=BoxArea+perimeter
    #     print('RealContourArea={}\n'.format(RealContourArea))

    #     print("the area of the shape")

    contours, hierarchy = cv2.findContours(Im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ShapeArea = cv2.contourArea(contours[0])
    #     print('contourArea={}\n'.format(ShapeArea))
    #     perimeter = cv2.arcLength(contours[0], True)
    #     print('contourPerimeter={}\n'.format(perimeter))
    #     RealContourArea=ShapeArea+perimeter
    #     print('RealContourArea={}\n'.format(RealContourArea))

    return IntersectArea, OverlapIm, BoxArea, ShapeArea

def get_iou(IntersectArea,BoxArea,ShapeArea):
    """calculate IOU (Intersection Over Union) of the ground truth box and the prediction box  """
    iou = IntersectArea / float(BoxArea + ShapeArea - IntersectArea)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


class MyLabelBinarizer(LabelBinarizer):
    """  one-hot encode the label """

    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)

def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()
def plot_accuracy(acc,val_acc):
  # Plot training & validation accuracy values
  plt.figure()
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()