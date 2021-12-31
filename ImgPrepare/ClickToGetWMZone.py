'''
:Author: chen kai
:Date: 2021/12/31
:Description: Get White Matter regions from entire images (.png) by clicking the center of the regions.
              These regions are used as the backgrounds of training images in the CJO.
:Inputs: 
       DICOMImg_dir -- The path of the folder where entire images are stored, such as "PNGimg_withoutLesion/Patient1/"

       SaveFolder -- The path of the folder where the output White Matter regions will be stored.
       Size -- Predefined size of White Matter regions. 

:Outputs: 
       White Matter regions (originName_size_ImgSize_index.png) stored in SaveFolder.

:Operation:
      Left click: draw the rectangle and save the image
      press key 'q': next image
'''
import cv2 
import os

img = None
img_name = ""
save_dir = ""
index = 1
size = 65

def OnLClick(event, x, y, flag, param):
    global index, size, save_dir, img, img_name
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x-int(size/2),y-int(size/2)
        x2, y2 = x-int(size/2)+size-1, y-int(size/2)+size-1
        cv2.rectangle(img,(x1, y1),(x2, y2),(255,0,0),2)
        cv2.imwrite(os.path.join(save_dir,"{}_size_{}_{}.png".format(".".join(img_name.split('.')[:-1]),size, index)), img[y1:y2, x1:x2])
        index += 1


def ClickToGetWMZone(DICOMImg_dir, SaveFolder, Size):
    global img,save_dir,index, size, img_name
    path = ""
    for dir in SaveFolder.split("/"):
        path = os.path.join(path, dir)
        if not os.path.exists(path):
            os.mkdir(path)
    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", OnLClick)
    save_dir = SaveFolder
    size = Size
    for img_name in os.listdir(DICOMImg_dir):
        img = cv2.imread(os.path.join(DICOMImg_dir, img_name))
        cv2.imshow("Select Region", img)
        while cv2.waitKey(20) != ord('q'):
            cv2.imshow("Select Region", img)
        

    cv2.destroyAllWindows()


if __name__ == "__main__":
    ClickToGetWMZone("Resize_factor_zoom_factor4/ARGUILLOT HELENA","WMZone/ARGUILLOT HELENA", 65)

