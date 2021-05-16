import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
print(tf.__version__)
from tensorflow.python.saved_model import tag_constants
import numpy as np
import cv2
import pytesseract
import re
from datetime import datetime
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class graph_class:
    def __init__(self,fileObj):
        self.fileObj = fileObj
        self.model_dir = "yolov4-416"
        self.custom_classes_dic = {0: "NumberPlate"}
        self.iou_thresh = 0.50
        self.score_threshold = 0.20
        self.croppedImagePath = "images/croppedimage.jpg"
        self.InputImagePath = "images/Inputimage.jpg"

        self.saved_model_loaded = tf.saved_model.load(self.model_dir, tags=[tag_constants.SERVING])


    def prediction(self,image_path):

        try:
            image = cv2.imread(image_path)
            frame = cv2.resize(image, (416, 416))
            im_height, im_width = frame.shape[:2]
            img = frame / 255

            expand = tf.expand_dims(img, axis=0)
            # tf2.x takes only float32 numpy type
            expand = tf.cast(expand, dtype="float32")
            
            boxes, scores, classes, num_of_detection = self.run_image_through_loaded_model(expand,self.iou_thresh,self.score_threshold)
            boxes = self.convert_boxes_into_actual_format(boxes,im_height,im_width)
            cropped = self.cropping_image_by_boxes(frame,boxes,self.croppedImagePath)

            if (cropped):
                result = self.ocr_function(self.croppedImagePath)
            else:
                result = "Unkonwn Plate"

            return result
        except Exception as e:
            raise e


    def run_image_through_loaded_model(self,expended_image,iou_thresh,score_threshold):

        try:

            # run image through loaded model
            batch_data = tf.constant(expended_image)
            infer = self.saved_model_loaded.signatures['serving_default']
            pred_bbox = infer(batch_data)

            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1,
                                              tf.shape(pred_conf)[-1])), max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou_thresh,
                score_threshold=score_threshold)

            # squeeze the dimension
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            num_of_detection = valid_detections.numpy()[0]

            # filter out the prediction as per num_of_detection
            boxes = boxes[0:int(num_of_detection)]
            scores = scores[0:int(num_of_detection)]
            classes = classes[0:int(num_of_detection)]

            return boxes, scores, classes, num_of_detection

        except Exception as e:

            raise e

    def convert_boxes_into_actual_format(self,boxes,im_height,im_width):

        try:
            new_boxes = []
            for box in boxes:
                ymin = int(box[0] * im_height)
                xmin = int(box[1] * im_width)
                ymax = int(box[2] * im_height)
                xmax = int(box[3] * im_width)
                b = [ymin,xmin,ymax,xmax]
                new_boxes.append(b)

            return np.array(new_boxes)
        except Exception as e:

            raise e
    def cropping_image_by_boxes(self,image_np,boxes,croppedImagePath):
        """ALWAYS PASS UNSCALLED IMAGES & SAME SIZE OF IMAGE WHICH WE HAVE FED TO Model
        SO THAT CORDINATES LOCATION WILL MATCH TO THE OUR IMAGE FOR CROPPING
        """
        try:
            cropped = False

            for box in boxes:
                ymin = box[0]
                xmin = box[1]
                ymax = box[2]
                xmax = box[3]


                cropped_img = image_np[ymin:ymax,xmin:xmax]
                # img = Image.open(InputImagePath)
                # cropped_image = img.crop((xmin,ymin,xmax,ymax))
                # cropped_image = cropped_image.convert("L")
                # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(croppedImagePath,cropped_img)
                cropped = True

            return cropped
        except Exception as e:
            raise e


    def ocr_function(self,cropped_image_path):
        try:
            now = datetime.now()
            current_date = now.date()
            current_time = now.strftime("%H%M%S")

            cropped_img = cv2.imread(cropped_image_path)
            result = pytesseract.image_to_string(cropped_img)
            final_result = re.sub('[^A-Z0-9]','',result)
            if len(final_result) ==10:
                with open("final_result.txt", "a+") as f:
                    f.write("Current date : {}, Current Time : {}, Result Plate Number : {} \n".format(str(current_date),str(current_time),str(final_result)))
                    f.close()
            else:
                final_result = "Unknown_NumberPlate"
                with open("final_result.txt", "a+") as f:
                    f.write("Current date : {}, Current Time : {}, Result Plate Number : {} \n".format(str(current_date),str(current_time),str(final_result)))
                    f.close()


            return final_result
        except Exception as e:
            raise e






