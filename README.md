# AutomaticNumberPlateDetection-OCR
AUTOMATIC VEHICLE NUMBER PLATE DETECTION & OCR
Present By:- Sarang tamrakar

PROBLEM STATEMENT :-  we have problem statement that there is movie theatre parking slot & owner is facing the issue of revenue , he is not getting actual revenue as per number of car. So he said to build such kind of automatic system that will satisfy that concern.

Solution:- there are various strategies that we have followed….
1.	Data collection:-  so we have collected the data from various sources:-
•	Via Video recording then convert into images.
•	Via Google’s Open image dataset.
There are approx. 5000 images we have take for model building.

2.	Data annotation:- we have annotated the number plate with yolo format and there is only one class  which is number plate.

3.	Model building:- we have used yolo V4 model which internally uses Darknet architecture so we have trained model in darknet itself then we have converted it into tensorflow  2.X extension.

4.	 Inferencing :- we have used trained tensorflow model to detect the Number plate then we have crop the detected number plate.

5.	OCR:- we have taken cropped image of detected number plate then converted it into the base64 format for fast processing then we have tested it with PYTESSERECT, GOOGLE CLOUD VISION API , AWS TEXTRACT.

6.	Revenue:-  Then we have taken each number plate of vehicle then generate the Receipt which have give actual Benefits to the owner of Parking slot.

Thank you…
