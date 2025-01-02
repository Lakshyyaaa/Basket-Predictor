import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

cap = cv2.VideoCapture('vid (1).mp4')
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 149, 'vmin': 0, 'hmax': 26, 'smax': 255, 'vmax': 255}

posListX, posListY = [], []
xList = [item for item in range(0, 1300)]
prediction = False

while True:
    isTrue, img = cap.read()
    if not isTrue:
        break

    img = img[0:900, :]
    imgColor, mask = myColorFinder.update(img, hsvVals)
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    if posListX:
        A, B, C = np.polyfit(posListX, posListY, 2)

        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            cv2.circle(imgContours, pos, 10, (0, 255, 0), cv2.FILLED)
            if i == 0:
                cv2.line(imgContours, pos, pos, (0, 255, 0), 5)
            else:
                cv2.line(imgContours, pos, (posListX[i - 1], posListY[i - 1]), (0, 255, 0), 5)

        for x in xList:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

        if len(posListX) < 10:
            a = A
            b = B
            c = C - 590

            x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))
            prediction = 330 < x < 430

        if prediction:
            cv2.putText(imgContours, "Basket", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 200, 0), 5, cv2.LINE_AA)
        else:
            cv2.putText(imgContours, "No Basket", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 200), 5, cv2.LINE_AA)

    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    cv2.imshow("ImageColor", imgContours)

    if cv2.waitKey(100) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
