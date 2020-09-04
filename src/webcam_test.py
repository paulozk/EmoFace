import cv2
from src.processing import *
from src.model import *
import time

# init CNN
weights_path = 'data/weights.h5'
model = CNN(height=48, width=48, n_classes=7, learning_rate=0.001)
model.load_model(weights_path)
cap = cv2.VideoCapture(0)
# init face extractor
extractor = FaceExtractor()

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (30, 200)
fontScale = 1
fontColor = (255, 0, 0)
lineType = 2

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # detect face in frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        face = extractor.detect(gray)[0]
        # draw rectangle
        cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)
        emotion = model.predict(gray)

        cv2.putText(frame, model.show_output(emotion),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        time.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass
    finally:
        cv2.imshow('frame', frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()