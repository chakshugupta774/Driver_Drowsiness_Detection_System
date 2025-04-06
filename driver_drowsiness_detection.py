# # import cv2
# # import os
# # from keras.models import load_model
# # import numpy as np
# # from pygame import mixer
# # import time


# # mixer.init()
# # sound = mixer.Sound('alarm.wav')

# # face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
# # leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
# # reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



# # lbl=['Close','Open']

# # model = load_model('models/cnncat2.h5')
# # path = os.getcwd()
# # cap = cv2.VideoCapture(0)
 
# # font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# # count=0
# # score=0
# # thicc=2
# # rpred=[99]
# # lpred=[99]

# # while(True):
# #     ret, frame = cap.read()
# #     height,width = frame.shape[:2] 

# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
# #     faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
# #     left_eye = leye.detectMultiScale(gray)
# #     right_eye =  reye.detectMultiScale(gray)

# #     cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

# #     for (x,y,w,h) in faces:
# #         cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

# #     for (x,y,w,h) in right_eye:
# #         r_eye=frame[y:y+h,x:x+w]
# #         count=count+1
# #         r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
# #         r_eye = cv2.resize(r_eye,(24,24))
# #         r_eye= r_eye/255
# #         r_eye=  r_eye.reshape(24,24,-1)
# #         r_eye = np.expand_dims(r_eye,axis=0)
# #         rpred = np.argmax(model.predict(r_eye), axis=-1)

# #         if(rpred[0]==1):
# #             lbl='Open' 
# #         if(rpred[0]==0):
# #             lbl='Closed'
# #         break

# #     for (x,y,w,h) in left_eye:
# #         l_eye=frame[y:y+h,x:x+w]
# #         count=count+1
# #         l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
# #         l_eye = cv2.resize(l_eye,(24,24))
# #         l_eye= l_eye/255
# #         l_eye=l_eye.reshape(24,24,-1)
# #         l_eye = np.expand_dims(l_eye,axis=0)
# #         lpred = np.argmax(model.predict(l_eye), axis=-1)
# #         if(lpred[0]==1):
# #             lbl='Open'   
# #         if(lpred[0]==0):
# #             lbl='Closed'
# #         break

# #     if(rpred[0]==0 and lpred[0]==0):
# #         score=score+1
# #         cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
# #     # if(rpred[0]==1 or lpred[0]==1):
# #     else:
# #         score=score-1
# #         cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
# #     if(score<0):
# #         score=0   
# #     cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
# #     if(score>15):
# #         #person is feeling sleepy so we beep the alarm
# #         cv2.imwrite(os.path.join(path,'image.jpg'),frame)
# #         try:
# #             sound.play()
            
# #         except:  # isplaying = False
# #             pass
# #         if(thicc<16):
# #             thicc= thicc+2
# #         else:
# #             thicc=thicc-2
# #             if(thicc<2):
# #                 thicc=2
# #         cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
# #     cv2.imshow('frame',frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# # cap.release()
# # cv2.destroyAllWindows()

 
# import cv2
# import os
# from keras.models import load_model
# import numpy as np
# from pygame import mixer
# import time


# mixer.init()
# sound = mixer.Sound('alarm.wav')

# face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
# leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
# reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



# lbl=['Close','Open']

# model = load_model('models/cnncat2.h5')
# path = os.getcwd()
# cap = cv2.VideoCapture(0)
 
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count=0
# score=0
# thicc=2
# rpred=[99]
# lpred=[99]

# # Add variables to track alarm state and timing
# alarm_on = False
# alarm_start_time = 0
# alarm_cooldown = False
# cooldown_start_time = 0
# cooldown_period = 10  # Seconds to wait before allowing alarm to trigger again

# while(True):
#     ret, frame = cap.read()
#     height,width = frame.shape[:2] 

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
#     left_eye = leye.detectMultiScale(gray)
#     right_eye = reye.detectMultiScale(gray)

#     cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

#     for (x,y,w,h) in right_eye:
#         r_eye=frame[y:y+h,x:x+w]
#         count=count+1
#         r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
#         r_eye = cv2.resize(r_eye,(24,24))
#         r_eye= r_eye/255
#         r_eye=  r_eye.reshape(24,24,-1)
#         r_eye = np.expand_dims(r_eye,axis=0)
#         rpred = np.argmax(model.predict(r_eye), axis=-1)

#         if(rpred[0]==1):
#             lbl='Open' 
#         if(rpred[0]==0):
#             lbl='Closed'
#         break

#     for (x,y,w,h) in left_eye:
#         l_eye=frame[y:y+h,x:x+w]
#         count=count+1
#         l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
#         l_eye = cv2.resize(l_eye,(24,24))
#         l_eye= l_eye/255
#         l_eye=l_eye.reshape(24,24,-1)
#         l_eye = np.expand_dims(l_eye,axis=0)
#         lpred = np.argmax(model.predict(l_eye), axis=-1)
#         if(lpred[0]==1):
#             lbl='Open'   
#         if(lpred[0]==0):
#             lbl='Closed'
#         break

#     if(rpred[0]==0 and lpred[0]==0):
#         score=score+1
#         cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
#     else:
#         score=score-1
#         cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
#     if(score<0):
#         score=0   
#     cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
#     # Check if cooldown period is active
#     if alarm_cooldown:
#         # Display cooldown message
#         cv2.putText(frame,'ALERT COOLDOWN',(width-300,height-20), font, 1,(0,255,255),1,cv2.LINE_AA)
        
#         # Check if cooldown period is over
#         if time.time() - cooldown_start_time >= cooldown_period:
#             alarm_cooldown = False
#             # Reset score to half threshold to give driver a chance
#             score = 8
    
#     # Modified alarm logic
#     if score > 15 and not alarm_cooldown:
#         # Person is feeling sleepy so we beep the alarm
#         cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        
#         # If alarm is not already playing, start it and record the time
#         if not alarm_on:
#             try:
#                 sound.play()
#                 alarm_on = True
#                 alarm_start_time = time.time()
#             except:
#                 pass
#         else:
#             # Check if 5 seconds have passed
#             if time.time() - alarm_start_time >= 5:
#                 # Stop the alarm
#                 sound.stop()
#                 alarm_on = False
                
#                 # Start the cooldown period
#                 alarm_cooldown = True
#                 cooldown_start_time = time.time()
        
#         if(thicc<16):
#             thicc= thicc+2
#         else:
#             thicc=thicc-2
#             if(thicc<2):
#                 thicc=2
#         cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
#     else:
#         # Reset alarm state if score drops below threshold
#         if alarm_on and not alarm_cooldown:
#             sound.stop()
#             alarm_on = False
            
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


from flask import Flask, Response, render_template, request, jsonify
import cv2
import os
import numpy as np
from keras.models import load_model
import time
import threading
from pygame import mixer

app = Flask(__name__)

# Global variables
outputFrame = None
lock = threading.Lock()
detection_active = False
score = 0
alarm_on = False
cooldown_active = False

# Initialize audio
mixer.init()
sound = mixer.Sound('static/alarm.wav')

# Load detection models
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')
model = load_model('models/cnncat2.h5')

# Settings with default values
settings = {
    'drowsiness_threshold': 15,
    'alarm_duration': 5,
    'cooldown_period': 10,
    'alert_sound': 'alarm.wav'
}

def detect_drowsiness():
    global outputFrame, lock, detection_active, score, alarm_on, cooldown_active
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    time.sleep(1.0)
    
    # Initialize variables
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count = 0
    score = 0
    thicc = 2
    rpred = [99]
    lpred = [99]
    alarm_start_time = 0
    cooldown_start_time = 0
    
    # Loop over frames from the camera
    while detection_active:
        # Read frame
        (grabbed, frame) = camera.read()
        if not grabbed:
            break
            
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces, eyes
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye_cascade.detectMultiScale(gray)
        right_eye = reye_cascade.detectMultiScale(gray)
        
        # Draw status background
        cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
        
        # Process face and eyes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)
            
        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y+h, x:x+w]
            count += 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye/255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = np.argmax(model.predict(r_eye), axis=-1)
            break
            
        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y+h, x:x+w]
            count += 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye/255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = np.argmax(model.predict(l_eye), axis=-1)
            break
            
        # Determine eye state and update score
        if rpred[0] == 0 and lpred[0] == 0:
            score += 1
            cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score -= 1
            cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            
        if score < 0:
            score = 0
            
        cv2.putText(frame, 'Score:'+str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Check for cooldown
        if cooldown_active:
            cv2.putText(frame, 'COOLDOWN', (width-200, height-20), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
            
            if time.time() - cooldown_start_time >= settings['cooldown_period']:
                cooldown_active = False
                score = settings['drowsiness_threshold'] // 2  # Reset to half threshold
                
        # Check for drowsiness and trigger alarm
        if score > settings['drowsiness_threshold'] and not cooldown_active:
            # Save drowsy state image
            cv2.imwrite('static/drowsy_state.jpg', frame)
            
            # Handle alarm
            if not alarm_on:
                try:
                    sound.play()
                    alarm_on = True
                    alarm_start_time = time.time()
                except:
                    pass
            else:
                # Check if alarm duration reached
                if time.time() - alarm_start_time >= settings['alarm_duration']:
                    sound.stop()
                    alarm_on = False
                    cooldown_active = True
                    cooldown_start_time = time.time()
                    
            # Visual alert
            if thicc < 16:
                thicc += 2
            else:
                thicc -= 2
                if thicc < 2:
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        else:
            # Stop alarm if eyes open during alarm
            if alarm_on and not cooldown_active:
                sound.stop()
                alarm_on = False
                
        # Add status text
        if alarm_on:
            cv2.putText(frame, "WAKE UP!", (int(width/2)-100, int(height/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                
        # Encode the frame as JPEG
        with lock:
            outputFrame = frame.copy()
            
    # Release resources
    camera.release()

def generate():
    # Generate JPEG frames for the web feed
    global outputFrame, lock
    
    while True:
        with lock:
            if outputFrame is None:
                continue
                
            # Encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            
            if not flag:
                continue
                
        # Yield the output frame in byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    # Return the main HTML page
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # Return the response generated along with the specific media type
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/start_detection", methods=['POST'])
def start_detection():
    global detection_active
    
    if not detection_active:
        detection_active = True
        t = threading.Thread(target=detect_drowsiness)
        t.daemon = True
        t.start()
        return jsonify({"status": "started"})
    
    return jsonify({"status": "already running"})

@app.route("/stop_detection", methods=['POST'])
def stop_detection():
    global detection_active, alarm_on
    
    detection_active = False
    
    if alarm_on:
        sound.stop()
        alarm_on = False
        
    return jsonify({"status": "stopped"})

@app.route("/update_settings", methods=['POST'])
def update_settings():
    global settings
    
    if request.method == 'POST':
        data = request.get_json()
        
        if 'drowsiness_threshold' in data:
            settings['drowsiness_threshold'] = int(data['drowsiness_threshold'])
            
        if 'alarm_duration' in data:
            settings['alarm_duration'] = int(data['alarm_duration'])
            
        if 'cooldown_period' in data:
            settings['cooldown_period'] = int(data['cooldown_period'])
            
        if 'alert_sound' in data:
            settings['alert_sound'] = data['alert_sound']
            sound = mixer.Sound(f"static/{data['alert_sound']}")
            
        return jsonify({"status": "settings updated", "settings": settings})
        
    return jsonify({"status": "error", "message": "Invalid request"})

@app.route("/get_settings", methods=['GET'])
def get_settings():
    global settings
    return jsonify(settings)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)