import cv2, mouse, autopy, math, threading
import mediapipe as mp
import tkinter as tk

#Gesture/User variables
perfect_distance = False
lm_dict = {}

#Backend definitions
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
assert cap.isOpened()

def gestures(lm_dict):
    while True:
        if len(lm_dict) == 21:
            palm = lm_dict[0]
            tips_to_palm = math.dist(lm_dict[4], palm) + math.dist(lm_dict[8], palm) + math.dist(lm_dict[12], palm) + math.dist(lm_dict[16], palm) + math.dist(lm_dict[20], palm)
            print("Distance ", tips_to_palm)
            if tips_to_palm > 1.3:
                print("open hand")
            else:
                print("closed hand")

def run():
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        mouse_coordiantes = ()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                for id, point in enumerate(hand_landmarks.landmark):
                    h,w,c= frame.shape
                    cx,cy=int(point.x * w) , int(point.y * h)
                    lm_dict[id] = [point.x,point.y]
                    cv2.putText(frame, str(id), (cx+5,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, cv2.LINE_AA)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            

            #TODO check user distance by asking for closed hand 

            #Move mouse
            #mouse_coordiantes = (int(lm_dict[8][0] * screen_width), int(lm_dict[8][1] * screen_height))
            #autopy.mouse.smooth_move(mouse_coordiantes[0], mouse_coordiantes[1])
            #autopy.mouse.move(mouse_coordiantes[0], mouse_coordiantes[1])
            #mouse.move(mouse_coordiantes[0], mouse_coordiantes[1])
            #gestures(lm_dict)
        cv2.imshow('Hand Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    t1 = threading.Thread(target = run)
    t2 = threading.Thread(target = gestures, args=(lm_dict,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
