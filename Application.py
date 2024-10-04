# Importing Libraries

import numpy as np

import cv2
import os
import sys
import time
import operator

from string import ascii_uppercase

import tkinter as tk
import pyttsx3
from PIL import ImageTk, Image
from hunspell import Hunspell
import enchant
import tensorflow as tf
print(tf.keras.__version__)
from tensorflow.keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

#Application :

class Application:

    def __init__(self):
        self.engine = pyttsx3.init()
        self.hs = Hunspell('en_US')
        self.vs = cv2.VideoCapture(1)
        self.current_image = None
        self.current_image2 = None
        self.json_file = open("Models\model_new.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()

        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("Models\model_new.h5")

   
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
          self.ct[i] = 0
        
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.title("Sign Language to Text Conversion")
        self.root.geometry("1000x900")

        # Title label
        self.title_label = tk.Label(self.root, text="Sign Language to Text Conversion", font=("Helvetica", 36, "bold"))
        self.title_label.pack(pady=20)

        # Panel for main camera feed
        self.camera_panel = tk.Label(self.root)
        self.camera_panel.pack(pady=10)

        # Current symbol label and value
        self.symbol_frame = tk.Frame(self.root)
        self.symbol_frame.pack(pady=10)

        self.symbol_label = tk.Label(self.symbol_frame, text="Character:", font=("Helvetica", 24))
        self.symbol_label.grid(row=0, column=0, padx=10)

        self.symbol_value = tk.Label(self.symbol_frame, text="_", font=("Helvetica", 24, "bold"))
        self.symbol_value.grid(row=0, column=1, padx=10)
    
        # Word label and value
        self.word_frame = tk.Frame(self.root)
        self.word_frame.pack(pady=10)

        self.word_label = tk.Label(self.word_frame, text="Word:", font=("Helvetica", 24))
        self.word_label.grid(row=0, column=0, padx=10)

        self.word_value = tk.Label(self.word_frame, text="_", font=("Helvetica", 24, "bold"))
        self.word_value.grid(row=0, column=1, padx=10)
        # self.delete_button = tk.Button(self.word_frame, text="Delete Char", font=("Helvetica", 15),bg="#ff4d4d", fg="white", activebackground="#ff1a1a", activeforeground="white",
        #                                relief="solid", borderwidth=2, command=self.delete_last_letter )
        # self.delete_button.grid(row=0, column=5, padx=10, sticky='w')

        # Sentence label and value
        self.sentence_frame = tk.Frame(self.root)
        self.sentence_frame.pack(pady=10)

        self.sentence_label = tk.Label(self.sentence_frame, text="Sentence:", font=("Helvetica", 24))
        self.sentence_label.grid(row=0, column=0, padx=10)

        self.sentence_value = tk.Label(self.sentence_frame, text="_", font=("Helvetica", 24, "bold"))
        self.sentence_value.grid(row=0, column=1, padx=10)

        # Suggestions label
        self.suggestions_label = tk.Label(self.root, text="Suggestions:", fg="red", font=("Helvetica", 28, "bold"))
        self.suggestions_label.pack(pady=20)

        # Buttons for suggestions
        self.suggestions_frame = tk.Frame(self.root)
        self.suggestions_frame.pack(pady=10)

        self.suggestion_btn1 = tk.Button(self.suggestions_frame, text="Suggestion 1", command=self.action1, font=("Helvetica", 20))
        self.suggestion_btn1.grid(row=0, column=0, padx=10)

        self.suggestion_btn2 = tk.Button(self.suggestions_frame, text="Suggestion 2", command=self.action2, font=("Helvetica", 20))
        self.suggestion_btn2.grid(row=0, column=1, padx=10)

        self.suggestion_btn3 = tk.Button(self.suggestions_frame, text="Suggestion 3", command=self.action3, font=("Helvetica", 20))
        self.suggestion_btn3.grid(row=0, column=2, padx=10)

        # self.delete_frame = tk.Frame(self.root)
        # self.delete_frame.pack(pady=10)
        # self.delete_button = tk.Button(self.delete_frame, text="Delete Char", font=("Helvetica", 15),bg="#ff4d4d", fg="white", activebackground="#ff1a1a", activeforeground="white",
        #                                relief="solid", borderwidth=2, command=self.delete_last_letter )
        # self.delete_button.grid(row=0, column=0, padx=10)

        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
     
        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()

    def delete_last_letter(self):
        # Delete the last letter in the current word
        if len(self.word) > 0:
            self.word = self.word[:-1]
            self.word_value_label.config(text=self.word)

    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0) ,1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = self.current_image)

            self.camera_panel.imgtk = imgtk
            self.camera_panel.config(image = imgtk)

            cv2image = cv2image[y1 : y2, x1 : x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            res_rgb = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
            res_rgb_resized = cv2.resize(res_rgb, (224, 224))
             # Preprocess the image for MobileNetV2
            input_image = np.expand_dims(res_rgb_resized, axis=0)
            input_image = tf.keras.applications.mobilenet_v2.preprocess_input(input_image)

            self.predict(input_image)

            self.current_image2 = Image.fromarray(res)

            imgtk = ImageTk.PhotoImage(image = self.current_image2)

            self.camera_panel.imgtk = imgtk
            self.camera_panel.config(image = imgtk)

            self.symbol_value.config(text = self.current_symbol, font = ("Courier", 30))

            self.word_value.config(text = self.word, font = ("Courier", 30))

            self.sentence_value.config(text = self.str,font = ("Courier", 30))

            predicts = self.hs.suggest(self.word)
            
            if(len(predicts) > 1):

                self.suggestion_btn1.config(text = predicts[0], font = ("Courier", 20))

            else:

                self.suggestion_btn1.config(text = "")

            if(len(predicts) > 2):

                self.suggestion_btn2.config(text = predicts[1], font = ("Courier", 20))

            else:

                self.suggestion_btn2.config(text = "")

            if(len(predicts) > 3):

                self.suggestion_btn3.config(text = predicts[2], font = ("Courier", 20))

            else:

                self.suggestion_btn3.config(text = "")


        self.root.after(5, self.video_loop)

    def predict(self, test_image):
  
        # Make predictions
        result = self.loaded_model.predict(test_image)
    

        prediction = {}
        
        prediction['blank'] = result[0][0]
        letters = [i for i in ascii_uppercase if i not in ['J', 'Z']]

        inde = 1

        for i in letters:

            prediction[i] = result[0][inde]

            inde += 1

        #LAYER 1

        prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

        self.current_symbol = prediction[0][0]
        
        if(self.current_symbol == 'blank'):

            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if(self.ct[self.current_symbol] > 60):

            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.ct[self.current_symbol] - self.ct[i]

                if tmp < 0:
                    tmp *= -1

                if tmp <= 20:
                    self.ct['blank'] = 0

                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return

            self.ct['blank'] = 0

            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':

                if self.blank_flag == 0:
                    self.blank_flag = 1

                    if len(self.str) > 0:
                        self.str += " "
                        
                    self.str += self.word
                    self.speak_word(self.word)
                    self.word = ""

            else:

                if(len(self.str) > 16):
                    self.str = ""

                self.blank_flag = 0

                self.word += self.current_symbol
                self.speak_word(self.current_symbol)

    
    def action1(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 0):

            self.word = ""

            self.str += " "

            self.str += predicts[0]
            self.speak_word(predicts[0])

    def action2(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 1):
            self.word = ""
            self.str += " "
            self.str += predicts[1]
            self.speak_word(predicts[1])
    def action3(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 2):
            self.word = ""
            self.str += " "
            self.str += predicts[2]
            self.speak_word(predicts[2])
    def action4(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 3):
            self.word = ""
            self.str += " "
            self.str += predicts[3]
            self.speak_word(predicts[3])
    def action5(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 4):
            self.word = ""
            self.str += " "
            self.str += predicts[4]
            self.speak_word(predicts[4])    

    def speak_word(self, word):
        """Speak the given word using pyttsx3."""
        self.engine.say(word)
        self.engine.runAndWait()

    def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

print("Starting Application...")

(Application()).root.mainloop()
