from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock

import cv2
import numpy as np
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
import pandas as pd

import mediapipe as mp
from mediapipe.python.solutions import pose
from mediapipe.python.solutions.pose import Pose

from landmarks import landmarks

import pickle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class MyCameraApp(App):
    def build(self):
        self.cam = cv2.VideoCapture(0)
        self.results = None
        self.current_stage = ''
        self.counter = 0
        self.bodylang_prob = np.array([0, 0])
        self.bodylang_class = ''
        self.pose = Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

        # Load the machine learning model
        with open("final_upperblock.pkl", 'rb') as f:
            self.model = pickle.load(f)

        # Create the main layout
        layout = BoxLayout(orientation='vertical')

        # Create widgets
        self.label = Label(text='STAGE', font_size='20sp')
        layout.add_widget(self.label)

        self.counter_label = Label(text='REPS', font_size='20sp')
        layout.add_widget(self.counter_label)

        self.prob_label = Label(text='PROB', font_size='20sp')
        layout.add_widget(self.prob_label)

        self.reset_button = Button(text='RESET', on_press=self.reset_counter, font_size='20sp')
        layout.add_widget(self.reset_button)

        # Create the image widget
        self.image = Image()

        # Add the image widget to the layout
        layout.add_widget(self.image)

        # Schedule the update method
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return layout

    def update(self, dt):
        ret, frame = self.cam.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose_process(image)

        try:
            row = np.array(
                [[res.x, res.y, res.z, res.visibility] for res in self.results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks)
            self.bodylang_prob = self.model.predict_proba(X)[0]
            self.bodylang_class = self.model.predict(X)[0]

            if self.bodylang_class == "right" and self.bodylang_prob[self.bodylang_prob.argmax()] > 0.70:
                self.current_stage = "right"
            elif self.current_stage == "right" and self.bodylang_class == "left" \
                    and self.bodylang_prob[self.bodylang_prob.argmax()] > 0.70:
                self.current_stage = "left"
                self.counter += 1

        except Exception as e:
            print(e)

        self.update_ui()

    def update_ui(self):
        self.label.text = f'STAGE: {self.current_stage}'
        self.counter_label.text = f'REPS: {self.counter}'
        self.prob_label.text = f'PROB: {self.bodylang_prob[self.bodylang_prob.argmax()]:.2f}'

        # Display the image in Kivy
        img_texture = self.convert_image(self.results)
        self.image.texture = img_texture

    def convert_image(self, results):
        image = cv2.cvtColor(results.image, cv2.COLOR_BGR2RGB)
        buf1 = cv2.flip(image, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='rgb')
        image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        return image_texture

    def reset_counter(self, instance):
        self.counter = 0

    def pose_process(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                                  mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))
        results.image = image_rgb
        return results


if __name__ == '__main__':
    MyCameraApp().run()