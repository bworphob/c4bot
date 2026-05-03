import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from .BrainComponent import build_architecture

# หา path ที่ถูกต้องสำหรับ Models โฟลเดอร์
_current_file = os.path.abspath(__file__)  # /path/to/src/Reinforcement/brains/ZeroBrain.py
_reinforcement_dir = os.path.dirname(os.path.dirname(_current_file))  # /path/to/src/Reinforcement
_models_dir = os.path.join(_reinforcement_dir, 'Models')

class ZeroBrain:
    def __init__(self, iteration):
        self.iteration = iteration
        # self.model_path = f'Models/model_v{iteration}.keras'  # เดิม
        # self.model_path = f'Reinforcement/Models/model_v{iteration}.keras'  # เดิม (relative)
        self.model_path = os.path.join(_models_dir, f'model_v{iteration}.keras')  # ใหม่ (absolute)

        
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            print(f"Creating directory: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
        
        
        
        if os.path.exists(self.model_path):
            print(f"Loading existing model: {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("Building new model from scratch...")
            self.model = self.setup_model()
            
        
        self.predict_function = tf.function(self.model)

    def setup_model(self):
        """ประกอบร่างสมองและตั้งค่าการเรียนรู้"""
       
        inputs, policy, value = build_architecture(res_blocks=7)
        
        
        model = Model(inputs=inputs, outputs=[policy, value])
        
        
        model.compile(
            loss={
                'policy_head': 'categorical_crossentropy', # Crossentropy for policy head to predict move probabilities
                'value_head': 'mean_squared_error'        # MSE for value head to predict win/loss probability
            },
            loss_weights={
                'policy_head': 0.5, 
                'value_head': 0.5
            },
            optimizer=Adam(learning_rate=0.001) # ปรับจูนความเร็วการเรียนรู้
        )
        # model.summary()
        # model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
        #               loss_weights={'value_head': 0.5, 'policy_head': 0.5},
        #               optimizer=Adam(),run_eagerly=False,steps_per_execution = 100)
        # model.summary()
        return model

    # def predict(self, board_state):
    #     """รับกระดาน (3, 6, 7) แล้วทายผลออกเป็น (โอกาสเลือกช่อง, โอกาสชนะ)"""
    #     # preprae input data by reshaping
    #     input_data = board_state.reshape((1, 3, 6, 7)).astype(np.float32)
        
    #     policy, value = self.predict_function(input_data, training=False)
        
    #     return policy.numpy()[0], value.numpy()[0][0]





    
    def predict(self, board_state):
        """รับกระดาน (3, 6, 7) แล้วทายผลออกเป็น (โอกาสเลือกช่อง, โอกาสชนะ)"""
        # เตรียมข้อมูลเบื้องต้นในรูปแบบ NCHW (1, 3, 6, 7) ตามมาตรฐานที่คุณใช้
        input_data = board_state.reshape((1, 3, 6, 7)).astype(np.float32)
        
        try:
            # พยายามรันด้วยรูปแบบปกติ (NCHW) ก่อน
            policy, value = self.predict_function(input_data, training=False)
        except tf.errors.UnimplementedError as e:
            # ถ้าเจอ Error ว่า CPU ไม่รองรับ NCHW ให้สลับเป็น NHWC (1, 6, 7, 3)
            if "NCHW" in str(e) or "Conv2D" in str(e):
                input_data_cpu = np.transpose(input_data, (0, 2, 3, 1))
                # รันด้วยความเร็วปกติผ่านโมเดลโดยตรง (ไม่ต้องใช้ tf.function ซ้อนในนี้เพื่อความชัวร์)
                # policy, value = self.model(input_data_cpu, training=False)#############
                policy, value = self.model.predict(input_data, verbose=0)
            else:
                raise e
        
        # return policy.numpy()[0], value.numpy()[0][0]############
        return policy[0], value[0][0]







    def train(self, memory_batch):
       
        states, target_policies, target_values = list(zip(*memory_batch))
        
        S = np.array(states)
        P = np.array(target_policies)
        V = np.array(target_values)
        
        self.model.fit(
            x=S, 
            y={'policy_head': P, 'value_head': V}, 
            batch_size=32, 
            epochs=2, 
            verbose=2
        )
        # update predict function after training
        self.predict_function = tf.function(self.model)

    # def save_model(self):
    #     """บันทึกสมองเก็บไว้ในตำแหน่งที่ตั้งไว้ใน model_path"""
    #     # หาชื่อโฟลเดอร์จาก path ที่ตั้งไว้ตอน __init__ (จะได้ 'Reinforcement/Models')
    #     model_dir = os.path.dirname(self.model_path)
    #     if not os.path.exists(model_dir):
    #         os.makedirs(model_dir, exist_ok=True)
            
    #     # บันทึกลงไปยัง Path ที่ถูกต้อง (Reinforcement/Models/model_vX.keras)
    #     self.model.save(self.model_path)
    #     print(f"Model successfully saved to: {self.model_path}")

    def save_model(self):
        """บันทึกสมองเก็บไว้ในตำแหน่งที่ตั้งไว้ใน model_path"""
        
        # actual_path = f'Reinforcement/Models/model_v{self.iteration}.keras'  # เดิม (relative)
        actual_path = os.path.join(_models_dir, f'model_v{self.iteration}.keras')  # ใหม่ (absolute)
        
        model_dir = os.path.dirname(actual_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            
        self.model.save(actual_path)
        
        self.model_path = actual_path
        print(f"Model successfully saved to: {actual_path}")