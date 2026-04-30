from email import policy
import os
import sys
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

current_file_path = os.path.abspath(__file__)
reinforcement_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(reinforcement_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

from GameBoard.GameBoard import Connect4Board

# Logger TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTBrainWrapper:
    def __init__(self, engine_path):
        # 1. Engine
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
       
        self.inputs, self.outputs, self.bindings = [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    # def predict(self, board):
    #     # 1. เตรียม Input (3 แผ่น) - Logic เดิมเป๊ะ
    #     raw_board = np.array(board.board)
    #     layer_ai = (raw_board == 2).astype(np.float32)
    #     layer_human = (raw_board == 1).astype(np.float32)
    #     layer_turn = np.ones((6, 7), dtype=np.float32)
        
    #     state = np.stack([layer_ai, layer_human, layer_turn])
    #     input_data = state.reshape((1, 3, 6, 7)).astype(np.float32)

    #     # 2. ส่งข้อมูลไป GPU และรัน Inference
    #     self.inputs[0]['host'][:] = input_data.flatten()
    #     cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
    #     self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
    #     cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
    #     self.stream.synchronize()

    #     # 3. ดึงผลลัพธ์ (Policy) - Logic เดิมคือหาอันที่ขนาดเป็น 7
    #     out0 = self.outputs[0]['host']
    #     out1 = self.outputs[1]['host'] if len(self.outputs) > 1 else out0
        
    #     if len(out0) == 7:
    #         policy = out0
    #     else:
    #         policy = out1
            
    #     return policy

    # def predict(self, board):
    #     # 1. ใช้ฟังก์ชันจาก GameBoard โดยตรงเพื่อให้แน่ใจว่า Layer เหมือนตอนเทรน
    #     state = board.getStateAsPlayer().astype(np.float32)
        
    #     # 2. ตรวจสอบ Shape (NCHW: 1, 3, 6, 7)
    #     input_data = np.expand_dims(state, axis=0) 

    #     # 3. ส่งข้อมูลไป GPU (เหมือนเดิม)
    #     self.inputs[0]['host'][:] = input_data.flatten()
    #     cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
    #     self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
    #     cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
    #     self.stream.synchronize()

    #     # 4. ดึง Policy (7 ช่อง)
    #     # ตรวจสอบว่า output ไหนคือ Policy (ขนาด 7) และอันไหนคือ Value (ขนาด 1)
    #     out0 = self.outputs[0]['host']
    #     out1 = self.outputs[1]['host'] if len(self.outputs) > 1 else out0
        
    #     policy = out0 if len(out0) == 7 else out1
    #     return policy


    def predict(self, board):
    # 1. pull state
        state = board.getStateAsPlayer().astype(np.float32)
    
    # 2. prepare input
        input_data = np.expand_dims(state, axis=0) 

    # 3. Transfer to GPU
    
        self.inputs[0]['host'][:] = input_data.ravel() 
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
    
    # 4. Run Inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
    
    # 5. Transfer output back to CPU
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()

    # 6. Seperate Policy  Value
        res0 = self.outputs[0]['host']
        res1 = self.outputs[1]['host'] if len(self.outputs) > 1 else res0
    
    # 7. if output = 7 -> policy, else value
        if len(res0) == 7:
            policy = res0
        else:
            policy = res1

        return policy

def play():
    print("Loading TensorRT Engine (Optimized for Jetson Nano) ...")
    
    model_path = os.path.join(reinforcement_dir, "Models", "model_v8.engine")
    
    if not os.path.exists(model_path):
        print(f"Error: Engine file not found at {model_path}")
        return

    zero_ai = TRTBrainWrapper(model_path)
    
    human_player = 1
    board = Connect4Board(first_player=human_player)

    print("\n--- Game Start (TensorRT Edition)! ---")
    
    while not board.isEnd:
        print(f"\nRound No: {board.round}")
        board.showBoard()
        
        if board.current_turn == human_player:
            try:
                move = int(input(f"Your Turn (Player {human_player}). Enter column (0-6): "))
                if move not in board.validAction():
                    print("Invalid move! Try again.")
                    continue
            except ValueError:
                print("Please enter a number between 0 and 6.")
                continue
        else:
            print("AI (TensorRT) is thinking...")
            policy = zero_ai.predict(board)
            valid_actions = board.validAction()
            
            masked_policy = np.full(policy.shape, -np.inf)
            masked_policy[valid_actions] = policy[valid_actions]
            move = np.argmax(masked_policy)
            
            print(f"AI chose column: {move}")

        board.insertColumn(move)

    board.showBoard()
    if board.winner == 0:
        print("\n--- Game Over: DRAW! ---")
    else:
        winner_name = "You" if board.winner == human_player else "AI"
        print(f"\n--- Game Over: {winner_name} WON! ---")

if __name__ == "__main__":
    play()