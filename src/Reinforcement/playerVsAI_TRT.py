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
from Reinforcement.players.ZeroPlayer import TRTPlayer

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
        # if len(res0) == 7:
        #     policy = res0
        # else:
        #     policy = res1
        if len(res0) == 7:
            policy = res0
            value = res1[0] if res1 is not None else 0
        else:
            policy = res1 if res1 is not None else res0
            value = res0[0]

        return policy, value


def play():
    print("Loading TensorRT Engine (Optimized for Jetson Nano) ...")
    
    model_path = os.path.join(reinforcement_dir, "Models", "model_v4.engine")
    
    if not os.path.exists(model_path):
        print(f"Error: Engine file not found at {model_path}")
        return

    # 1. create Brain Wrapper
    ai_brain = TRTBrainWrapper(model_path)

    # 2. wrap with TRTPlayer for MCTS usage instead of direct engine calls
    # n_simulations=400 is recommended for Jetson Nano
    ZeroAI = TRTPlayer(ai_brain, n_simulations=400)
    
    human_player = 1
    board = Connect4Board(first_player=human_player)

    print("\n--- Game Start (TensorRT Edition with MCTS)! ---")
    
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
            print("AI (TensorRT + MCTS) is thinking...")
            
            # --- switch to MCTS search here ---
            # move and mcts_policy are computed by 400 simulation rollouts
            move, mcts_policy = ZeroAI.act(board, tau=0)

            # fetch current value estimate to display win likelihood
            _, value = ai_brain.predict(board)
            # ------------------------------------------
            
            ai_win_percent = (value + 1) / 2 * 100
            print(f"AI Confidence (from Current State): {ai_win_percent:.2f}%")
            print(f"AI chose column: {move}")
            print(f"AI Evaluation (Value): {value:.4f}")
            
        board.insertColumn(move)

    board.showBoard()
    if board.winner == 0:
        print("\n--- Game Over: DRAW! ---")
    else:
        winner_name = "You" if board.winner == human_player else "AI"
        print(f"\n--- Game Over: {winner_name} WON! ---")


if __name__ == "__main__":
    play()