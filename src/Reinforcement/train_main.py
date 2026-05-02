# import os
# import pickle
# import datetime
# import random
# import numpy as np
# from tqdm import tqdm
# from argparse import ArgumentParser
# import glob


# from GameBoard.GameBoard import Connect4Board
# from brains.ZeroBrain import ZeroBrain
# from players.ZeroPlayer import ZeroPlayer



# def save_experience(folder_name, filename, data):
#     # path = f"./datasets/{folder_name}"
#     path = f"./Reinforcement/datasets/{folder_name}"

#     if not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)
#     with open(os.path.join(path, filename), 'wb') as f:
#         pickle.dump(data, f)

# def load_experience(file_path):
#     with open(file_path, 'rb') as f:
#         return pickle.load(f)

# # --- self-play ---

# def run_self_play(iteration, num_games):
#     print(f"\n--- Self-Play Iteration {iteration} ---")
#     brain = ZeroBrain(iteration)
#     player = ZeroPlayer(brain)
    
#     # TURN_TAU0: 8 ตาแรกจะสุ่มเดิน (Explore) ตาที่เหลือจะเดินจริงจัง (Exploit)
#     TURN_TAU0 = 8 

#     for i in tqdm(range(num_games), desc="Self-Playing"):
#         game = Connect4Board(first_player=1)
#         history = [] # เก็บ [state, policy]

#         while not game.isEnd:
#             state = game.getStateAsPlayer()
            
#             # tau = 1 (Explore), Tau = 0 (Exploit)
#             tau = 1 if game.round < TURN_TAU0 else 0
#             action, policy = player.act(game, tau=tau)
            
#             history.append([state, policy])
#             game.insertColumn(action)

       
#         winner_value = 0
#         if game.winner == 1: winner_value = 1
#         elif game.winner == 2: winner_value = -1

#         # เตรียมข้อมูลประสบการณ์ (Experience Package)
#         experiences = []
#         for idx, (state, policy) in enumerate(history):
#             # ตาแรกๆ ให้ value เป็น 0 (Neutral) ตาหลังๆ ให้ตามผลชนะจริง
#             current_value = 0 if idx == 0 else winner_value
#             experiences.append([state, policy, current_value])

#         # บันทึกไฟล์แยกตามเกมและวันที่
#         # date_str = datetime.datetime.today().strftime("%Y-%m-%d")
#         date_str = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
#         save_experience(f"iter_{iteration}", f"game_{i}_{date_str}.pickle", experiences)

# # --- Training ---

# def run_training(iteration):
#     print(f"\n--- Training Brain for Iteration {iteration + 1} ---")
#     # โหลดสมองรุ่นปัจจุบันมาเทรนเป็นรุ่นถัดไป
#     brain = ZeroBrain(iteration)
    
#     # ดึงข้อมูลจากหลายๆ Iteration ล่าสุดมาเทรน
#     all_datasets = []
#     # dataset_root = "./datasets/"
#     dataset_root = "./Reinforcement/datasets/"
#     # วนลูปย้อนหลังไป 5 รุ่นล่าสุด
#     folders = sorted(os.listdir(dataset_root))[-5:]
#     for folder in folders:
#         folder_path = os.path.join(dataset_root, folder)
#         for file in os.listdir(folder_path):
#             all_datasets.extend(load_experience(os.path.join(folder_path, file)))

#     print(f"Total samples found: {len(all_datasets)}")
    
#     # สุ่ม Batch มาเทรน 15 รอบ รอบละ 2048 ตัวอย่างตามรุ่นพี่
#     for _ in range(30):
#         sample_batch = random.sample(all_datasets, min(2048, len(all_datasets)))
#         brain.train(sample_batch)

#     # บันทึกเป็นสมองรุ่นใหม่
#     brain.iteration = iteration + 1
#     brain.save_model()

# # --- ขั้นตอนที่ 3: Evaluation (ทดสอบฝีมือ) ---

# # def run_evaluation(old_iter, new_iter):
# #     print(f"\n--- Evaluating: Model {old_iter} vs {new_iter} ---")
# #     player_old = ZeroPlayer(ZeroBrain(old_iter))
# #     player_new = ZeroPlayer(ZeroBrain(new_iter))
    
# #     new_wins = 0
# #     total_games = 10 # เล่นทดสอบ 10 เกม

# #     for i in range(total_games):
# #         game = Connect4Board(first_player=random.choice([1, 2]))
# #         while not game.isEnd:
# #             # สลับตาเดิน
# #             current_player = player_new if game.current_turn == 1 else player_old
# #             action, _ = current_player.act(game, tau=0)
# #             game.insertColumn(action)
        
# #         if game.winner == 1: # สมมติรุ่นใหม่เป็นเลข 1
# #             new_wins += 1
            
# #     win_rate = (new_wins / total_games) * 100
# #     print(f"Model {new_iter} Win Rate: {win_rate}%")
# #     return win_rate > 55 # ต้องชนะมากกว่า 55% ถึงจะผ่านตามเกณฑ์รุ่นพี่

# # def run_evaluation(old_iter, new_iter):
# #     print(f"\n--- Evaluating: Model {old_iter} vs {new_iter} ---")
# #     player_old = ZeroPlayer(ZeroBrain(old_iter))
# #     player_new = ZeroPlayer(ZeroBrain(new_iter))
    
# #     new_wins = 0
# #     total_games = 10 

# #     for i in range(total_games):
# #         # สลับฝั่งกันเริ่ม: เกมคู่ให้รุ่นน้องเริ่มก่อน เกมคี่ให้รุ่นพี่เริ่มก่อน
# #         new_player_id = 1 if i % 2 == 0 else 2
# #         game = Connect4Board(first_player=1) # หรือใช้ random.choice([1, 2])
        
# #         while not game.isEnd:
# #             # ตรวจสอบว่าตาใคร แล้วใช้ Player คนนั้นเดิน
# #             if game.current_turn == new_player_id:
# #                 action, _ = player_new.act(game, tau=0)
# #             else:
# #                 action, _ = player_old.act(game, tau=0)
# #             game.insertColumn(action)
        
# #         # ตรวจสอบว่าคนที่ชนะ คือ ID ของรุ่นน้องหรือไม่
# #         if game.winner == new_player_id:
# #             new_wins += 1
            
# #     win_rate = (new_wins / total_games) * 100
# #     print(f"Model {new_iter} Win Rate: {win_rate}%")
# #     return win_rate > 55

# def run_evaluation(old_iter, new_iter, total_games=50, threshold=0.6):
#     print(f"\n--- Evaluating: Model {old_iter} vs {new_iter} ---")

#     player_old = ZeroPlayer(ZeroBrain(old_iter))
#     player_new = ZeroPlayer(ZeroBrain(new_iter))

#     new_score = 0  # ใช้ score แทน wins (รองรับ draw)

#     for i in range(total_games):
#         # --- สลับฝั่งเริ่ม ---
#         if i % 2 == 0:
#             player1 = player_new
#             player2 = player_old
#             new_player_id = 1
#         else:
#             player1 = player_old
#             player2 = player_new
#             new_player_id = 2

#         game = Connect4Board(first_player=1)

#         # --- เล่นเกม ---
#         while not game.isEnd:
#             if game.current_turn == 1:
#                 action, _ = player1.act(game, tau=0)
#             else:
#                 action, _ = player2.act(game, tau=0)

#             game.insertColumn(action)

#         # --- คำนวณผล ---
#         if game.winner == new_player_id:
#             new_score += 1
#         elif game.winner == 0:  # draw
#             new_score += 0.5
#         # ถ้าแพ้ = +0

#     win_rate = new_score / total_games

#     print(f"New Model Score: {new_score}/{total_games}")
#     print(f"Win Rate: {win_rate*100:.2f}%")

#     return win_rate >= threshold

# # --- MAIN LOOP ---

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--games", type=int, default=50)
#     args = parser.parse_args()

#     # current_iter = 0
#     # --- ส่วนที่ปรับปรุงเพื่อหาเลขล่าสุด ---
#     # ค้นหาไฟล์ .keras ทั้งหมดในโฟลเดอร์ Models
#     model_files = glob.glob("./Reinforcement/Models/model_v*.keras")
    
#     if model_files:
#         # ดึงตัวเลขจากชื่อไฟล์ เช่น 'model_v2.keras' -> 2
#         iters = [int(f.split('_v')[-1].split('.')[0]) for f in model_files]
#         current_iter = max(iters)
#         print(f">>> Resuming from Iteration: {current_iter}")
#     else:
#         current_iter = 0
#         print(">>> No existing models found. Starting from Iteration 0.")
#     # ----------------------------------


#     while True:
#         # 1. สร้างข้อมูล
#         run_self_play(current_iter, args.games)
        
#         # 2. เทรนสมองรุ่นใหม่
#         run_training(current_iter)
        
#         # 3. ตรวจสอบคุณภาพ
#         if current_iter >= 0:
#             # is_better = run_evaluation(current_iter, current_iter + 1)
#             is_better = run_evaluation(current_iter, current_iter + 1, total_games=10, threshold=0.55)
#             if not is_better:
#                 print("Model didn't improve, retraining with more data...")
#                 # วนลูปเก็บข้อมูลเพิ่มโดยไม่เปลี่ยนเลข Iteration
#                 continue 
        
#         current_iter += 1











import os
import pickle
import datetime
import random
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import glob

from GameBoard.GameBoard import Connect4Board
from brains.ZeroBrain import ZeroBrain
from players.ZeroPlayer import ZeroPlayer

def save_experience(folder_name, filename, data):
    path = f"./Reinforcement/datasets/{folder_name}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(data, f)

def load_experience(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# --- self-play ---

def run_self_play(iteration, num_games):
    print(f"\n--- Self-Play Iteration {iteration} ---")
    brain = ZeroBrain(iteration)
    player = ZeroPlayer(brain)
    
    # TURN_TAU0: 8 ตาแรกจะสุ่มเดิน (Explore) ตาที่เหลือจะเดินจริงจัง (Exploit)
    TURN_TAU0 = 8 

    for i in tqdm(range(num_games), desc="Self-Playing"):
        game = Connect4Board(first_player=1)
        history = [] # เก็บ [state, policy, turn] เพื่อใช้คำนวณ Relative Value

        while not game.isEnd:
            state = game.getStateAsPlayer()
            
            # tau = 1 (Explore), Tau = 0 (Exploit)
            tau = 1 if game.round < TURN_TAU0 else 0
            action, policy = player.act(game, tau=tau)
            
            # [ปรับปรุง] เก็บ game.current_turn ไว้ใน history เพื่อเช็คผลแพ้ชนะอิงตามผู้เล่นในตานั้น
            history.append([state, policy, game.current_turn])
            game.insertColumn(action)

        # เตรียมข้อมูลประสบการณ์ (Experience Package)
        experiences = []
        for idx, (state, policy, turn) in enumerate(history):
            # [ปรับปรุง] คำนวณ Value แบบ Relative Perspective
            if game.winner == 0:
                # เสมอ
                current_value = 0
            else:
                # ถ้าผู้ชนะคือคนเดียวกับเจ้าของ State นั้น (คนที่กำลังจะเดินในจังหวะนั้น) ให้เป็น 1
                # ถ้าไม่ใช่ (แสดงว่าคนเดินจังหวะนั้นแพ้ในอนาคต) ให้เป็น -1
                current_value = 1 if game.winner == turn else -1
            
            experiences.append([state, policy, current_value])

        # บันทึกไฟล์แยกตามเกมและวันที่
        date_str = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        save_experience(f"iter_{iteration}", f"game_{i}_{date_str}.pickle", experiences)

# --- Training ---

def run_training(iteration):
    print(f"\n--- Training Brain for Iteration {iteration + 1} ---")
    brain = ZeroBrain(iteration)
    
    all_datasets = []
    dataset_root = "./Reinforcement/datasets/"
    
    if not os.path.exists(dataset_root):
        print("No dataset folder found.")
        return

    # วนลูปย้อนหลังไป 5 รุ่นล่าสุด
    folders = sorted(os.listdir(dataset_root))[-5:]
    for folder in folders:
        folder_path = os.path.join(dataset_root, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                all_datasets.extend(load_experience(os.path.join(folder_path, file)))

    print(f"Total samples found: {len(all_datasets)}")
    
    if len(all_datasets) == 0:
        print("Dataset is empty, skipping training.")
        return

    # สุ่ม Batch มาเทรน 30 รอบ รอบละ 2048 ตัวอย่าง
    for _ in range(30):
        sample_batch = random.sample(all_datasets, min(2048, len(all_datasets)))
        brain.train(sample_batch)

    brain.iteration = iteration + 1
    brain.save_model()

# --- Evaluation ---

def run_evaluation(old_iter, new_iter, total_games=10, threshold=0.55):
    print(f"\n--- Evaluating: Model {old_iter} vs {new_iter} ---")

    player_old = ZeroPlayer(ZeroBrain(old_iter))
    player_new = ZeroPlayer(ZeroBrain(new_iter))

    new_score = 0 

    for i in range(total_games):
        if i % 2 == 0:
            player1, player2 = player_new, player_old
            new_player_id = 1
        else:
            player1, player2 = player_old, player_new
            new_player_id = 2

        game = Connect4Board(first_player=1)

        while not game.isEnd:
            if game.current_turn == 1:
                action, _ = player1.act(game, tau=0)
            else:
                action, _ = player2.act(game, tau=0)
            game.insertColumn(action)

        if game.winner == new_player_id:
            new_score += 1
        elif game.winner == 0:
            new_score += 0.5

    win_rate = new_score / total_games
    print(f"New Model Score: {new_score}/{total_games} (Win Rate: {win_rate*100:.2f}%)")
    return win_rate >= threshold

# --- MAIN LOOP ---

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    args = parser.parse_args()

    model_files = glob.glob("./Reinforcement/Models/model_v*.keras")
    
    if model_files:
        iters = [int(f.split('_v')[-1].split('.')[0]) for f in model_files]
        current_iter = max(iters)
        print(f">>> Resuming from Iteration: {current_iter}")
    else:
        current_iter = 0
        print(">>> No existing models found. Starting from Iteration 0.")

    while True:
        run_self_play(current_iter, args.games)
        run_training(current_iter)
        
        if current_iter >= 0:
            is_better = run_evaluation(current_iter, current_iter + 1)
            if not is_better:
                print("Model didn't improve, retraining with more data...")
                continue 
        
        current_iter += 1