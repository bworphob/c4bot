import os
import sys
import numpy as np
import time

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
# sys.path.append(os.path.join(os.getcwd(), 'src'))
# --- ปรับปรุงส่วนนี้เพื่อให้รันได้ทั้งในและนอก src ---
# หาตำแหน่งของไฟล์นี้
current_file_path = os.path.abspath(__file__)
# หาตำแหน่งโฟลเดอร์ c4bot (Root)
# ถ้าอยู่ใน src/ โฟลเดอร์หลักคือ dirname ของ dirname
# ถ้าอยู่ Root โฟลเดอร์หลักคือ dirname
base_dir = os.path.dirname(current_file_path)
if os.path.basename(base_dir) == 'src':
    base_dir = os.path.dirname(base_dir)

src_path = os.path.join(base_dir, 'src')

# เพิ่ม Path เข้าไปเพื่อให้ import module อื่นๆ เจอ
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# ----------------------------------------------

from GameBoard.GameBoard import Connect4Board
from Reinforcement.playerVsAI_TRT import TRTBrainWrapper
from GPIO.jetson_hardware import GPIO_Module
from ImageProcess.Image_Processing import Image_Processing
from luma.oled.device import sh1106
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from Reinforcement.players.ZeroPlayer import TRTPlayer

# def update_oled(device, title, msg, col=None):
#     with canvas(device) as draw:
#         draw.text((5, 5), title, fill="white")
#         draw.text((5, 20), msg, fill="white")
        
#         if col is not None:
#             draw.rectangle((40, 35, 80, 60), outline="white")
#             draw.text((48, 42), "C{}".format(col), fill="white")

def update_oled(device, title, msg, col=None, win_chance=None):
    with canvas(device) as draw:
        draw.text((5, 5), title, fill="white")
        draw.text((5, 20), msg, fill="white")
        
        # ถ้ามีค่าโอกาสชนะ ให้แสดงเป็น % ที่มุมขวาล่าง
        if win_chance is not None:
            draw.text((5, 45), f"AI Win: {win_chance:.1f}%", fill="white")
            
        if col is not None:
            draw.rectangle((85, 35, 120, 60), outline="white") # ขยับกรอบหลบตัวเลข
            draw.text((93, 42), "C{}".format(col), fill="white")

# def run_main():
#     # 1. Setup All Modules
#     hw = GPIO_Module()
#     vision = Image_Processing()
#     engine_path = "src/Reinforcement/Models/model_v6.engine"
#     ai_brain = TRTBrainWrapper(engine_path)
    
#     serial = i2c(port=1, address=0x3C)
#     device = sh1106(serial)

#     # 2. เริ่มเกม
#     print("--- C4BOT VISION MAIN GAME ---")
#     try:
#         first_p = int(input("Who starts first? (1: Human, 2: AI): "))
#     except ValueError:
#         first_p = 1
        
#     game = Connect4Board(first_player=first_p)

#     try:
#         while not game.isEnd:
#             hw.off_all_led()
            
#             if game.current_turn == 1: # --- ตาเรา (Human) ---
#                 print("\n>>> YOUR TURN")
#                 update_oled(device, "ROUND {}".format(game.round), "YOUR TURN\nDrop & Push")
#                 hw.wait_push()
                
#                 print("Scanning board...")
#                 update_oled(device, "SCANNING...", "Please wait")
#                 game.board = vision.scan_board()
                
#                 next_turn = 2 # เตรียมสลับตา
                
#             else: # --- ตา AI ---
#                 print("\n>>> AI THINKING...")
#                 update_oled(device, "AI THINKING...", "Processing...")
#                 policy = ai_brain.predict(game)
#                 valid_actions = game.validAction()
                
#                 masked_policy = np.full(policy.shape, -np.inf)
#                 masked_policy[valid_actions] = policy[valid_actions]
#                 move = int(np.argmax(masked_policy))
                
#                 print("AI SUGGESTS: Column {}".format(move))
#                 hw.on_led(move) 
#                 update_oled(device, "AI SUGGESTS:", "Drop for AI\n& Push", col=move)
                
#                 hw.wait_push()
#                 hw.off_all_led()
                
#                 print("Scanning AI move...")
#                 game.board = vision.scan_board()
                
#                 next_turn = 1 # เตรียมสลับตา
#                 game.round += 1

#             game.showBoard()

#             # --- ส่วนเช็คจบเกม (ใช้ตัวแปรชั่วคราวเช็คชนะ) ---
#             valid_moves = game.validAction()
#             found_winner = False
            
#             for r in range(6):
#                 for c in range(7):
#                     piece = game.board[r, c]
#                     if piece != 0:
#                         # บังคับ turn ชั่วคราวเพื่อให้ฟังก์ชัน check ของ GameBoard ทำงานได้
#                         game.current_turn = piece 
#                         if game.checkEndGameFromInsert(r, c) != 0:
#                             game.isEnd = True
#                             game.winner = piece
#                             found_winner = True
#                             break
#                 if found_winner: break
            
#             # --- จัดการ Turn และจบเกม ---
#             if not game.isEnd:
#                 game.current_turn = next_turn # สลับตาจริงที่นี่
#                 if len(valid_moves) == 0:
#                     game.isEnd = True
#                     game.winner = 0
#             # ถ้า game.isEnd เป็น True แล้ว game.winner จะค้างค่าที่เจอในลูปไว้

#         # 3. สรุปผลหลังจบ Loop
#         winner_text = "YOU WON!" if game.winner == 1 else "AI WON!"
#         if game.winner == 0: winner_text = "DRAW!"
        
#         print("\nGAME OVER: {}".format(winner_text))
#         update_oled(device, "GAME OVER", winner_text)
#         hw.off_all_led()
#         hw.show_winner(game.winner)

#     except KeyboardInterrupt:
#         hw.cleanup()




def run_main():
    
    hw = GPIO_Module()
    vision = Image_Processing()
    # engine_path = "src/Reinforcement/Models/model_v8.engine"
    engine_path = os.path.join(base_dir, "src/Reinforcement/Models/model_v4.engine")
    ai_brain = TRTBrainWrapper(engine_path)
    ZeroAI = TRTPlayer(ai_brain, n_simulations=400) 
    
    serial = i2c(port=1, address=0x3C)
    device = sh1106(serial)

    
    print("--- C4BOT VISION READY ---")
    print("Waiting for first push to determine who starts...")
    update_oled(device, "C4BOT READY", "Insert to Start\nor Push for AI")
    
    # wait for first push to determine order
    hw.wait_push()
    print("Initial scanning...")
    update_oled(device, "SCANNING...", "Determining order")
    
    
    first_scan = vision.scan_board()
    
    # check wether there are coins on the board to determine who starts first
    coin_count = np.count_nonzero(first_scan)
    
    if coin_count > 0:
        # if there are coins --> human started first
        print("Human started first.")
        game = Connect4Board(first_player=1) 
        game.board = first_scan
        # game.current_turn = 2 # Force AI to be the second player since human started first
        game.current_turn = 1 # Human goes first
        game.showBoard()
    else:
        # if no coins --> AI started first
        print("AI starts first.")
        game = Connect4Board(first_player=2)
        game.current_turn = 2 # Re-confirm AI starts first
        

    
    try:
        while not game.isEnd:
            hw.off_all_led()
            
            if game.current_turn == 1: # Human turn
                print("\n>>> YOUR TURN")
                update_oled(device, "ROUND {}".format(game.round), "YOUR TURN\nDrop & Push")
                hw.wait_push()
                
                print("Scanning board...")
                update_oled(device, "SCANNING...", "Please wait")
                game.board = vision.scan_board()
                next_turn = 2 
                
            # else: # AI turn
            #     print("\n>>> AI THINKING...")
            #     update_oled(device, "AI THINKING...", "Processing...")
            #     # policy = ai_brain.predict(game)
            #     policy, value = ai_brain.predict(game)####
            #     ai_win_percent = (1 + value) / 2 * 100####
            #     valid_actions = game.validAction()
                
            #     masked_policy = np.full(policy.shape, -np.inf)
            #     masked_policy[valid_actions] = policy[valid_actions]
            #     move = int(np.argmax(masked_policy))
                
            #     print(f"AI Win Chance: {ai_win_percent:.1f}%")####
            #     # hw.on_led(move)####
            #     # print("AI SUGGESTS: Column {}".format(move))
            #     update_oled(device, "AI SUGGESTS:", "Drop for AI\n& Push", col=move, win_chance=ai_win_percent)
            #     hw.on_led(move) 
            #     # update_oled(device, "AI SUGGESTS:", "Drop for AI\n& Push", col=move)
                
            #     hw.wait_push()
            #     hw.off_all_led()
                
            #     print("Scanning AI move...")
            #     game.board = vision.scan_board()
            #     next_turn = 1 
            #     game.round += 1

            else: # AI turn
                # --- โค้ดเก่า (Direct Predict) ---
                # print("\n>>> AI THINKING...")
                # update_oled(device, "AI THINKING...", "Processing...")
                # policy, value = ai_brain.predict(game)
                # ai_win_percent = (1 + value) / 2 * 100
                # valid_actions = game.validAction()
                # masked_policy = np.full(policy.shape, -np.inf)
                # masked_policy[valid_actions] = policy[valid_actions]
                # move = int(np.argmax(masked_policy))
                
                # --- โค้ดใหม่ (ใช้ MCTS ผ่าน TRTPlayer) ---
                print("\n>>> AI THINKING (MCTS Search)...")
                update_oled(device, "AI THINKING...", "MCTS Searching")
                
                # เรียกใช้ MCTS Search 400 รอบ (ZeroAI คืออินสแตนซ์ของ TRTPlayer)
                move, mcts_policy = ZeroAI.act(game, tau=0) 
                
                # ยังคงดึงค่า Value จากการ Predict สถานะปัจจุบันเพื่อโชว์ % โอกาสชนะบน OLED
                _, value = ai_brain.predict(game)
                ai_win_percent = (1 + value) / 2 * 100
                # ---------------------------------------

                print(f"AI Win Chance: {ai_win_percent:.1f}%")
                print(f"AI RECOMMENDS: Column {move}")
                
                # แสดงผลบน OLED และเปิดไฟ LED
                update_oled(device, "AI SUGGESTS:", "Drop for AI\n& Push", col=move, win_chance=ai_win_percent)
                hw.on_led(move) 
                
                hw.wait_push()
                hw.off_all_led()
                
                print("Scanning AI move...")
                game.board = vision.scan_board()
                next_turn = 1 
                game.round += 1

            game.showBoard()

            # --- check for winner part ---
            valid_moves = game.validAction()
            found_winner = False
            for r in range(6):
                for c in range(7):
                    piece = game.board[r, c]
                    if piece != 0:
                        game.current_turn = piece 
                        if game.checkEndGameFromInsert(r, c) != 0:
                            game.isEnd = True
                            game.winner = piece
                            found_winner = True
                            break
                if found_winner: break
            
            if not game.isEnd:
                game.current_turn = next_turn 
                if len(valid_moves) == 0:
                    game.isEnd = True
                    game.winner = 0

        # End
        winner_text = "YOU WON!" if game.winner == 1 else "AI WON!"
        if game.winner == 0: winner_text = "DRAW!"
        print("\nGAME OVER: {}".format(winner_text))
        update_oled(device, "GAME OVER", winner_text)
        hw.off_all_led()
        hw.show_winner(game.winner)

    except KeyboardInterrupt:
        hw.cleanup()

if __name__ == "__main__":
    run_main()