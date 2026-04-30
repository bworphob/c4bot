# import os
# import sys
# import numpy as np
# import time

# # แทรก Path เพื่อให้เรียกใช้ Module ใน src ได้
# sys.path.append(os.path.join(os.getcwd(), 'src'))

# from GameBoard.GameBoard import Connect4Board
# from Reinforcement.playerVsAI_TRT import TRTBrainWrapper
# from GPIO.jetson_hardware import GPIO_Module
# # หมายเหตุ: ถ้าใช้ OLED ให้ import luma ตรงนี้ด้วย
# from luma.oled.device import sh1106
# from luma.core.interface.serial import i2c
# from luma.core.render import canvas

# def update_oled(device, title, msg, col=None):
#     with canvas(device) as draw:
#         draw.text((5, 5), title, fill="white")
#         draw.text((5, 20), msg, fill="white")
#         if col is not None:
#             draw.rectangle((40, 35, 80, 60), outline="white")
#             draw.text((48, 42), f"C{col}", fill="white")

# def run_test():
#     # 1. Setup Hardware & AI
#     hw = GPIO_Module()
#     engine_path = "src/Reinforcement/Models/model_v6.engine"
#     ai_brain = TRTBrainWrapper(engine_path)
    
#     # Setup OLED
#     serial = i2c(port=1, address=0x3C)
#     device = sh1106(serial)

#     # 2. เริ่มเกม
#     first_p = int(input("Who starts first? (1: Human, 2: AI): "))
#     game = Connect4Board(first_player=first_p)

#     try:
#         while not game.isEnd:
#             game.showBoard()
#             hw.off_all_led()
            
#             if game.current_turn == 1: # ตาเรา (Human)
#                 print(">>> YOUR TURN: Input column in console, drop coin, THEN push button.")
#                 update_oled(device, f"ROUND {game.round}", "YOUR TURN\nInput & Push")
                
#                 col = int(input("Enter column (0-6): "))
#                 if col not in game.validAction():
#                     print("Invalid column!")
#                     continue
                
#                 # จำลองการหยอดหมาก
#                 game.insertColumn(col)
#                 hw.wait_push() # รอเรากดปุ่มเพื่อยืนยันว่าหยอดแล้วและเปลี่ยนตา
                
#             else: # ตา AI
#                 print("AI is thinking...")
#                 update_oled(device, "AI THINKING...", "Processing...")
                
#                 # AI คำนวณ
#                 policy = ai_brain.predict(game)
#                 valid_actions = game.validAction()
#                 masked_policy = np.full(policy.shape, -np.inf)
#                 masked_policy[valid_actions] = policy[valid_actions]
#                 move = np.argmax(masked_policy)
                
#                 # แสดงผลทาง Hardware
#                 print(f"AI RECOMMENDS: Column {move}")
#                 hw.on_led(move) # เปิดไฟคอลัมน์ที่แนะนำ
#                 update_oled(device, "AI SUGGESTS:", "Drop for AI\n& Push", col=move)
                
#                 # รอเราหยอดให้ AI และกดปุ่มเปลี่ยนตา
#                 hw.wait_push()
#                 game.insertColumn(move)

#         # จบเกม
#         game.showBoard()
#         winner_text = "YOU WON!" if game.winner == 1 else "AI WON!"
#         if game.winner == 0: winner_text = "DRAW!"
        
#         print(f"GAME OVER: {winner_text}")
#         update_oled(device, "GAME OVER", winner_text)
#         hw.show_winner(game.winner)

#     except KeyboardInterrupt:
#         hw.cleanup()

# if __name__ == "__main__":
#     run_test()






import os
import sys
import numpy as np
import time

# Append src path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from GameBoard.GameBoard import Connect4Board
from Reinforcement.playerVsAI_TRT import TRTBrainWrapper
from GPIO.jetson_hardware import GPIO_Module
from luma.oled.device import sh1106
from luma.core.interface.serial import i2c
from luma.core.render import canvas

def update_oled(device, title, msg, col=None):
    with canvas(device) as draw:
        draw.text((5, 5), title, fill="white")
        draw.text((5, 20), msg, fill="white")
        if col is not None:
            draw.rectangle((40, 35, 80, 60), outline="white")
            # เปลี่ยน f-string เป็น .format() เพื่อรองรับ Python 3.6 รุ่นเก่า
            draw.text((48, 42), "C{}".format(col), fill="white")

def run_test():
    # 1. Setup Hardware & AI
    hw = GPIO_Module()
    engine_path = "src/Reinforcement/Models/model_v8.engine"
    ai_brain = TRTBrainWrapper(engine_path)
    
    # Setup OLED (Bus 1, Address 0x3C)
    serial = i2c(port=1, address=0x3C)
    device = sh1106(serial)

    # 2. Start
    print("--- C4BOT Integration Test ---")
    try:
        first_p = int(input("Who starts first? (1: Human, 2: AI): "))
    except ValueError:
        first_p = 1
        
    game = Connect4Board(first_player=first_p)

    try:
        while not game.isEnd:
            game.showBoard()
            hw.off_all_led()
            
            if game.current_turn == 1: # Human
                update_oled(device, "ROUND {}".format(game.round), "YOUR TURN\nInput & Push")
                
                success = False
                while not success:
                    try:
                        col = int(input("Enter your column (0-6): "))
                        if game.insertColumn(col):
                            print("PUSH button.")
                            success = True
                        else:
                            print("Column FULL or Invalid! Try again.")
                    except ValueError:
                        print("Please enter a number 0-6.")
                
                # push to confirm move and switch turn
                hw.wait_push() 
                
            else: # AI
                update_oled(device, "AI THINKING...", "Processing...")
                print("AI is calculating...")
                
                # AI calculates move
                policy = ai_brain.predict(game)
                valid_actions = game.validAction()
                
                masked_policy = np.full(policy.shape, -np.inf)
                masked_policy[valid_actions] = policy[valid_actions]
                move = int(np.argmax(masked_policy))
                
                # Display Hardware
                print("AI RECOMMENDS: Column {}".format(move))
                hw.on_led(move) 
                update_oled(device, "AI SUGGESTS:", "Drop for AI\n& Push", col=move)
                
                # Wait for human to drop AI's coin and push to switch turn
                success = False
                while not success:
                    try:
                        ai_move_input = int(input("Input AI move ({}): ".format(move)))
                        if game.insertColumn(ai_move_input):
                            print("PUSH button.")
                            success = True
                        else:
                            print("Sync Error: Column Full!")
                    except ValueError:
                        pass
                
                # push to confirm move and switch turn
                hw.wait_push()
                hw.off_all_led()

        # End
        game.showBoard()
        winner_text = "YOU WON!" if game.winner == 1 else "AI WON!"
        if game.winner == 0: winner_text = "DRAW!"
        
        print("GAME OVER: {}".format(winner_text))
        update_oled(device, "GAME OVER", winner_text)
        hw.show_winner(game.winner)

    except KeyboardInterrupt:
        hw.cleanup()

if __name__ == "__main__":
    run_test()