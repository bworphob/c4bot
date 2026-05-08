

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
from Reinforcement.players.ZeroPlayer import TRTPlayer

def update_oled(device, title, msg, col=None, conf=None):
    with canvas(device) as draw:
        draw.text((5, 5), title, fill="white")
        draw.text((5, 20), msg, fill="white")
        # Display confidence value if available
        if conf is not None:
            draw.text((5, 45), "AI_win: {:.1f}%".format(conf), fill="white")
        # Display column frame if available
        if col is not None:
            draw.rectangle((85, 35, 120, 60), outline="white")
            draw.text((93, 42), "C{}".format(col), fill="white")

def run_test():
    # 1. Setup Hardware & AI
    hw = GPIO_Module()
    engine_path = "src/Reinforcement/Models/model_v4.engine"
    ai_brain = TRTBrainWrapper(engine_path)
    ZeroAI_MCTS = TRTPlayer(ai_brain, n_simulations=400)

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
                
            # else: # AI
            #     update_oled(device, "AI THINKING...", "Processing...")
            #     print("AI is calculating...")
                
            #     # AI calculates move
            #     # policy = ai_brain.predict(game)
            #     policy, value = ai_brain.predict(game)
            #     ai_win_percent = (value + 1) / 2 * 100
            #     valid_actions = game.validAction()
                
            #     masked_policy = np.full(policy.shape, -np.inf)
            #     masked_policy[valid_actions] = policy[valid_actions]
            #     move = int(np.argmax(masked_policy))
                
            #     # Display Hardware
            #     print("AI RECOMMENDS: Column {}".format(move))
            #     print("AI Confidence: {:.2f}%".format(ai_win_percent))
            #     update_oled(device, "AI THINKING", "Confidence:", conf=ai_win_percent)######
            #     hw.on_led(move) 
            #     # update_oled(device, "AI SUGGESTS:", "Drop for AI\n& Push", col=move) 
            #     update_oled(device, "AI SUGGESTS", "Drop & Push", col=move, conf=ai_win_percent) ######
                
            #     # Wait for human to drop AI's coin and push to switch turn
            #     success = False
            #     while not success:
            #         try:
            #             ai_move_input = int(input("Input AI move ({}): ".format(move)))
            #             if game.insertColumn(ai_move_input):
            #                 print("PUSH button.")
            #                 success = True
            #             else:
            #                 print("Sync Error: Column Full!")
            #         except ValueError:
            #             pass
                
            #     # push to confirm move and switch turn
            #     hw.wait_push()
            #     hw.off_all_led()



            else: # AI
                # --- previous direct-predict path ---
                # update_oled(device, "AI THINKING...", "Processing...")
                # print("AI is calculating...")
                # policy, value = ai_brain.predict(game)
                # ai_win_percent = (value + 1) / 2 * 100
                # valid_actions = game.validAction()
                # masked_policy = np.full(policy.shape, -np.inf)
                # masked_policy[valid_actions] = policy[valid_actions]
                # move = int(np.argmax(masked_policy))

                # --- current path: AI uses MCTS search ---
                update_oled(device, "AI THINKING...", "Searching")
                print("AI is calculating with MCTS...")

                # use MCTS via ZeroAI_MCTS defined at the start of run_test
                move, mcts_policy = ZeroAI_MCTS.act(game, tau=0)

                # get current value estimate and display win probability
                _, value = ai_brain.predict(game)
                ai_win_percent = (value + 1) / 2 * 100
                # --------------------------------------------------

                # Display Hardware (can keep using move and ai_win_percent)
                print("AI RECOMMENDS: Column {}".format(move))
                print("AI Confidence: {:.2f}%".format(ai_win_percent))
                
                update_oled(device, "AI THINKING", "Confidence:", conf=ai_win_percent)
                hw.on_led(move) 
                # update_oled(device, "AI SUGGESTS", "Drop & Push", col=move, conf=ai_win_percent) 
                update_oled(device, "AI SUGGESTS:", "Drop for AI\n& Push", col=move, conf=ai_win_percent)
                # wait for the human to drop AI's move and confirm with the button
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