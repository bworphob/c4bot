import cv2
import time
import numpy as np



class Image_Processing:
    def __init__(self):
        # Init color value (will be updated after execution calibration)
        self.red_low = np.array([140, 110, 25])
        self.red_up = np.array([180, 255, 255])
        self.yellow_low = np.array([5, 20, 60])
        self.yellow_up = np.array([70, 200, 255])

        # Init image
        self.src_pts = np.float32([[0, 21], [35, 463], [629, 20], [590, 465]])
        self.width, self.height = 640, 480
        self.dest_pts = np.float32([[0, 0], [0, self.height], [self.width, 0], [self.width, self.height]])

        # Init kernel for morphological
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        
    def capture(self):
        cap = cv2.VideoCapture(0)
        # warm up
        for _ in range(20):
            ret, frame = cap.read()

        ret_final, frame = cap.read()
        cap.release()

        if ret_final:
            cv2.imwrite('last_capture.jpg', frame)
            return frame
        return None
    
    def pre_process(self, frame):
        # PerspectiveTransform + blur = to get top down view and reduce reflection by blurring the image
        M = cv2.getPerspectiveTransform(self.src_pts, self.dest_pts)
        warped = cv2.warpPerspective(frame, M, (self.width, self.height))
        blurred = cv2.blur(warped, (12, 12))
        return blurred
    
    def get_slot(self, hsv_img, row, col):
        step_x = self.width // 7
        step_y = self.height // 6
        slot = hsv_img[row*step_y : (row+1)*step_y, col*step_x : (col+1)*step_x]
        return slot
    
    def calibration(self):
        # Calibration process: capture image, pre-process it,
        # and randomly sample 25 pixels in each of the 12 known slots (6 rows x 2 columns) to find the average HSV values for Red and Yellow coins. 
        print("Starting Calibration...")
        raw_frame = self.capture()
        processed = self.pre_process(raw_frame)
        hsv_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        # Randomly check the color from the slots we know in advance.
        for r in range(6):
            for j in range(2): # Check 2 columns (e.g., leftmost and rightmost).
                c = j * 6 
                slot = self.get_slot(hsv_frame, r, c)
                
                # Randomly select 25 points in that space to find the average color value.
                h_rand = np.random.randint(5, slot.shape[0]-5, 25)
                w_rand = np.random.randint(5, slot.shape[1]-5, 25)

                # Choose to update either Red or Yellow based on position (chessboard pattern).
                is_red = (r + j) % 2 == 0
                target_low = self.red_low if is_red else self.yellow_low
                target_up = self.red_up if is_red else self.yellow_up

                for i in range(25):
                    pixel_hsv = slot[h_rand[i], w_rand[i]]
                    # Update the Low value (find the smallest value) and the Up value (find the largest value).
                    for channel in range(3):
                        if pixel_hsv[channel] < target_low[channel]:
                            target_low[channel] = pixel_hsv[channel]
                        if pixel_hsv[channel] > target_up[channel]:
                            target_up[channel] = pixel_hsv[channel]
        
        print("Calibration Finished")

    def scan_board(self):
        # return 6x7 matrix of the board state (0 = empty, 1 = yellow coin, 2 = red coin)
        raw_frame = self.capture()
        processed = self.pre_process(raw_frame)
        hsv_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        board_matrix = np.zeros((6, 7), dtype=np.int8)

        for r in range(6):
            for c in range(7):
                slot = self.get_slot(hsv_frame, r, c)

                # 1. create mask for seperating yellow and red coins
                mask_y = cv2.inRange(slot, self.yellow_low, self.yellow_up)
                mask_r = cv2.inRange(slot, self.red_low, self.red_up)

                # 2. Morphology: MORPH_CLOSE (Dilation -> Erosion)
                mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, self.kernel)
                mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, self.kernel)

                # 3. The decision is based on the number of colored pixels found.
                if np.sum(mask_y != 0) > 1000:
                    board_matrix[r, c] = 1 # Player
                elif np.sum(mask_r != 0) > 1000:
                    board_matrix[r, c] = 2 # AI
                else:
                    board_matrix[r, c] = 0 # Empty

        return board_matrix