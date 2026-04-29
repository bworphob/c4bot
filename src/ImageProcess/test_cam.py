import cv2
import numpy as np
import os
import sys

# แทรก Path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from ImageProcess.Image_Processing import Image_Processing

def main():
    img_proc = Image_Processing()
    
    print("Capturing image...")
    raw_frame = img_proc.capture()
    if raw_frame is None:
        print("Error: Cannot capture image from camera (check index)")
        return

    # 1. ภาพหลังทำ Perspective Transform
    M = cv2.getPerspectiveTransform(img_proc.src_pts, img_proc.dest_pts)
    warped = cv2.warpPerspective(raw_frame, M, (img_proc.width, img_proc.height))
    cv2.imwrite('debug_1_warped.jpg', warped)

    # 2. ภาพหลัง Blur
    blurred = cv2.blur(warped, (12, 12))
    cv2.imwrite('debug_2_blurred.jpg', blurred)

    # 3. ภาพหลัง Convert to HSV
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # หมายเหตุ: ภาพ HSV จะดูสีเพี้ยนๆ เมื่อเซฟเป็น JPG เป็นเรื่องปกติครับ
    cv2.imwrite('debug_3_hsv.jpg', hsv_frame)

    # 4. ภาพหลังทำ Morphological (ดึงมา 1 ช่องเพื่อเทส)
    # ทดลองสแกนทั้งบอร์ดแล้วเซฟ Mask ของช่องแรกที่มีหมาก
    mask_yellow_full = cv2.inRange(hsv_frame, img_proc.yellow_low, img_proc.yellow_up)
    mask_red_full = cv2.inRange(hsv_frame, img_proc.red_low, img_proc.red_up)
    
    # ทำ Morphological Close
    morphed_y = cv2.morphologyEx(mask_yellow_full, cv2.MORPH_CLOSE, img_proc.kernel)
    morphed_r = cv2.morphologyEx(mask_red_full, cv2.MORPH_CLOSE, img_proc.kernel)
    
    cv2.imwrite('debug_4_mask_yellow.jpg', morphed_y)
    cv2.imwrite('debug_4_mask_red.jpg', morphed_r)

    print("Debug images saved: debug_1 to debug_4")
    
    # ทดสอบ Scan Board ออกมาเป็น Matrix
    matrix = img_proc.scan_board()
    print("\nScanned Board Matrix:")
    print(matrix)

if __name__ == "__main__":
    main()