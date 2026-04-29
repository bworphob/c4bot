# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

# แทรก Path เพื่อให้เรียกใช้ Module ใน src ได้ (เผื่อรันจาก Root ของโปรเจกต์)
# os.getcwd() จะคืนค่า /home/bworphob/c4bot
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ImageProcess.Image_Processing import Image_Processing

def main():
    # 1. เริ่มต้นระบบ Vision
    img_proc = Image_Processing()
    
    print("--- C4BOT Camera Pipeline & Scan Test ---")
    
    # 2. รันการบันทึกภาพในแต่ละ Stage (ฟังก์ชันที่เพื่อนเพิ่มมา)
    # มันจะสร้างโฟลเดอร์ pipeline_debug/ ให้เอง
    # ลองสแกนแถว 5 (ล่างสุด) คอลัมน์ 0 (ซ้ายสุด) เพื่อดูผลลัพธ์หมากตัวอย่าง
    print("\n[Stage 1] Saving pipeline images for debugging...")
    img_proc.save_pipeline_images(output_dir="pipeline_debug", example_row=5, example_col=0)
    
    # 3. ทดสอบการ Scan Board จริงๆ ออกมาเป็น Matrix
    print("\n[Stage 2] Scanning current board state...")
    matrix = img_proc.scan_board()
    
    print("\n" + "="*30)
    print(" SCANNED BOARD MATRIX")
    print(" (0:Empty, 1:Yellow, 2:Red)")
    print("="*30)
    print(matrix)
    print("="*30)

    print("\nNext Steps:")
    print("1. Open 'http://<JETSON_IP>:8000/pipeline_debug' in your browser.")
    print("2. Check '2_perspective.jpg' to see if the board is aligned.")
    print("3. Check '5_morph_yellow/red' to see if coins are detected clearly.")

if __name__ == "__main__":
    main()