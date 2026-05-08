import Jetson.GPIO as GPIO
import time

class GPIO_Module:
    def __init__(self):
        GPIO.setmode(GPIO.BOARD)  
        self.push_button_pin = 13  
        # self.led_pins = [16, 18, 22, 29, 31, 36, 37]
        self.led_pins = [12, 15, 16, 18, 19, 21, 22]  

        GPIO.setup(self.push_button_pin, GPIO.IN)
        GPIO.setup(self.led_pins, GPIO.OUT)
        
        self.off_all_led()

    def wait_push(self):
        while GPIO.input(self.push_button_pin) == GPIO.LOW:
            time.sleep(0.01)
        print("Button Pushed")
        time.sleep(0.2)

    def on_all_led(self):
        GPIO.output(self.led_pins, GPIO.HIGH)
    
    def off_all_led(self):
        GPIO.output(self.led_pins, GPIO.LOW)

    def on_led(self, col_index):
        GPIO.output(self.led_pins[col_index], GPIO.HIGH) # Action: on LED
        print(f"LED at column {col_index} (Pin {self.led_pins[col_index]}) is ON")

    def off_led(self, col_index):
        GPIO.output(self.led_pins[col_index], GPIO.LOW) # Action: off LED
        print(f"LED at column {col_index} (Pin {self.led_pins[col_index]}) is OFF")

    def show_winner(self, winner):
        self.off_all_led()
        
        if winner == 1:
            blink_pin = self.led_pins[0]
        elif winner == 2:
            blink_pin = self.led_pins[6]
        else:
            blink_pin = self.led_pins[3]
        
        for i in range(20): 
            
            GPIO.output(self.led_pins, GPIO.LOW) 
            
            # Turn on only the winning LED
            GPIO.output(blink_pin, GPIO.HIGH)
            time.sleep(0.5)
            
            # Turn off the winning LED
            GPIO.output(blink_pin, GPIO.LOW)
            time.sleep(0.5)
        

    def showConfirmButton(self):
        self.on_all_led()
        time.sleep(1)
        self.off_all_led()

    def cleanup(self):
        self.off_all_led()
        GPIO.cleanup()
        print("GPIO Cleanup")


    # def on_all_led(self):
    #     for pin in self.led_pins:
    #         GPIO.output(pin, GPIO.HIGH)
    
    # def off_all_led(self):
    #     for pin in self.led_pins:
    #         GPIO.output(pin, GPIO.LOW)

    # def on_led(self, col_index):
    #     if 0 <= col_index < len(self.led_pins):
    #         GPIO.output(self.led_pins[col_index], GPIO.HIGH) # Action: on LED
    #         print(f"LED at column {col_index} (Pin {self.led_pins[col_index]}) is ON")

    # def off_led(self, col_index):
    #     if 0 <= col_index < len(self.led_pins):
    #         GPIO.output(self.led_pins[col_index], GPIO.LOW) # Action: off LED
    #         print(f"LED at column {col_index} (Pin {self.led_pins[col_index]}) is OFF")
