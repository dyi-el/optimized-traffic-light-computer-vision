import pyfirmata
import time
if __name__ == '__main__':
    board = pyfirmata.Arduino('/dev/tty.usbmodem1101')
    print("Communication Successfully started")
    
    while True:
        board.digital[12].write(1)
        board.digital[8].write(1)
        board.digital[7].write(1)
        time.sleep(1)
        board.digital[12].write(0)
        board.digital[8].write(0)
        board.digital[7].write(0)
        time.sleep(1)