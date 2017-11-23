import serial
import os, sys, time
from time import sleep
import matplotlib.pyplot as plt
import numpy as np


def bad_angle(angle):
    if angle < 30:
        print('BADDD ANGLEEE')
        return True
    return False


def open_serial():
    dev = '/dev/cu.usbmodem3018921'
    ser_port = serial.Serial(dev, 115200, timeout=0.001)  # Establish the connection on a specific port

    sleep(1)  # The Arduino needs 5 seconds to wake up unless the code is uploaded as hex.
    flush_all(ser_port)

    return ser_port


# clear the serial buffer and force output to the screen from stdout
def flush_all(ser_port):
    ser_port.flush()
    sys.stdout.flush()  # This forces the output to the screen (avoid delays)


# Reset the program
def reset_VESC(ser_port):
    ser_port.write(b'r\n')


# Put vesc into running mode (BEGIN program)
def begin_VESC(ser_port):
    ser_port.write(b'b\n')


# Start encoder readings
def begin_encoder(ser_port):
    ser_port.write(b'e\n')


def stop_encoder(ser_port):
    ser_port.write(b'!e\n')


def estop_VESC(ser_port):
    ser_port.write(b's\n')


# Send vesc command
def send_vesc_command(ser_port, pos, kp, kd):
    ser_port.write(('G1 X' + str(pos) + ' P' + str(kp) + ' D' + str(kd)).encode('utf-8'))


teensy = open_serial()
reset_VESC(teensy)
sleep(0.1)
begin_VESC(teensy)
sleep(0.1)

# Phase 0
# Send initial 180 deg command and wait to settle

send_vesc_command(teensy, 180.0, 0.10, 0.001)

# Phase 1: 0 - 500 ms:    Wait 500ms at 180 deg
# Phase 2: 500 - 1000ms:  Wait 500ms at 90 deg
# Phase 3: 1000 - 1500ms: Wait 500ms at 180 deg for settling (DO I NEED THIS)

# Prepare phase variables
phase1 = True
phase2 = False
phase3 = False

# Store encoder readings in this variable
data = np.zeros((1500, 1))

# Data entry index
index = 0

# Variable to build messages with
message_string = ''

# Wait for the motor to settle after the 180deg command then begin encoder
sleep(0.5)
begin_encoder(teensy)

# Start step function program
start_time = time.time() * 1000
while phase1 or phase2 or phase3:
    now = time.time() * 1000

    # Phase 2 transition
    if now - start_time > 500 and phase1:
        send_vesc_command(teensy, 90.0, 0.1, 0.001)

        phase1 = False
        phase2 = True
        phase3 = False

    # Phase 3 transition
    if now - start_time > 1000 and phase2:
        # Enter phase 3
        send_vesc_command(teensy, 180.0, 0.1, 0.001)

        phase1 = False
        phase2 = False
        phase3 = True

    # Exit transition
    if now - start_time > 1500:
        phase1 = False
        phase2 = False
        phase3 = False
        break

    # check if data has been received
    if teensy.inWaiting() > 0:
        # read one byte from the buffer
        in_byte = teensy.read().decode("utf-8")

        # Check if a full line has been sent
        if in_byte == '\n':
            # Debug print
            # print('Message: ' + message_string + ' ' + str(len(message_string)), sep='', end='')

            # Attempt to parse a float from the received string
            time_angle = message_string.split(' ')
            try:
                angle = float(time_angle[0])

                # Debug print
                # print(' Angle: ' + str(angle))

                data[index] = angle
                index += 1
            except:
                # Don't do anything with non-float messages from the Teensy
                # print(' Not Float ')
                pass

            # reset message_string
            message_string = ''

        # Full line hasn't been sent so continue to build up the message,
        # as long as the received char isn't a line feed
        elif in_byte != '\r':
            message_string += in_byte

# Stop encoder readings
stop_encoder(teensy)

# Wait 0.1s
sleep(0.1)

# Put into ESTOp
estop_VESC(teensy)

# Close the serial port
teensy.close()

print("Done")

# Plot vesc encoder data
plt.plot(data)
plt.show()
