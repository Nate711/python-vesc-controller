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

def open_serial(dev,baudrate):
    # dev = '/dev/cu.usbmodem3018921'
    ser_port = serial.Serial(dev, baudrate, timeout=0.001)  # Establish the connection on a specific port

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

### CONFIGURATION ###
BAUDRATE = 500000
teensy_serial_port = '/dev/cu.usbmodem3018921'

data_folder = 'data/'


phase0_angle = 180.0
phase1_angle = phase0_angle
phase2_angle = 90.0
phase3_angle = 180.0

I_max = 15

teensy = open_serial(teensy_serial_port,BAUDRATE)

for kd in [0.0005,0.001,0.002,0.004, 0.008]:
    for kp in [0.025, 0.05,0.1,0.2, 0.3]:
        reset_VESC(teensy)
        begin_VESC(teensy)

        # Phase 0
        # Send initial 180 deg command and wait to settle

        send_vesc_command(teensy, phase0_angle, kp, kd)

        # Phase 1: 0 - 500 ms:    Wait 500ms at 180 deg
        # Phase 2: 500 - 1000ms:  Wait 500ms at 90 deg
        # Phase 3: 1000 - 1500ms: Wait 500ms at 180 deg for settling (DO I NEED THIS)

        # Prepare phase variables
        phase1 = True
        phase2 = False
        phase3 = False

        record_size = 1500
        # Store encoder readings in this variable
        time_angle = np.empty((record_size, 2))
        time_angle.fill(np.nan)

        computer_timestamps = np.empty((record_size,1))
        computer_timestamps.fill(np.nan)

        # Data entry index
        index = 0

        # Variable to build messages with
        message_string = ''

        # Wait for the motor to settle after the 180deg command then begin encoder
        sleep(0.5)
        begin_encoder(teensy)

        # Start step function program
        start_time = time.time() * 1000
        teensy_start_time = np.nan

        while phase1 or phase2 or phase3:
            now = time.time() * 1000

            # Phase 2 transition
            if now - start_time > 500 and phase1:
                send_vesc_command(teensy, phase2_angle, kp,kd)

                phase1 = False
                phase2 = True
                phase3 = False

            # Phase 3 transition
            if now - start_time > 1000 and phase2:
                # Enter phase 3
                send_vesc_command(teensy, phase3_angle,kp,kd)

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
                    # print('Message: ' + message_string + ' at comp time: ' + str(now - start_time))
                    # print('Elapsed: ' + str())

                    # Attempt to parse a float from the received string
                    time_angle_datum = message_string.split(' ')
                    try:
                        angle1 = float(time_angle_datum[1])
                        time1 = float(time_angle_datum[0])

                        if np.isnan(teensy_start_time):
                            teensy_start_time = time1


                        # Debug print
                        # print('Elapsed: ' + str(time1-teensy_start_time) + ' Angle: ' + str(angle1))

                        # Record latest angle and teensy timestamp
                        time_angle[index,:] = [time1 - teensy_start_time,angle1]

                        # Record time when computer received this datum
                        computer_timestamps[index] = now - start_time

                        index += 1
                    except:
                        # Don't do anything with non-float messages from the Teensy
                        print(message_string)
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

        print('Done')
        print('Recorded Data:')
        print(time_angle)
        print(computer_timestamps)

        np.save('{}time_angle_I{}p{}d{}.npy'.format(data_folder,int(I_max),int(kp*1000),int(kd*1000)),time_angle)
        np.save('{}computer_timestamps_I{}p{}d{}.npy'.format(data_folder,int(I_max),int(kp*1000),int(kd*1000)),computer_timestamps)


# Close the serial port
teensy.close()


# Plot vesc encoder data
# plt.figure()
# plt.plot(time_angle[:,0],time_angle[:,1])
# plt.show()

