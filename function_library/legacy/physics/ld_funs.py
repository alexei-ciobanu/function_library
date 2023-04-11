import serial
import time
import numpy as np

def calculate_crc(data, astype='int'):
    if isinstance(data, bytes):
        data = list(data)
    crc = 0
    for el in data:
        crc = crc16_update(crc, el)
        
    if astype == 'int':
        return crc
    elif astype == 'bytes':
        return int.to_bytes(crc, length=2, byteorder='little')

def crc16_update(crc, a, reverse_poly=0xA001):
    crc ^= a
    for i in range(0,8):
        if crc & 1:
            crc = (crc >> 1) ^ reverse_poly
        else:
            crc = (crc >> 1)
    return crc

def crc16(data):
    crc = calculate_crc(data, 'bytes')
    return data+crc

def crc_validate(data):
    crc = calculate_crc(data)
    if not crc:
        print('crc validation failed')
        return False
    else:
        return True
    
SERIAL_SLEEP = 0.003

def read_temperature(ser):
    time.sleep(SERIAL_SLEEP)
    a = ser.write(crc16(b'TCA'))
    time.sleep(SERIAL_SLEEP)
    b = ser.read_all()
    crc_validate(b)
    dd = b[3:5]
    idd = int.from_bytes(dd, byteorder='big')
    conversion = 0.0625
    temperature = idd*conversion
    ser.flushInput()
    ser.flushOutput()
    return temperature

def read_current(ser):
    ic = 293
    time.sleep(SERIAL_SLEEP)
    a = ser.write(crc16(b'Ar\x00\x00'))
    time.sleep(SERIAL_SLEEP)
    b = ser.read_all()
    val = int.from_bytes(b[1:3],byteorder='big')
    i0 = val*ic / 4096
    ser.flushInput()
    ser.flushOutput()
    return i0

def set_current(ser, i0):
    ic = 293
    val = int(np.round(i0/ic * 4096))
    val_bytes = int.to_bytes(val, length=2, byteorder='big')
    time.sleep(SERIAL_SLEEP)
    a = ser.write(crc16(b'Aw'+val_bytes))
    ser.flushInput()
    ser.flushOutput()
    return val

def handshake(ser):
    ser.flushInput()
    ser.flushOutput()
    time.sleep(SERIAL_SLEEP)
    a = ser.write(b'Start')
    time.sleep(SERIAL_SLEEP)
    b = ser.read_all()
    ser.flushInput()
    ser.flushOutput()
    return a,b

class LiquidLens(serial.Serial):
    def __init__(self, path='/dev/ttyACM0', *args, **kwargs):
        self.path = path
        super().__init__(path, 115200, timeout=1, 
                         parity=serial.PARITY_NONE, stopbits=1, bytesize=8, *args, **kwargs)

try:

    LD0 = '/dev/ttyACM0'
    LD1 = '/dev/ttyACM1'

    with LiquidLens(LD0) as ld0:
        a = handshake(ld0)
        b = read_current(ld0)
        c = read_temperature(ld0)
        e = set_current(ld0, 10)
        f = read_current(ld0)
        
    print(a)
    print(c)
    print(f)

    with LiquidLens(LD1) as ld1:
        a = handshake(ld1)
        b = read_current(ld1)
        c = read_temperature(ld1)
        e = set_current(ld1, 15)
        f = read_current(ld1)
        
    print(a)
    print(c)
    print(f)

except Exception as e:
    print('Warning: ', str(e))


def set_dc(ser, i0):
    time.sleep(0.1)
    a4 = ser.write(crc16(b'MwSD'))
    time.sleep(0.1)
    a1 = ser.write(crc16(b'PwFA'+b'\x00\x00\x00\x00'))
    time.sleep(0.1)
    b4 = ser.read_all()
    val = set_current(ser, i0)
    ser.flushInput()
    ser.flushOutput()
    return val

def set_sinusoidal(ser, f=1, iL=0, iU=290):
    ic = 293
    iL_val = int(np.round(iL/ic * 4096))
    iU_val = int(np.round(iU/ic * 4096))
    iL_bytes = int.to_bytes(iL_val, length=2, byteorder='big')
    iU_bytes = int.to_bytes(iU_val, length=2, byteorder='big')
    
    f = int(f)
    f_bytes = int.to_bytes(f, length=4, byteorder='big')
    
    time.sleep(0.1)
    a1 = ser.write(crc16(b'PwFA'+f_bytes))
    time.sleep(0.1)
    a2 = ser.write(crc16(b'PwUA'+iU_bytes+b'\x00\x00'))
    time.sleep(0.1)
    a3 = ser.write(crc16(b'PwLA'+iL_bytes+b'\x00\x00'))
    time.sleep(0.1)
    
    a4 = ser.write(crc16(b'MwSA'))
    time.sleep(0.1)
    b4 = ser.read_until()
