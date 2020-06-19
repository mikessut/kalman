import socket
import struct
import fixgw.netfix as netfix
import math
from gdl90 import decodeGDL90


if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('', 4000))

    netfix_client = netfix.Client('127.0.0.1', 3490)
    netfix_client.connect()

    while True:
        msg, adr = s.recvfrom(8192)
        #print(adr)
        #print(msg)
        #print(f"0x{struct.unpack('B', msg[1:2])[0]:02X}")
        #msg = struct.unpack('B'*len(msg), msg)

        msg_cleaned = decodeGDL90(msg)

        if len(msg_cleaned) < 1:
            continue

        if msg_cleaned[0] == 0x4c:
            #msg = msg[1:]
            roll = 0
            gx, gy, gz      = struct.unpack('<fff', msg_cleaned[4:4+4*3])
            ax, ay, az      = struct.unpack('<fff', msg_cleaned[16:16+4*3])
            mx, my, mz      = struct.unpack('<fff', msg_cleaned[28:28+4*3])
            fstr = '{:7.3f}'*3 + '{:7.1f}'*6
            print(fstr.format(gx,gy,gz,ax,ay,az,mx,my,mz))
        elif msg[0] == 0x0a:
            # ownship report
            #msg = msg[1:]
            alt = struct.unpack('>h', msg[11:13])[0]
            tmp = struct.unpack('BB', msg[14:16])
            gnd_speed = (tmp[0] << 4) | (tmp[1] >> 4)
            #print(f"gnd_speed: {gnd_speed}")
            netfix_client.writeValue("IAS", gnd_speed)
