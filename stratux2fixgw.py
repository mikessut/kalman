import socket
import struct
import fixgw.netfix as netfix
import math
from gdl90 import decodeGDL90

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('', 4000))

netfix_client = netfix.Client('127.0.0.1', 3490)
netfix_client.connect()

while True:
    msg, adr = s.recvfrom(8192)
    msg = decodeGDL90(msg)

    if len(msg) < 1:
        continue

    if msg[0] == 0x4c:
        roll      = struct.unpack('>h', msg[4:6])[0]/10.0
        pitch     = struct.unpack('>h', msg[6:8])[0]/10.0
        heading   = struct.unpack('>h', msg[8:10])[0]/10.0
        slipskid  = struct.unpack('>h', msg[10:12])[0]/10.0
        yawrate   = struct.unpack('>h', msg[12:14])[0]/10.0
        g         = struct.unpack('>h', msg[14:16])[0]/10.0
        ias       = struct.unpack('>h', msg[16:18])[0]/10.0
        alt       = struct.unpack('>h', msg[18:20])[0] - 5000.5
        vs        = struct.unpack('>h', msg[20:22])[0]

        netfix_client.writeValue("PITCH", pitch)
        netfix_client.writeValue("ROLL", roll)
        netfix_client.writeValue("HEAD", heading)

        netfix_client.writeValue("ALAT", -math.sin(slipskid*math.pi/180))
        netfix_client.writeValue("ALT", alt)
        netfix_client.writeValue("VS", vs)
        #print(f"alt: {alt}; vs: {vs}")
        #print(f"slipski: {slipskid}")
    elif msg[0] == 0x0a:
        # ownship report
        alt = struct.unpack('>h', msg[11:13])[0]
        tmp = struct.unpack('BB', msg[14:16])
        gnd_speed = (tmp[0] << 4) | (tmp[1] >> 4)
        netfix_client.writeValue("IAS", gnd_speed)
