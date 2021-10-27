import bluetooth
import json

try:
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", bluetooth.PORT_ANY))
    server_sock.listen(1)
except OSError:
    pass

def clean_data(data):
    data_str = data.decode("utf-8")
    start = len(data_str.split("{")[0])
    end = -len(data_str.split("}")[-1]) if len(data_str.split("}")[-1]) != 0 else len(data_str)
    cln_data_str = data_str[start:end]
    return cln_data_str,data_str[:start],data_str[-end:]

port = server_sock.getsockname()[1]
prefix  = ""
postfix = ""
uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"

bluetooth.advertise_service(server_sock, "SampleServer", service_id=uuid,
                            service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                            profiles=[bluetooth.SERIAL_PORT_PROFILE],
                            # protocols=[bluetooth.OBEX_UUID]
                            )

print("Waiting for connection on RFCOMM channel", port)

client_sock, client_info = server_sock.accept()
print("Accepted connection from", client_info)

acc_f = open("G:/MyBluetoothSensor/acc_file.txt", "w")
gyro_f = open("G:/MyBluetoothSensor/gyro_file.txt", "w")

try:
    while True:
        data = client_sock.recv(1024)
        cleaned_data = clean_data(data)
        json_str   = cleaned_data[0]
        prefix,postfix = cleaned_data[1],cleaned_data[2]
        for entry in json_str.split('#'):
            if len(entry)!=0:

                json_data = json.loads(entry)
                if json_data['Sensor'] == 'Acc':
                    print("Writing to file Acc\n")
                    acc_f.write(json_data["Value"][1:-1]+"\n")
                elif json_data['Sensor'] == 'Gyro':
                    print("Writing to file Gyro\n")
                    gyro_f.write(json_data["Value"][1:-1]+"\n")
except OSError:
    pass

print("Disconnected.")
acc_f.close()
gyro_f.close()
client_sock.close()
server_sock.close()
print("All done.")
