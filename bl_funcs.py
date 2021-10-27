import bluetooth
import json
from collections import deque
import threading
import time



class bl_receiver(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
        self.lock = threading.Lock()
        self.server_socket = None
        self.client_socket = None
        self.client_info = None
        self.buffer = {"Acc":[],"Gyro":[]}
        self.buffer_limit = 1000
        self.write_to_file = False
        self.file_acc = None
        self.file_gyro = None

    def set_buffer_limit(self,limit):
        self.buffer_limit = limit

    def get_buffer_data(self):

        self.lock.acquire()
        data = self.buffer.copy()
        self.lock.release()

        return data

    def create_file(self,location=""):
        self.file_acc = open(location+"acc.csv","w")
        self.file_gyro = open(location + "gyro.csv", "w")
        return self.file_acc,self.file_gyro

    def create_socket(self):
        try:
            server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            server_sock.bind(("", bluetooth.PORT_ANY))
            server_sock.listen(1)
            self.server_socket = server_sock
            print("Socket was created")
        except OSError:
            print("Bluetooth not enabled")

    def search_connection(self):

        try:
            bluetooth.advertise_service(self.server_socket, "SampleServer", service_id=self.uuid,
                                        service_classes=[self.uuid, bluetooth.SERIAL_PORT_CLASS],
                                        profiles=[bluetooth.SERIAL_PORT_PROFILE],
                                        protocols=[bluetooth.OBEX_UUID]
                                        )
            port = self.server_socket.getsockname()[1]
            print("Waiting for connection on RFCOMM channel", port)
            client_sock, client_info = self.server_socket.accept()
            print("Accepted connection from", client_info)
            self.client_socket,self.client_info=client_sock, client_info

        except Exception as e:
            print("Couldn't connect to device",e)

    def clean_data(self,data, prefix=""):
        data_str = prefix + data.decode("utf-8")
        start = len(data_str.split("{")[0])
        end = -len(data_str.split("}")[-1]) if len(data_str.split("}")[-1]) != 0 else len(data_str)
        cln_data_str = data_str[start:end]
        return cln_data_str, data_str[-end:]


    def process_value(self,value):
        value_list = []
        value_str = value[1:-1]
        for i,number in enumerate(value_str.split(",")):
            if i == 0:
                value_list.append(int(number))
            else:
                value_list.append(float(number))
        return value_list


    def run(self):
        self.create_socket()
        self.search_connection()
        if self.write_to_file:
            acc_file,gyro_file = self.create_file("")

        prefix = ""

        while True:
            if self.client_socket is not None:
                try:
                    data = self.client_socket.recv(1024)
                    cleaned_data, prefix = self.clean_data(data)
                    for entry in cleaned_data.split('#'):
                        if len(entry) != 0:

                            json_data = json.loads(entry)
                            if self.write_to_file:
                                if json_data['Sensor'] == 'Acc':
                                    acc_file.write(json_data['Value'][1:-1]+"\n")
                                else:
                                    gyro_file.write(json_data['Value'][1:-1]+"\n")

                            processed_value = self.process_value(json_data["Value"])


                            self.lock.acquire()
                            if len(self.buffer[json_data["Sensor"]]) >= self.buffer_limit:
                                self.buffer[json_data["Sensor"]].pop(0)

                            self.buffer[json_data["Sensor"]].append(processed_value)
                            self.lock.release()

                except Exception as e:

                    print(e)

if __name__ == '__main__':


    b = bl_receiver()
    b.start()

    start = time.time()

    while True:

        if time.time() - start > 1:

            try:

                data = b.get_buffer_data()
                print((data['Acc'][-1][0]-data['Acc'][0][0])/1000000000)
            except Exception as e:
                print(e)

            start = time.time()





