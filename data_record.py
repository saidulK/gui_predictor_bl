import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow,QLabel,QGridLayout,QVBoxLayout,QFrame
from pyqtgraph import *

import numpy as np
import time

from bl_funcs import *
from bl_pred import *
from GraphWidget import *
from BarChartWidget import *
from Labels import *

GRAPH_UPDATE_DELAY = 10

class Window(QMainWindow):

    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self,*args, **kwargs)

        
        self.setupUI() 
        self.server = bl_receiver(self)
        self.server.start()
        

        self.graph_timer = QtCore.QTimer()
        self.graph_timer.timeout.connect(self.update_graphs)
        self.graph_timer.start(GRAPH_UPDATE_DELAY)

    def button_clicked(self):
        pass


    def setupUI(self):

        self.central_widget = QWidget(self)
        self.setStyleSheet('background-color:#222133')

        self.layout = QGridLayout(self.central_widget)

        self.header = HeaderLabel(self.central_widget)
        self.layout.addWidget(self.header, 0, 0, 2, 10)

        #left_layout = QVBoxLayout(self.central_widget)
        self.graphWidget = GraphWidget(self.central_widget, 3, 2)
        self.layout.addWidget(self.graphWidget, 2, 0, 8, 10)
        #left_layout.addWidget(self.graphWidget)

        
        self.connectionWidget = ConnectionLabel(self.central_widget)
        self.layout.addWidget(self.connectionWidget, 10, 0, 2, 10)
        #left_layout.addWidget(self.connectionWidget)

        self.yesButton = ButtonWidget(self.central_widget,self.button_clicked)
        
        self.layout.addWidget(self.yesButton, 12, 4, 10, 3)
        #self.yesButton.clicked.connect(lambda: self.prompt_reply(Dialog))
        #self.layout.addLayout(left_layout, 1, 0, 1, 2)

        

        # self.layout.addWidget(self.graphWidget,0,0)

        # self.layout.addWidget(self.confidenceWidget,0,1)

        self.setCentralWidget(self.central_widget)
    
    

    def update_graphs(self):
        data = self.server.get_buffer_data()
        if len(data['Acc']) != 0 and len(data['Gyro']) != 0:
            acc_data = np.array(data['Acc']).T[1:]
            acc_time = np.array(data['Acc']).T[0]
            acc_time = (acc_time - acc_time[0])/1000000000

            gyro_data = np.array(data['Gyro']).T[1:]
            gyro_time = np.array(data['Gyro']).T[0]
            gyro_time = (gyro_time - gyro_time[0])/1000000000

            data = []
            time = []
            for acc, gyro in zip(acc_data, gyro_data):
                data.append([acc, gyro])
                time.append([np.around(acc_time, 2), np.around(gyro_time, 2)])

            self.graphWidget.set_graph(data, time)
    

    def update_connection_status(self,socket=None,client=None):
        
        if client is not None:
            self.connectionWidget.set_connection_status(socket,client)
        elif socket is not None:
            self.connectionWidget.set_connection_status(socket)

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    Gui = Window()
    Gui.show()
    sys.exit(app.exec_())