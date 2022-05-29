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


PREDICT_EVERY = 100
GRAPH_UPDATE_DELAY = 30

class Window(QMainWindow):

    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self,*args, **kwargs)

        self.predictor = predictor()
        self.predictor.setup_prediction()
        self.setupUI() 
        self.server = bl_receiver(self)
        self.server.start()
        

        self.graph_timer = QtCore.QTimer()
        self.graph_timer.timeout.connect(self.update_graphs)
        self.graph_timer.start(GRAPH_UPDATE_DELAY)

        self.prediction_timer = QtCore.QTimer()
        self.prediction_timer.timeout.connect(self.make_prediction)
        self.prediction_timer.start(PREDICT_EVERY)


    def setupUI(self):
        self.central_widget = QWidget(self)
        self.setStyleSheet('background-color:#222133')

        self.layout = QGridLayout(self.central_widget)

        self.header = HeaderLabel(self.central_widget)
        self.layout.addWidget(self.header, 0, 0, 1, 2)

        left_layout = QVBoxLayout(self.central_widget)
        self.graphWidget = GraphWidget(self.central_widget, 3, 2)
        left_layout.addWidget(self.graphWidget)

        self.connectionWidget = ConnectionLabel(self.central_widget)
        left_layout.addWidget(self.connectionWidget)

        self.layout.addLayout(left_layout, 1, 0)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 30, 0, 0)
        self.confidenceWidget = ConfidenceChart(self.central_widget, self.predictor.activities)
        self.predictionWidget = PredictionLabel(self.central_widget)
        right_layout.addWidget(self.confidenceWidget)
        right_layout.addWidget(self.predictionWidget)
        self.layout.addLayout(right_layout, 1, 1)

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


    def make_prediction(self):

        data = self.server.get_buffer_data()
        pred = self.predictor.predict(data)

        if pred is not None:
            print("predict")
            activity = self.predictor.activities[pred[1]]
            self.predictionWidget.set_prediction(activity)
            self.confidenceWidget.set_values(pred[0])
    
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