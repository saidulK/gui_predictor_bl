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


GRAPH_REFRESH_INTERVAL = 30
PREDICTION_INTERVAL = 500
SERVER_STORAGE_TIME = 5



class dataBuffer:

    def __init__(self,parent,server):
        self.parent = parent
        self.server = server
        self.data = np.array([])
        self.time = np.array([])
        self.server.start()

    def get_graph_data(self):
        data = self.server.return_value()
        if len(data['acc'])!=0 and len(data['gyro'])!=0 :
            acc_data = np.array(data['acc']).T[1:]
            acc_time = np.array(data['acc']).T[0]
            acc_time = (acc_time - acc_time[0])/1000000000

            gyro_data = np.array(data['gyro']).T[1:]
            gyro_time = np.array(data['gyro']).T[0]
            gyro_time = (gyro_time - gyro_time[0])/1000000000

            data = []
            time = []
            for acc,gyro in zip(acc_data,gyro_data):
                data.append([acc,gyro])
                time.append([np.around(acc_time,2),np.around(gyro_time,2)])
            self.data = np.array(data)

            self.time = np.array(time)


            return self.data,self.time

        else:
            #data = np.random.randint(10, size=(ROW * COLUMN, 10))
            return self.data,self.time

    def get_pred_data(self):
        return self.server.return_value()


class Window(QMainWindow):

    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self,*args, **kwargs)


        self.server = bl_receiver()
        self.server.start()
        self.predictor = predictor()
        self.predictor.setup_prediction()
        self.setupUI()

        '''self.dataBuffer = dataBuffer(self,self.server)
        self.setupUI()

        self.connectionWidget.setIPPort(str(self.server.getIP()),str(self.server.getPort()))'''
        self.graph_timer = QtCore.QTimer()
        self.graph_timer.timeout.connect(self.update_graph)
        self.graph_timer.start(GRAPH_REFRESH_INTERVAL)


        self.pred_timer = QtCore.QTimer()
        self.pred_timer.timeout.connect(self.make_prediction)
        self.pred_timer.start(PREDICTION_INTERVAL)

        #self.show()



    def setupUI(self):
        self.central_widget = QWidget(self)
        self.setStyleSheet('background-color:#222133')

        self.layout = QGridLayout(self.central_widget)
        self.header = HeaderLabel(self.central_widget)
        self.layout.addWidget(self.header,0,0,1,2)

        left_layout = QVBoxLayout(self.central_widget)
        self.graphWidget = GraphWidget(self.central_widget,3,2)
        self.connectionWidget = ConnectionLabel(self.central_widget)
        left_layout.addWidget(self.graphWidget)
        left_layout.addWidget(self.connectionWidget)
        self.layout.addLayout(left_layout,1,0)


        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0,30,0,0)
        self.confidenceWidget = ConfidenceChart(self.central_widget,self.predictor.activities)
        self.predictionWidget = PredictionLabel(self.central_widget)
        right_layout.addWidget(self.confidenceWidget)
        right_layout.addWidget(self.predictionWidget)
        self.layout.addLayout(right_layout,1,1)

        #self.layout.addWidget(self.graphWidget,0,0)
        #self.layout.addWidget(self.confidenceWidget,0,1)

        self.setCentralWidget(self.central_widget)


    def update_graph(self):
        data,time = self.dataBuffer.get_graph_data()
        self.graphWidget.set_graph(data,time)

    def make_prediction(self):
        data = self.server.get_buffer_data()
        pred = self.predictor.predict(data)
        if pred is not None:
            confidence,act_no = pred
            activity = self.predictor.activities[act_no]
            self.predictionWidget.set_prediction(activity)
            self.confidenceWidget.set_values(confidence)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    Gui = Window()
    Gui.show()
    sys.exit(app.exec_())