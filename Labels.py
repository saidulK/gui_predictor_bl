from PyQt5.QtWidgets import QApplication, QMainWindow,QLabel,QFrame,QGridLayout,QWidget,QVBoxLayout,QHBoxLayout
from PyQt5 import QtWidgets
import sys
from PyQt5.QtChart import QChart, QChartView, QBarSet, QPercentBarSeries, QBarCategoryAxis,QBarSeries
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt
from PyQt5 import QtCore,QtGui
import numpy as np


class HeaderLabel(QLabel):
    def __init__(self,parent):
        QLabel.__init__(self,parent)
        self.colors = {'bg_color':'#222133','text_color':'#EAE48F'}
        self.setStyleSheet("background-color: {}; border-radius:10px; color: {};padding :15px".format(self.colors['bg_color'],self.colors['text_color']))
        self.setFont(QtGui.QFont('Monserrat', 30))
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText("Human Activity Prediction")

    def sizeHint(self):
        return QtCore.QSize(400,100)



class ConnectionLabel(QLabel):

    def __init__(self,parent):
        QLabel.__init__(self,parent)
        self.setContentsMargins(0,0,0,0)
        self.colors = {'bg_color': '#222133', 'text_color': '#A7A9AC','text_second_color':'#EAE48F'}
        self.setStyleSheet("background-color: {}; border-radius:10px;color: {};padding :15px".format(self.colors['bg_color'],self.colors['text_color']))
        self.setFont(QtGui.QFont('Monserrat', 18))
        self.setAlignment(QtCore.Qt.AlignLeft)
        self.setIPPort("","")

    def setIPPort(self,IP,PORT):
        header = "<font color={}>Connection:</font>".format(self.colors['text_color'])
        if IP=="" or PORT =="" or IP==None or PORT == None:
            text = header+'<br>'+'Not Connected to any Device'
            self.setText(text)
        else:
            server_text ="<font color={}>Server running at IP : </font><font color={}> {} </font>  ".format(self.colors['text_color'],self.colors['text_second_color'],IP)
            port_text   = "<font color={}>PORT:</font><font color={}>{}</font>".format(self.colors['text_color'],self.colors['text_second_color'],PORT)
            text = header+"<br>"+server_text+port_text
            self.setText(text)

class PredictionLabel(QWidget):

    def __init__(self,parent):
        QWidget.__init__(self,parent)
        self.colors = {'bg_color':'#222133',
                       'header_bg_color':'#222133',
                       'header_text_color':'#EAE48F',
                       'prediction_bg_color':'#101019',
                       'prediction_text_color':'#29BBEF'}

        self.setStyleSheet("background-color: {}; border-radius:10px;".format(self.colors['bg_color']))
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        self.layout = QVBoxLayout(self)
        self.header = QLabel(self)
        self.header.setStyleSheet("background-color: {}; border-radius:10px; color: {};padding :15px".format(self.colors['header_bg_color'],self.colors['header_text_color']))
        self.header.setFont(QtGui.QFont('Monserrat', 25,weight=QtGui.QFont.Bold))
        self.header.setAlignment(QtCore.Qt.AlignCenter)
        self.header.setText("PREDICTION:")
        self.layout.addWidget(self.header)

        self.prediction = QLabel(self)
        self.prediction.setStyleSheet("background-color: {}; border-radius:10px; color: {};padding :15px".format(self.colors['prediction_bg_color'],self.colors['prediction_text_color']))
        self.prediction.setFont(QtGui.QFont('Monserrat',30,weight=QtGui.QFont.Bold))
        self.prediction.setAlignment(QtCore.Qt.AlignCenter)
        self.prediction.setText("NO PREDICTION")
        self.layout.addWidget(self.prediction)

    def set_prediction(self,prediction):
        self.prediction.setText(prediction)

    def sizeHint(self):
        return QtCore.QSize(400,200)


if __name__ =='__main__':
    App = QApplication(sys.argv)
    w = QMainWindow()

    h = PredictionLabel(w)

    w.setCentralWidget(h)
    w.show()
    sys.exit(App.exec_())
