from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets
from pyqtgraph import *

graph_titles = [["Accelerometer","Gyroscope"],["AccelerometerY","GyroscopeY"],["AccelerometerZ","GyroscopeZ"]]
graph_left   = [["X Axis"],["Y Axis"],["Z Axis"]]
graph_Yrange = [ [[-20,20],[-20,20]]
                ,[[-20,20],[-20,20]]
                ,[[-20,20],[-20,20]]]

graph_Xrange = [ [[],[]]
                ,[[],[]]
                ,[[],[]]]

class GraphWidget(GraphicsLayoutWidget):

    def __init__(self,parent,row=3,col=2,border=None):
        GraphicsLayoutWidget.__init__(self,parent)
        self.parent = parent
        self.row = row
        self.col = col
        self.colors = {'bg_color':'#222133',
                       'graph_color':'#29BBEF',
                       'graph_bg_color':'#3B3B5E',
                       'graph_title_color':'#A7A9AC',
                       'bottom_axis_color':'#3B3B5E',
                       'left_axis_color':'#EAE48F',
                       'bottom_text_color':'#EAE48F',
                       'left_text_color':'#EAE48F',
                       'bottom_label_color':'#A7A9AC',
                       'left_label_color':'#A7A9AC'
                       }
        self.setupUI()
        #self.timer = QtCore.QTimer()
        #self.timer.timeout.connect(self.update_graph)
        #self.timer.start(30)

    def setupUI(self):
        self.setBackground(self.colors['bg_color'])
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        #self.ci.layout.setSpacing(0)
        self.ci.layout.setContentsMargins(20,0,10,20)

        self.init_graphs()

    def init_graphs(self):
        for i in range(self.row):
            for j in range(self.col):
                plotItem = self.addPlot(row=i, col=j, name=graph_titles[i][j], labels=None, title=None, viewBox=None)

                #Change Range Settings
                plotItem.enableAutoRange()

                pen = mkPen(self.colors['graph_color'])
                plotItem.plot([], clear=True,pen=pen)

                vb = plotItem.getViewBox()
                vb.setBackgroundColor(self.colors['graph_bg_color'])

                plotItem.getAxis('bottom').setPen(self.colors['bottom_axis_color'])
                plotItem.getAxis('bottom').setTextPen(self.colors['bottom_text_color'])
                plotItem.getAxis('left').setPen(self.colors['left_axis_color'])
                plotItem.getAxis('left').setTextPen(self.colors['left_text_color'])


                if (i == 0):
                    plotItem.setTitle(graph_titles[i][j],**{'color': self.colors['graph_title_color'],'size': '12pt','bold': True})
                elif i == self.row-1:
                    plotItem.setLabel('bottom', 'Time(s)', **{'color': self.colors['bottom_label_color'],'font-size': '11pt','bold': True})

                if j==0:
                    plotItem.setLabel('left', graph_left[i][j], **{'color': self.colors['left_label_color'], 'font-size': '11pt','bold': True})


                if len(graph_Yrange[i][j])!=0:
                    plotItem.setRange(yRange=graph_Yrange[i][j])

                if len(graph_Xrange[i][j])!=0:
                    plotItem.setRange(xRange=graph_Xrange[i][j])


    def set_graph(self,data,time):
        #data = self.parent.dataBuffer.get_data()

        if len(data)!=0:

            for i in range(self.row):
                for j in range(self.col):
                    total_ticks = 5
                    inc = int(len(time[i][j])/total_ticks)
                    if inc == 0:
                        inc = 1
                    ticks = [[(t, str(time[i][j][t])) for t in range(0,len(time[i][j]),inc)]]

                    self.getItem(i, j).getAxis('bottom').setTicks(ticks)
                    self.getItem(i, j).listDataItems()[0].setData(data[i][j])
                    #self.getItem(i,j).listDataItems()[0].setData(data[i*(self.row -1)+j])
                    #self.getItem(i,j).plot(data[i*(self.row -1)+j],clear=True)

    def sizeHint(self):
        return QtCore.QSize(600,400)





if __name__ =='__main__':
    App = QApplication(sys.argv)

    #window = Window()
    w = QMainWindow()

    graph = GraphWidget(w,3,2)
    #graph = newPlot(w)
    w.setCentralWidget(graph)


    w.show()
    sys.exit(App.exec_())
