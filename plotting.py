import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import pyqtgraph as pg
import queue
import threading
import time
import itertools


class Plotter(pyqtgraph.GraphicsLayoutWidget):

    data_acquired = pyqtSignal(np.ndarray)

    def __init__(self, width, q, names):
        super().__init__()
        ### START QtApp #####
        #self.app = QtGui.QApplication([])            # you MUST do this once (initialize things)
        ####################

        win = pg.GraphicsWindow(title="Kalman Filter Scope") # creates a window
        self.win = win
        self.nplots = 1
        self.ncurves = len(names)
        self.curves2plots = [0, 0, 0, 0] # , 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        #names = ['true turn rate', 'kf turn rate', 'q0 P', 'q1 P', 'q2 P', 'q3 P']

        self.Xm = []
        self.curve = []
        windowWidth = width                       # width of the window displaying the curve
        colors = itertools.cycle([{'color': x} for x in 'rgbcmykw'])
        plots = []
        for _ in range(self.nplots):
            plots.append(win.addPlot())  #  Creates PlotItem
            plots[-1].addLegend()

        for nc in range(self.ncurves):            
            self.curve.append(plots[self.curves2plots[nc]].plot(pen=next(colors), name=names[nc]))
            self.Xm.append(np.linspace(0, 0, windowWidth))

        self.ptr = -windowWidth                      # set first x position
        self.q = q
        self.data_acquired.connect(self.update)
        self.start_thread_queue()
        #self.app.exec_()

    @pyqtSlot(np.ndarray)
    def update(self, value):
        for n in range(self.ncurves):
            self.Xm[n][:-1] = self.Xm[n][1:]                      # shift data in the temporal mean 1 sample left
            
            self.Xm[n][-1] = float(value[n])                 # vector containing the instantaneous values      
            self.ptr += 1                              # update x position for displaying the curve
            self.curve[n].setData(self.Xm[n])                     # set the curve with this data
            self.curve[n].setPos(self.ptr,0)                   # set x position in the graph to 0

    def start_thread_queue(self):
        
        def thread_func():
            while True:
                val = self.q.get()
                #self.update(val)
                self.data_acquired.emit(val)

        thread = threading.Thread(target=thread_func)
        self.thread = thread
        thread.start()


# p = Plotter(100)
# ### MAIN PROGRAM #####    
# # this is a brutal infinite loop calling your realtime data plot
# while True: 
#     p.update(np.random.randn(1))
# 
# ### END QtApp ####
# pg.QtGui.QApplication.exec_() # you MUST put this at the end
# ##################