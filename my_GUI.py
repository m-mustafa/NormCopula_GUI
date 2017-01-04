# -*- coding: utf-8 -*-
"""
Created on %(30-June-2016)s

@author: %(Faizan Anwar)s
"""
#from tkinter import *
import os
import sys
import timeit
import time
import pandas as pd
import numpy as np
import xlrd
import matplotlib.pylab as plt
from matplotlib import cm
from PyQt4.QtCore import *
from PyQt4.QtGui import *
plt.ioff()
import matplotlib.gridspec as gridspec
import matplotlib
import FileDialog # needed by pyinstaller
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
space = " "

class OutLog:
    '''
    A class to print console messages to the messages window in the GUI
    '''
    def __init__(self, edit, out=None):
        """(edit, out=None) -> can write stdout, stderr to a
        QTextEdit.
        edit = QTextEdit
        out = alternate stream ( can be the original sys.stdout )
        """
        self.edit = edit
        self.out = out

    def write(self, m):
        self.edit.insertPlainText(m)


class MainWindow(QTabWidget):
    '''
    Control panel of the plotting GUI
    '''
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.conf_input_tab = QWidget()
        self.render_tab = QWidget()


        self.addTab(self.conf_input_tab, 'Configure Input')
        self.addTab(self.render_tab, 'Render')
        self.conf_input_tab_UI()
        self.render_tab_UI()
#        self.render_tab.setDisabled(True)
        self.read_succ = True

        print '#' * 50
        print 'Ready....'
        print '#' * 50, '\n'

#
    def conf_input_tab_UI(self):
        '''
        The Configure Input tab
        '''
        self.main_layout = QVBoxLayout()
        self.grid = QGridLayout()
        
        


        i = 0

        self.main_dir_lab = QLabel('Main Directory:')
        self.main_dir_line = QLineEdit('')
        self.main_dir_btn = QPushButton('Browse')
        self.main_dir_btn.clicked.connect(lambda: self.get_main_dir(_input=True))
        self.grid.addWidget(self.main_dir_lab, i, 0)
        self.grid.addWidget(self.main_dir_line, i, 1, 1, 20)
        self.grid.addWidget(self.main_dir_btn, i, 21)
        self.main_dir_lab.setToolTip(str('Full path to the main directory.'))
        self.main_dir_line.setToolTip(str('Full path to the main directory.'))
        self.main_dir_btn.setToolTip(str('Browse to the main directory.'))
        i += 1
# 
        self.in_q_orig_file_lab = QLabel('Input File:')
        self.in_q_orig_file_line = QLineEdit('')
        self.in_q_orig_file_btn = QPushButton('Browse')
        self.in_q_orig_file_btn.clicked.connect(lambda: self.get_input_dir(_input=True))
        self.grid.addWidget(self.in_q_orig_file_lab, i, 0)
        self.grid.addWidget(self.in_q_orig_file_line, i, 1, 1, 20)
        self.grid.addWidget(self.in_q_orig_file_btn, i, 21)
        self.in_q_orig_file_lab.setToolTip(str('Full path to the input csv file.'))
        self.in_q_orig_file_line.setToolTip(str('Full path to the input csv file.'))
        self.in_q_orig_file_btn.setToolTip(str('Browse to the input csv file.'))
        i += 1
        
        self.in_coords_file_lab = QLabel('Coordinates File:')
        self.in_coords_file_line = QLineEdit('')
        self.in_coords_file_btn = QPushButton('Browse')
        self.in_coords_file_btn.clicked.connect(lambda: self.get_coords_dir(_input=True))
        self.grid.addWidget(self.in_coords_file_lab, i, 0)
        self.grid.addWidget(self.in_coords_file_line, i, 1, 1, 20)
        self.grid.addWidget(self.in_coords_file_btn, i, 21)
        self.in_coords_file_lab.setToolTip(str('Full path to the input csv file.'))
        self.in_coords_file_line.setToolTip(str('Full path to the input csv file.'))
        self.in_coords_file_btn.setToolTip(str('Browse to the input csv file.'))
        i += 1

        self.out_dir_lab = QLabel('Output Directory:')
        self.out_dir_line = QLineEdit('')
        self.out_dir_btn = QPushButton('Browse')
        self.out_dir_btn.clicked.connect(lambda: self.get_out_dir(_input=True))
        self.grid.addWidget(self.out_dir_lab, i, 0)
        self.grid.addWidget(self.out_dir_line, i, 1, 1, 20)
        self.grid.addWidget(self.out_dir_btn, i, 21)
        self.out_dir_lab.setToolTip(str('Directory and figure name with extension e.g. png.'))
        self.out_dir_line.setToolTip(str('Directory and figure name with extension e.g. png.'))
        self.out_dir_btn.setToolTip(str('Browse to output directory'))
        i += 1

        self.date_fmt_lab = QLabel('Date Format:')
        self.date_fmt_box = QComboBox()
        self.date_fmt_box.addItems(['','%Y-%m-%d','%d-%m-%Y','%m-%d-%Y'])
#        self.in_sheet_box.currentIndexChanged.connect(self.select_sheet)
        self.grid.addWidget(self.date_fmt_lab, i, 0)
        self.grid.addWidget(self.date_fmt_box, i, 1, 1, 20)
        self.date_fmt_lab.setToolTip(str('Date Format.'))
        self.date_fmt_box.setToolTip(str('Date Format.'))
        i += 1

        self.infill_stns_lab = QLabel('Infill Stations:')
        self.infill_stns_line = QLineEdit(str(''))
#        self.infill_stns_btn = QPushButton('Add')
#        self.infill_stns_list = []
#        self.infill_stns_btn.clicked.connect(lambda: self.infill_add_btn())
        self.grid.addWidget(self.infill_stns_lab, i, 0)
        self.grid.addWidget(self.infill_stns_line, i, 1, 1, 20)
#        self.grid.addWidget(self.infill_stns_btn, i, 21)
        self.infill_stns_lab.setToolTip(str('list of infill stations'))
        self.infill_stns_line.setToolTip(str('list of infill stations.'))  
        i += 1
                
        self.drop_stns_lab = QLabel('Drop Stations:')
        self.drop_stns_line = QLineEdit('')
#        self.drop_stns_btn = QPushButton('Add')
#        self.infill_stns_list = []
#        self.drop_stns_btn.clicked.connect(lambda: self.drop_add_btn())
        self.grid.addWidget(self.drop_stns_lab, i, 0)
        self.grid.addWidget(self.drop_stns_line, i, 1, 1, 20)
#        self.grid.addWidget(self.drop_stns_btn, i, 21)
        self.drop_stns_lab.setToolTip(str('list of stations to drop.'))
        self.drop_stns_line.setToolTip(str('list of stations to drop.'))  
        i += 1
        
        self.censor_period_lab = QLabel('Censor Period:')
        self.censor_period_line = QLineEdit('')
#        self.censor_period_btn = QPushButton('Add')
#        self.censor_period_btn.clicked.connect(lambda: self.censor_add_btn())
        self.grid.addWidget(self.censor_period_lab, i, 0)
        self.grid.addWidget(self.censor_period_line, i, 1, 1, 20)
#        self.grid.addWidget(self.censor_period_btn, i, 21)
#        self.drop_stns_lab.setToolTip(str('list of input stations'))
#        self.drop_stns_line.setToolTip(str('list of input stations.'))  
        i += 1

        self.infill_interval_type_lab = QLabel('Infill interval type:')
        self.infill_interval_type_box = QComboBox()
        self.infill_interval_type_box.addItems(['slice','individual','all'])
#        self.in_sheet_box.currentIndexChanged.connect(self.select_sheet)
        self.grid.addWidget(self.infill_interval_type_lab, i, 0)
        self.grid.addWidget(self.infill_interval_type_box, i, 1, 1, 20)
        self.infill_interval_type_lab.setToolTip(str('Infill Interval Type.'))
        self.infill_interval_type_box.setToolTip(str('Infill Interval Type.'))
        i += 1
        
        self.infill_type_lab = QLabel('Infill type:')
        self.infill_type_box = QComboBox()
        self.infill_type_box.addItems(['precipitation','discharge'])
#        self.in_sheet_box.currentIndexChanged.connect(self.select_sheet)
        self.grid.addWidget(self.infill_type_lab, i, 0)
        self.grid.addWidget(self.infill_type_box, i, 1, 1, 20)
        self.infill_type_lab.setToolTip(str('Infill Type.'))
        self.infill_type_box.setToolTip(str('Infill Type.'))
        i += 1
        
        self.n_nrn_min_lab = QLabel('Minimum Nearest stations:')
        self.n_nrn_min_line = QLineEdit('')
        self.grid.addWidget(self.n_nrn_min_lab, i, 0)
        self.grid.addWidget(self.n_nrn_min_line, i, 1, 1, 20)
#        self.n_nrn_min_lab.setToolTip(str('list of input stations'))
#        self.n_nrn_min_line.setToolTip(str('list of input stations.'))  
        i += 1     
        
        self.n_nrn_max_lab = QLabel('Maximum Nearest Stations:')
        self.n_nrn_max_line = QLineEdit('')
        self.grid.addWidget(self.n_nrn_max_lab, i, 0)
        self.grid.addWidget(self.n_nrn_max_line, i, 1, 1, 20)
#        self.n_nrn_max_lab.setToolTip(str('list of input stations'))
#        self.n_nrn_max_line.setToolTip(str('list of input stations.'))  
        i += 1
        
        self.ncpus_lab = QLabel('Number of CPUs:')
        self.ncpus_box = QComboBox()
        self.ncpus_box.addItems(['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
#        self.in_sheet_box.currentIndexChanged.connect(self.select_sheet)
        self.grid.addWidget(self.ncpus_lab, i, 0)
        self.grid.addWidget(self.ncpus_box, i, 1, 1, 20)
        self.ncpus_lab.setToolTip(str('Infill Type.'))
        self.ncpus_box.setToolTip(str('Infill Type.'))
        i += 1
        
        self.sep_lab = QLabel('Seperator:')
        self.sep_box = QComboBox()
        self.sep_box.addItems([';'])
#        self.in_sheet_box.currentIndexChanged.connect(self.select_sheet)
        self.grid.addWidget(self.sep_lab, i, 0)
        self.grid.addWidget(self.sep_box, i, 1, 1, 20)
        self.sep_lab.setToolTip(str('seperator.'))
        self.sep_box.setToolTip(str('seperator.'))
        i += 1        


        self.freq_lab = QLabel('Freq:')
        self.freq_box = QComboBox()
        self.freq_box.addItems(['D'])
#        self.in_sheet_box.currentIndexChanged.connect(self.select_sheet)
        self.grid.addWidget(self.freq_lab, i, 0)
        self.grid.addWidget(self.freq_box, i, 1, 1, 20)
        self.ncpus_lab.setToolTip(str('freq.'))
        self.ncpus_box.setToolTip(str('freq.'))
        i += 1

        self.read_data_lab = 'Read data!'
        self.read_data_btn = QPushButton()
        self.read_data_btn.setText(self.read_data_lab)
        self.read_data_btn.clicked.connect(lambda: self.read_data())
        self.grid.addWidget(self.read_data_btn, i, 1, 1, 20)
        self.read_data_btn.setToolTip(str('Read all the data under the columns specified above!'))
        i += 1
        
        self.CheckBox = QCheckBox('Check', self)
        self.CheckBox.stateChanged.connect(self.check)
        self.CheckBox.setText("yea baby")
        self.grid.addWidget(self.CheckBox, i, 1, 1, 20)
        
        
        self.main_layout.addLayout(self.grid)
        self.conf_input_tab.setLayout(self.main_layout) 

    def render_tab_UI(self):
        '''
        The Render tab
        '''
        self.render_layout = QVBoxLayout()
        self.render_grid = QGridLayout()
        
        self.render_layout.addLayout(self.render_grid)
        self.render_tab.setLayout(self.render_layout)        
    def check(self, state):
        if self.CheckBox.isChecked() == True:
            print "it works!!"
        else:
            print "sorry bro"
            
    def get_main_dir(self, _input=True):
        if _input:
#            dlg = QFileDialog.getExistingDirectory(self)
#            dlg = QFileDialog()
#            dlg.setFileMode(QFileDialog.FileMode())
#            if dlg.exec_():
            self.main_dir = str(QFileDialog.getExistingDirectory(self))
            self.main_dir_line.setText(self.main_dir)

                
    def get_input_dir(self, _input=True):
        if _input:

            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.FileMode())
            if dlg.exec_():
                self.in_q_orig_file = dlg.selectedFiles()[0]
                self.in_q_orig_file_line.setText(self.in_q_orig_file)
                
    def get_out_dir(self, _input=True):
        if _input:

#            dlg = QFileDialog()
#            dlg.setFileMode(QFileDialog.FileMode())
#            if dlg.exec_():
            self.out_dir = str(QFileDialog.getExistingDirectory(self))
            self.out_dir_line.setText(self.out_dir)
                
                
    def get_coords_dir(self, _input=True):
        if _input:

            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.FileMode())
            if dlg.exec_():
                self.in_coords_file = dlg.selectedFiles()[0]
                self.in_coords_file_line.setText(self.in_coords_file)
                
#    def infill_add_btn(self):
#        self.infill_stns = str(self.infill_stns_line.text())
##        self.infill_stns_list.append(self.infill_stns_line.text())
##        self.infill_stns_line.clear()
#        self.infill_stns_list = self.infill_stns.split()
#        print "Infill Stations Added Successfully.... Current Infill Stations:", self.infill_stns_list
        
#    def drop_add_btn(self):
#        self.drop_stns = str(self.drop_stns_line.text())
##        self.infill_stns_list.append(self.infill_stns_line.text())
##        self.infill_stns_line.clear()
#        self.drop_stns_list = self.drop_stns.split()
#        print "Drop Stations Added Successfully.... Current Drop Stations:", self.drop_stns_list
#        
#    def censor_add_btn(self):
#        self.censor_period = str(self.censor_period_line.text())
##        self.infill_stns_list.append(self.infill_stns_line.text())
##        self.infill_stns_line.clear()
#        self.censor_period_list = self.censor_period.split()
#        print "censor period Added Successfully.... Current period:", self.censor_period_list
        

    def read_data(self):
        try:
            assert self.main_dir
            print u'\u2714', "Main Directory is:", self.main_dir
        except Exception as msg:
            self.show_error('Please Select a valid Main Directory', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Main directory"
        try:
            assert self.in_q_orig_file
            print u'\u2714', "Input file is:", self.in_q_orig_file
        except Exception as msg:
            self.show_error('Please Select a valid Input file', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading input file"
        try:
            assert self.in_coords_file
            print u'\u2714', "Coordinates file is:", self.in_coords_file
        except Exception as msg:
            self.show_error('Please Select a valid Coordinates file', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Main Coordinates file"
        try:
            assert self.out_dir
            print u'\u2714', "Output directory is:", self.out_dir
        except Exception as msg:
            self.show_error('Please Select a valid Output Directory', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Output directory"
        try:
            self.date_fmt = self.date_fmt_box.currentText()
            assert self.date_fmt
            print u'\u2714', "Date Format is:", self.date_fmt
        except Exception as msg:
            self.show_error('Please Select a valid Date format', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Date format"
        try:
            self.infill_stns = str(self.infill_stns_line.text())
            self.infill_stns_list = self.infill_stns.split()
            assert self.infill_stns_list
            print u'\u2714', "Infill Stations:", self.infill_stns_list
        except Exception as msg:
            self.show_error('Please Specify infill stations (values must be separated by a space)', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading infill stations"
        try:
            self.drop_stns = str(self.drop_stns_line.text())
            self.drop_stns_list = self.drop_stns.split()
            assert self.drop_stns_list
            print u'\u2714', "Stations to drop:", self.drop_stns_list
        except:
            print u'\u2714', "No stations to drop"
        try:
            self.censor_period = str(self.censor_period_line.text())
            self.censor_period_list = self.censor_period.split()
            assert self.censor_period_list
            print u'\u2714', "Censor period:", self.censor_period_list
        except Exception as msg:
            self.show_error('Please Specify a valid Censor Period, (values must be separated by a space)', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Censor Period"
        try:
            self.infill_interval_type = self.infill_interval_type_box.currentText()
            assert self.infill_interval_type
            print u'\u2714', "Infill Interval Type is:", self.infill_interval_type
        except Exception as msg:
            self.show_error('Please Select a valid infill interval type', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Interval type"
        try:
            self.infill_type = self.infill_type_box.currentText()
            assert self.infill_type
            print u'\u2714', "Infill Type is:", self.infill_type
        except Exception as msg:
            self.show_error('Please Select a valid infill type', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Infill Type"
        try:
            self.n_nrn_min = str(self.n_nrn_min_line.text())
            assert self.n_nrn_min
            print u'\u2714', "Minimum nearest stations:", self.n_nrn_min
        except Exception as msg:
            self.show_error('Please specify a valid value in Minimum nearest stations box', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Minimum nearest station"
        try: 
            self.n_nrn_max = str(self.n_nrn_max_line.text())
            assert self.n_nrn_max
            print u'\u2714', "Maximum nearest stations:", self.n_nrn_max
        except Exception as msg:
            self.show_error('Please specify a valid value in Maximum nearest stations box', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Maximum nearest stations"
        try:
            self.ncpus = self.ncpus_box.currentText()
            assert self.ncpus
            print u'\u2714', "Number of CPUs:", self.ncpus
        except Exception as msg:
            self.show_error('Please Select a valid Number of CPUs', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Number of CPUs"
        try:
            self.sep = self.sep_box.currentText()
            assert self.sep
            print u'\u2714', "Separator is:", self.sep
        except Exception as msg:
            self.show_error('Please Select a valid Separator', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Separator"
        try:
            self.freq = self.freq_box.currentText()
            assert self.freq
            print u'\u2714', "Freq is:", self.freq
        except Exception as msg:
            self.show_error('Please Select a valid Freq', QMessageBox.Critical, details=repr(msg))
            print u'\u2716', "Error reading Freq"
        print '\n'    
        print '#' * 50
        print 'Finished Reading Data!'
        print '#' * 50, '\n'    
             
            
            
            
            
            
            
    def show_error(self, msg, icon, details='Whoops, No details.'):
        '''
        Show an error message box
        '''
        widget = QWidget()
        err_box = QMessageBox(widget)
        err_box.setIcon(icon)
        err_box.setText(msg)
        err_box.setWindowTitle('Error')
        err_box.setDetailedText('The details are as follow:\n' + details)
        err_box.setStandardButtons(QMessageBox.Ok)
        err_box.exec_()
class NormaCopula(QMainWindow):
    '''
    The whole window of the GUI
    This holds the tabs and all the other stuff
    '''
    def __init__(self, parent=None):
        super(NormaCopula, self).__init__(parent)

        self.window_area = QMdiArea()
        self.setCentralWidget(self.window_area)
        self.setWindowTitle('NormaCopula Plotter v0.1')

        self.msgs_wid = QTextEdit()
        self.msgs_wid.setReadOnly(True)
        sys.stdout = OutLog(self.msgs_wid, sys.stdout)
        sys.stderr = OutLog(self.msgs_wid, sys.stderr)

        self.msgs_window = QMdiSubWindow()
        self.msgs_wid.setWindowTitle('Messages')
        self.msgs_window.setWidget(self.msgs_wid)
        self.window_area.addSubWindow(self.msgs_window)
        self.msgs_window.setWindowFlags(Qt.FramelessWindowHint)
        self.msgs_window.setWindowFlags(Qt.WindowTitleHint)

        self.input_window = QMdiSubWindow()
        self.input_window.setWindowTitle('Control Panel')
        self.input_window.setWidget(MainWindow())
        self.window_area.addSubWindow(self.input_window)
        self.input_window.setWindowFlags(Qt.FramelessWindowHint)
        self.input_window.setWindowFlags(Qt.WindowTitleHint)
        self.showMaximized()
        self.input_window.show()
        self.window_area.tileSubWindows()


def main():
    app = QApplication(sys.argv)
    ex = NormaCopula()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    print '\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime()
    start = timeit.default_timer() # to get the runtime of the program

    main()

    stop = timeit.default_timer()  # Ending time
    print '\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' % (time.asctime(), stop-start)
