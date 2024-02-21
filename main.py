# -*- coding: utf-8 -*-
import sys
import numpy as np
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTabWidget, QComboBox
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.optimize import curve_fit


def gaussian_u(x,A,sig,mu):
    return A*(2*np.pi*sig**2)**(-0.5)*np.exp(-(x-mu)**2/(2*sig**2))

class MyApplication(QWidget):
    def __init__(self):
        super(MyApplication, self).__init__()
        
        self.flag = 2
        
        #create a horizontal layout - holding ((input_layout, tabs_layout),(en_layout,entabs_layout))
        main_layout = QHBoxLayout()
        
        # Create two vertical layouts
        self.input_layout = QVBoxLayout()
        self.tabs_layout = QVBoxLayout()

        # Set up input_layout
        self.setup_input_layout()
       # Set up tabs_layout
        self.setup_tabs_layout()
     
        # Combine both (input_layout, tabs_layout) in the fmain_layout
        fmain_layout = QVBoxLayout()
        fmain_layout.addLayout(self.input_layout)
        fmain_layout.addLayout(self.tabs_layout)

        
        # Create two more vertical layouts
        self.en_layout = QVBoxLayout()
        self.entabs_layout = QVBoxLayout()
        self.en2_layout = QVBoxLayout()

        # Set up en_layout
        self.setup_en_layout()
       # Set up entabs_layout
        self.setup_entabs_layout()
        #set up en2_layout
        self.setup_en2_layout()
     
        # Combine both (en_layout, entabs_layout) in the smain_layout
        smain_layout = QVBoxLayout()
        smain_layout.addLayout(self.en_layout)
        smain_layout.addLayout(self.entabs_layout)
        smain_layout.addLayout(self.en2_layout)



        main_layout.addLayout(fmain_layout)
        main_layout.addLayout(smain_layout)
        self.setLayout(main_layout)
        self.setMinimumSize(1500, 1000)

        
    def setup_tabs_layout(self):
            self.tabs = QTabWidget()
            self.tabs_layout.addWidget(self.tabs)
            
    def setup_entabs_layout(self):
            self.entabs = QTabWidget()
            self.entabs_layout.addWidget(self.entabs)

    def setup_input_layout(self):
        
        qvalues_label = QLabel("Qvalues:")
        self.qvalues_entry = QLineEdit()

        fno_label_data = QLabel("File # (Data):")
        self.fno_entry_data = QLineEdit()
        
        fno_label_bkg = QLabel("File # (Bkg):")
        self.fno_entry_bkg = QLineEdit()

        indices_label_data = QLabel("Indices (data):")
        self.indices_entry_data = QLineEdit()
        
        indices_label_bkg = QLabel("Indices (Bkg):")
        self.indices_entry_bkg = QLineEdit()
     
        path_label = QLabel("Path:")
        self.path_entry = QLineEdit()


        extract_button = QPushButton("Extract Data")
        extract_button.clicked.connect(self.extract_data)
        

        # Set default values for the QLineEdit widgets
        self.qvalues_entry.setText("0.8, 1.08, 1.32")
        
        self.fno_entry_data.setText("1, 1, 1")
        self.fno_entry_bkg.setText("1, 1, 1")
        
        self.indices_entry_data.setText("650, 646, 647")
        self.indices_entry_bkg.setText("643, 598, 641")
        
        self.path_entry.setText("/mnt/6585437601FD8F0E/Office/QENS/MARX_analysis/ACM_DES_water")


        # Add widgets to the input_layout
        qhbox = QHBoxLayout()
        qhbox.addWidget(qvalues_label)
        qhbox.addWidget(self.qvalues_entry)
        self.input_layout.addLayout(qhbox)
        
        fno_layout = QHBoxLayout()
        fno_layout.addWidget(fno_label_data)
        fno_layout.addWidget(self.fno_entry_data)
        fno_layout.addWidget(fno_label_bkg)
        fno_layout.addWidget(self.fno_entry_bkg)
        self.input_layout.addLayout(fno_layout)
        
        indices_layout = QHBoxLayout()
        indices_layout.addWidget(indices_label_data)
        indices_layout.addWidget(self.indices_entry_data)
        indices_layout.addWidget(indices_label_bkg)
        indices_layout.addWidget(self.indices_entry_bkg)
        self.input_layout.addLayout(indices_layout)
        
        phbox = QHBoxLayout()
        phbox.addWidget(path_label)
        phbox.addWidget(self.path_entry)
        self.input_layout.addLayout(phbox)
        
        
        
        self.input_layout.addWidget(extract_button)

    
    def setup_en_layout(self):
        cf_label = QLabel("Conversion factor:")
        self.cf_entry = QLineEdit()


        d_label = QLabel("2d (Angs)")
        self.d_entry = QLineEdit()
        
        Ei_label = QLabel("Incident energy (Ei)")
        self.Ei_entry = QLineEdit()
        

        conv_energy = QPushButton("Convert to energy")
#        extract_button.clicked.connect(self.extract_data)
        
        # Set default values for the QLineEdit widgets
        self.cf_entry.setText("0.04202")
        self.d_entry.setText("6.67")
        self.Ei_entry.setText("5.11")
        
        # Add widgets to the en_layout
        cfhbox = QHBoxLayout()
        cfhbox.addWidget(cf_label)
        cfhbox.addWidget(self.cf_entry)
        self.en_layout.addLayout(cfhbox)
        
        dhbox = QHBoxLayout()
        dhbox.addWidget(d_label)
        dhbox.addWidget(self.d_entry)
        self.en_layout.addLayout(dhbox)
        
        Ehbox = QHBoxLayout()
        Ehbox.addWidget(Ei_label)
        Ehbox.addWidget(self.Ei_entry)
        self.en_layout.addLayout(Ehbox)
        
        self.en_layout.addWidget(conv_energy)
        conv_energy.clicked.connect(self.to_energy)
        
    def setup_en2_layout(self):
        
        self.sfigure = Figure(figsize=(5,4),dpi=100)
        self.scanvas = FigureCanvas(self.sfigure)
        
        self.en2_layout.addWidget(NavigationToolbar(self.scanvas, self))
        
        self.en2_layout.addWidget(self.scanvas)
        
        
        self.save_button = QPushButton("Save energy data")
        self.save_label = QLabel("Filename (including path)")
        self.save_fname = QLineEdit(self)
        
        self.en2_layout.addWidget(self.save_label)
        self.en2_layout.addWidget(self.save_fname)
        self.en2_layout.addWidget(self.save_button)
        
    def fit_plot(self,en,sqe):
        
        widths = np.zeros((self.qvalues.shape[0]))
        centre = np.zeros((self.qvalues.shape[0]))
        for i in range(0,len(self.qvalues)):
            
            #just fitting all the curves with a gaussian_u to see how it fares
            popt, pcov = curve_fit(gaussian_u,en,sqe[i,:])
            
            centre[i] = popt[2]
            widths[i] = abs(popt[1])
        
        sax1 = self.sfigure.add_subplot(211)
        sax2 = self.sfigure.add_subplot(212)
        
        sax1.plot(self.qvalues, widths, "o", fillstyle='none',label="Widths")
        sax2.plot(self.qvalues, centre, "s", fillstyle='none',label="Centres")
        sax2.axhline(0.0,color='black')
        
        sax2.set_ylim(-0.01,0.01)
        sax1.set_ylim(0.,0.2)
        
        sax1.xaxis.set_visible(False)
        
        
        sax1.set_ylabel("Widths")
        sax2.set_ylabel("Centres")
        sax1.set_xlabel("$Q \\AA^{-1}$")
        
#        sleg = sax.legend()
#        sleg.draggable(True)
        self.sfigure.subplots_adjust(hspace=0)
        # print ("Type of energy array: ", type(self.energy), self.energy)
        # print ("Type of sqe array: ", type(self.sqw), self.sqw)
        
        self.save_button.clicked.connect(self.write_to_file)


    
    def read_data(self,Q, nfiles, index, path):
        # Generate filenames
        prefix = 'QENS0'
        suffix = '0101.txt'
        fname = []
    
        for k in range(0, index.shape[0]):
            folder_name = path +"/"+ prefix + str(index[k]) + '/'
            fname.append(folder_name + prefix + str(index[k]) + suffix)
    
        # Initialize arrays
        record_time_data = np.zeros((Q.shape[0]), dtype='float64')
        total_counts_data = np.zeros((Q.shape[0]), dtype='float64')
        data = np.zeros((Q.shape[0], 1024), dtype='float64')
        fno = 0
    
        for i in range(0, Q.shape[0]):
            for j in range(0, nfiles[i]):
                # Read one file at a time for each individual Q
                data_raw_file = open(fname[fno], 'r')
                print("Reading file " + fname[fno] + " for the Q-value " + str(Q[i]))
                fno = fno + 1
                lines = data_raw_file.readlines()
    
                # Get the total record time and total event counts
                for line in lines:
                    if "Total Time" in line:
                        record_time_data[i] = record_time_data[i] + float(line.split()[6])
                        total_counts_data[i] = total_counts_data[i] + float(line.split()[12])
    
                # Read from 8th line, leave the first element in split in every line
                k = 0  # Channel number
                for line in lines[7:110]:
                    for l in range(0, 10):
                        data[i, k] = data[i, k] + float(line.split()[l + 1])
                        k += 1
                        if k > 1023:
                            break
    
                data_raw_file.close()
    
            record_time_data[i] = record_time_data[i] / nfiles[i]
            total_counts_data[i] = total_counts_data[i] / nfiles[i]
            data[i, :] = data[i, :] / nfiles[i]

        return data
    
    
    def ns(self, i):
        
        
        
        self.figure2[i].clf()
        
        dt = self.data[i]
        bg = self.bkg[i]
        #checking whether peak value is integer
        if (self.flag_boxes[i].text() == ''):
            self.peak = None
        else:
            self.peak = int(self.flag_boxes[i].text())  
#        print( self.peak, type(self.peak))

                 
            
        #normalising the data and background with a constant
        data_1 = dt[ 300:400]
        bkg_1 = bg[300:400]
        factors = np.divide(bkg_1,data_1)
       
        squares = np.zeros(factors.shape[0])
        for k in range(0,factors.shape[0]):
           squares += (np.multiply(1/factors[k],bkg_1) -  data_1)**2 
        
        squares = squares/squares.shape[0]
        min_pos = np.argmin(squares)
        best_factor = factors[min_pos]
#        print( min_pos+300,data[min_pos+300])
        to_sub = dt[min_pos+300]
        
        #subtract the background with appropriate peak positions
        #find the peak positions first 
        ##################

        if (self.flag == 0):
            cno = np.arange(400,700,1)
            
            
            data1 = dt - to_sub    
            data_peak1 = 450+np.argmax(dt[450:700])    
            poptd, pcov = curve_fit(gaussian_u,cno,data1[400:700],p0=[1000,100,data_peak1])#,bounds=(lower,upper),verbose=2)
            data_peak = poptd[2]
            print ("B: ",poptd[-1])
            print (poptd)
            print ("Data")
            print  ("The centre channel from maxima: ",round(data_peak1,5))
            print ("The centre channel from the fitting: ",round(data_peak,5))
            print ("The difference between the two: ",round(abs(data_peak1-data_peak),5))
            data_peak = int(round(data_peak))
                    
            bkg1 = bg/best_factor - to_sub
            bkg_peak1 = 450+np.argmax(bg[450:700])
            poptb, pcov = curve_fit(gaussian_u,cno,bkg1[400:700],p0=[1000,100,bkg_peak1])#,bounds=(lower,upper))
            bkg_peak = poptb[2]
            print ("B: ",poptb[-1])
            print (poptb)
            print ("Background")
            print  ("The centre channel from maxima: ",round(bkg_peak1,5))
            print ("The centre channel from the fitting: ",round(bkg_peak,5))
            print ("The difference between the two: ",round(abs(bkg_peak1-bkg_peak),5))
            bkg_peak = int(round(bkg_peak))                  
        
        
            peak_diff = bkg_peak - data_peak
        
        
        elif (self.flag == 1):
            bkg_peak = 450+np.argmax(bg[450:700])
            data_peak = 450+np.argmax(dt[450:700])    
            peak_diff = bkg_peak - data_peak
            
        elif (self.flag == 2):
            bg_peak = np.argmax(bg[0:250])
            dt_peak = np.argmax(dt[0:250])
            peak_diff = bg_peak - dt_peak
        
        else: 
            peak_diff = self.peak
        
        
        print ("The background peak is ahead by : ",peak_diff)
        
        #Create an axis
        ax2 = self.figure2[i].add_subplot(111)
        
        
        if (peak_diff > 0):
            self.ns_data[i] = dt[:-peak_diff] - 1/best_factor*bg[peak_diff:] 
            
            ax2.plot(1/factors[min_pos]*bg[peak_diff:],'o',ms=3,alpha=0.3,fillstyle='none',color='red',label='Background')
            ax2.plot(dt[:-peak_diff],'o',ms=3,alpha=0.3,fillstyle='none',color='blue',label='Data')
        
        
        elif (peak_diff < 0):
            self.ns_data[i] = dt[-peak_diff:] - 1/best_factor*bg[:peak_diff] 
            
            ax2.plot(1/factors[min_pos]*bg[:peak_diff],'o',ms=3,alpha=0.3,fillstyle='none',color='red',label='Background')
            ax2.plot(dt[-peak_diff:],'o',ms=3,alpha=0.3,fillstyle='none',color='blue',label='Data')
        
        else:
            self.ns_data[i] =  dt - 1/best_factor*bg 
            
            ax2.plot(1/factors[min_pos]*bg,'o',ms=3,alpha=0.5,color='red',label='Background')
            ax2.plot(dt,'o',ms=3,fillstyle='none',alpha=0.3,color='blue',label='Data')
        
        pd_txt = "Peak difference:"+str(peak_diff)
        # txt = ax2.text(400,5000,str(peak_diff),fontsize=18)
    
        a = np.nan
        b = a
    
        ax2.plot(self.ns_data[i],label="Subtracted")
        
        ax2.plot(a,b,".",ms=1,label=pd_txt)
#        ax2.plot(bg[i],label="Bkg")
        
        ax2.set_xlabel("Channel")
        ax2.set_ylabel("Counts")
        
        ax2.set_ylim(-500,)
        ax2.grid()
    
        leg = ax2.legend(loc='upper right' )
        leg.set_alpha(1.0)
        leg.set_draggable(True)
        
        
        self.figure2[i].tight_layout()
        self.canvas2[i].draw()
        
#        self.ns_datal.append(self.ns_data)
        
    def clear_plots(self, tab_index):
        # Clear the existing plots in ax2
        self.figure2[tab_index].clf()  # Clear the figure
        
    def handle_selection_change(self, index, tab_index):
#        combo_box = self.flag_sels[tab_index]
        
        # Set the value of self.flag directly
        self.flag = index

        # Enable/disable the text box based on the flag value
        self.flag_boxes[tab_index].setEnabled(self.flag == 3)
        
    #writes a two column file taking two numpy arrays 
    def write_to_file(self):
        x = self.energy
        y = self.sqw
        fname = self.save_fname.text()        #'./test.dat'
        
        if (x.ndim > 1):
            print ("Can write only one dimensional x-axis array, The dimension of x-array is ", x.ndim)
        elif(y.ndim > 2):
            print ("Can write only two dimensional y-axis array, The dimension of y-array is ", y.ndim)
        elif (y.ndim == 2 and x.shape[0] ==  y.shape[1]):
            f = open(fname,'w')
            for i in range(x.shape[0]):
                ystr = ''
                for k in range(y.shape[0]):
                    ystr = ystr+'\t'+str(y[k,i])
                f.write(str(x[i])+ystr+'\n')
            f.close()
        elif (y.ndim == 1 and  x.shape[0] ==  y.shape[0]):
            f = open(fname,'w')
            for i in range(x.shape[0]):
                f.write(str(x[i])+'\t'+str(y[i])+'\n')
            f.close()
        else:
            print ("The arrays have incompatible number of rows with ", x.shape[0], " and ", y.shape[1])
        return    
        
    
    #conversion to energy scale
    def to_energy(self):
        
        Qlen = len(self.qvalues)
        
        
        
#    
        #create an empty list of arrays sqw and en
        sqw_tmp = []
        en_tmp = []
        
        # channel no. for maximum intensity
        for k in range(len(self.qvalues)):
            
            e0 = float(self.Ei_entry.text())#5.11
            theta = 73.69
            twod = float(self.d_entry.text()) #6.67
            con = 81.8
            kbt = 25.8
            cf = float(self.cf_entry.text())#0.04202
            
            mch1 = 350+np.argmax(self.ns_data[k][350:700]) #calculate this
            
            #calculate by fitting it with a gaussian_u
            #generate x-axis, by generating channel numbers from 250 to 750
            cno = np.arange(350,700,1)
            popt, pcov = curve_fit(gaussian_u,cno,self.ns_data[k][350:700],p0=[1000,100,mch1])
            sig = popt[1]
            print ("FWHM in channel numbers from gaussian_u fit: ", round(2.3548*abs(sig),5))
            mu = popt[2]
            print  ("The centre channel from maxima: ",round(mch1,5))
            print ("The centre channel from the fitting: ",round(mu,5))
            print ("The difference between the two: ",round(abs(mch1-mu),5))
            mch = int(round(mu))    
            
            mst = 300 #channel number to start with in conversion
        
#            mchs = mch - mst
            #channel no. to end with in conversion
            med = 2*mch - mst
            #angle in degrees
            theta = theta - (mch - mst)*cf
        
            print ("The starting, centre and ending channels are: ",mst,mch,med)
            
            sqw_l = np.zeros((med-mst),dtype='float64')
            energy_l = np.zeros((med-mst),dtype='float64')
        
            for i in range(0,med-mst):
                theta_r = theta*np.pi/(2.*180.)
                lam = twod*np.sin(theta_r)
                ljac = np.tan(theta_r)/lam
        
                #energy calculation
                energy_l[i] = con/lam**2 - e0
                #debye-waller factor
                dbf = np.exp(-energy_l[i]/(2*kbt))
                #calculation of S(Q,E)
                sqw_l[i] = dbf * ( lam**4. * ljac * self.ns_data[k][i+mst] )/( 8.*con )
        
                theta = theta + cf
        
            energy_l = np.flipud(energy_l)
            sqw_l = np.flipud(sqw_l)
            
            en_tmp.append(energy_l)
            sqw_tmp.append(sqw_l)
        
        
        delE = 0.02
        self.energy = np.arange(-1.0, 1.0+delE, delE)
        self.sqw = np.zeros((Qlen,self.energy.shape[0]),dtype='float64')
        
        self.entabs.clear()
        
        
        for i in range(0,len(self.qvalues)):
            self.sqw[i,:] = np.interp(self.energy,en_tmp[i],sqw_tmp[i])
            
            #Create a new tab for each qvalue
            etab = QWidget()
            self.entabs.addTab(etab, "Q"+str(i))
            
            efigure = Figure(figsize=(3,4), dpi=100)
            ecanvas = FigureCanvas(efigure)
            self.entabs_layout = QVBoxLayout(etab)
            self.entabs_layout.addWidget(NavigationToolbar(ecanvas, self))
            self.entabs_layout.addWidget(ecanvas)
                        
            
            axe = efigure.add_subplot(111)
            axe.plot(en_tmp[i], sqw_tmp[i],"x",color='red',ms=3,alpha=0.3)
            axe.plot(self.energy, self.sqw[i],"o",color='blue',ms=3,alpha=0.3)
            
            axe.set_xlabel("Energy (meV)")
            axe.set_ylabel("$S(Q,E)$")
            
            efigure.tight_layout()
            ecanvas.draw()
        
        self.fit_plot(self.energy, self.sqw)
        

    def extract_data(self):
        # Example function for data extraction
        self.qvalues = np.fromstring(self.qvalues_entry.text(), sep=',')
        path = self.path_entry.text()
        
        fn_data = np.fromstring(self.fno_entry_data.text(), dtype='int',sep=',')
        ind_data = np.fromstring(self.indices_entry_data.text(), dtype='int',sep=',')
        
        
        fn_bkg = np.fromstring(self.fno_entry_bkg.text(), dtype='int',sep=',')
        ind_bkg = np.fromstring(self.indices_entry_bkg.text(), dtype='int',sep=',')
        
#        print (qvalues)
#        print (file_number)
#        print (indices)
#        print (path)
        
        self.data = self.read_data(self.qvalues, fn_data, ind_data, path)
        self.bkg = self.read_data(self.qvalues, fn_bkg, ind_bkg, path)
#        print (data.shape)
        
        ## an empty list of normalized subtracted data 
#        self.ns_datal = []
        
        # Clear previous plot
        self.tabs.clear()

        self.figure2 = []
        self.canvas2 = []
        
        self.flag_sels = []
        self.flag_boxes = []
        
        #an empty list of normalized subtracted data
        self.ns_data = [None] * len(self.qvalues)
        
        # Plot all Q-data in individual tabs
        for i in range(len(self.qvalues)):
            #Create a new tab for each qvalue
            tab = QWidget()
            self.tabs.addTab(tab, "Q"+str(i))
            
                     
            #setup the matplotlib figure and canvas for the tab
            figure = Figure(figsize=(5,4), dpi=100)
            canvas = FigureCanvas(figure)
            self.tabs_layout = QVBoxLayout(tab)
            self.tabs_layout.addWidget(NavigationToolbar(canvas, self))
            self.tabs_layout.addWidget(canvas)
            
#            label_text = 
            label = QLabel(self.tabs)
            label.setTextFormat(Qt.RichText)
#            label.setText(label_text)
            
            #Create an axis
            ax = figure.add_subplot(111)
            
            ax.plot(self.data[i],label="Data")
            ax.plot(self.bkg[i],label="Bkg")
            
            ax.set_xlabel("Channel")
            ax.set_ylabel("Counts")
        
            ax.legend()
            figure.tight_layout()
            
            canvas.draw()
            
            #create an Hbox to hold drop down, flag input and buttons
            self.input_controls = QHBoxLayout()                      
            
            
            #drop down menu 
            flag_sel = QComboBox()
            flag_sel.addItem("Gaussian fit")
            flag_sel.addItem("Maxima method")
            flag_sel.addItem("Side peak method")
            flag_sel.addItem("Manual peak value")
            flag_sel.setCurrentIndex(2)
            
            self.flag_sels.append(flag_sel)
            
             # Connect the currentIndexChanged signal to handle_selection_change
            self.flag_sels[i].currentIndexChanged.connect(lambda index, i=i: self.handle_selection_change(index, i))
            
            # adding combo box into the input_controls layout
            self.input_controls.addWidget(self.flag_sels[i])
            
            
            
            
            # adding a text box into input_controls layout
            self.flag_box = QLineEdit(self)
            self.flag_box.setPlaceholderText("Peak position")
            
            self.flag_boxes.append(self.flag_box)
            
            self.flag_boxes[i].textChanged.connect(lambda text, i=i: self.clear_plots(i))

            
            self.input_controls.addWidget(self.flag_boxes[i])
            # Disable the text box initially
            self.flag_boxes[i].setEnabled(False)
            
             #button for normalize and subtract
            ns_button = QPushButton("Normalize and subtract")
            ns_button.clicked.connect(lambda checked, i=i: self.ns(i))
            self.input_controls.addWidget(ns_button)
            
            
            #adding input_controls layout into tabs_layout
            self.tabs_layout.addLayout(self.input_controls)
            
            #setup the matplotlib figure and canvas for the tab
            self.figure2.append( Figure(figsize=(5,4), dpi=100))
            self.canvas2.append( FigureCanvas(self.figure2[i]))
            
            self.tabs_layout.addWidget(NavigationToolbar(self.canvas2[i], self))
            self.tabs_layout.addWidget(self.canvas2[i])
        
            
            
#            self.canvas2[i].draw()
                

if __name__ == "__main__":
    app = QApplication(sys.argv)
    my_app = MyApplication()
    my_app.show()
    sys.exit(app.exec_())
