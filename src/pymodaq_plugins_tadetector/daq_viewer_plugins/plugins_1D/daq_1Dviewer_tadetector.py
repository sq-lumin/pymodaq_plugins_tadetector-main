import numpy as np
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter

from pymodaq_plugins_chopper.daq_viewer_plugins.plugins_0D.daq_0Dviewer_chopper import DAQ_0DViewer_chopper
from pymodaq_plugins_shutters.daq_viewer_plugins.plugins_0D.daq_0Dviewer_shutters import DAQ_0DViewer_shutters
from pymodaq_plugins_lightfield.daq_viewer_plugins.plugins_1D.daq_1Dviewer_lightfield import DAQ_1DViewer_lightfield

from qtpy.QtCore import QThread

from time import perf_counter_ns, time

#This plugins subclasses my lightfield, shutters and chopper plugins. Order is important !
class DAQ_1DViewer_tadetector(DAQ_1DViewer_lightfield, DAQ_0DViewer_chopper, DAQ_0DViewer_shutters):
    """ Instrument plugin class for a 1D viewer.
    
    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    TODO Complete the docstring of your plugin with:
        * The set of instruments that should be compatible with this instrument plugin.
        * With which instrument it has actually been tested.
        * The version of PyMoDAQ during the test.
        * The version of the operating system.
        * Installation instructions: what manufacturer’s drivers should be installed to make it run?

    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.
         
    # TODO add your particular attributes here if any

    """
    #I have to do this to avoid repeating the common params mutiple times
    params = (comon_parameters + [{'title':'Take Backgrounds', 'name':'take_background', 'type' : 'bool_push'},
                                  {'title':'Compute TA stdev', 'name':'compute_sigTA', 'type' : 'bool', 'value' : False}]
                                  + [param for param in DAQ_1DViewer_lightfield.params if param['name'] == 'lightfield_params'] 
                                  + [param for param in DAQ_0DViewer_shutters.params if param['name'] == 'shutters_params']
                                  + [param for param in DAQ_0DViewer_chopper.params if param['name'] == 'chopper_params']
                                  + [{'title' : 'Threshold Voltage (V)', 'name' : 'threshold_voltage', 'type' : 'float', 'value' : 2.5},
                                     {'title' : 'Decision Index', 'name' : 'decision_index', 'type' : 'int', 'value' : 6}])
    
    def __init__(self, parent=None, params_state=None):
        for parent_cls in DAQ_1DViewer_tadetector.__bases__:
            parent_cls.__init__(self, parent = parent, params_state = params_state)
        #identical to : 
        # DAQ_1DViewer_lightfield.__init__(self, parent = parent, params_state = params_state)
        # DAQ_0DViewer_chopper.__init__(self, parent = parent, params_state = params_state)
        # DAQ_0DViewer_shutters.__init__(self, parent = parent, params_state = params_state)
        
        
    def ini_attributes(self):
        for parent_cls in DAQ_1DViewer_tadetector.__bases__:
            parent_cls.ini_attributes(self)
        #identical to :
        # DAQ_1DViewer_lightfield.ini_attributes(self)
        # DAQ_0DViewer_chopper.ini_attributes(self)
        # DAQ_0DViewer_shutters.ini_attributes(self)
        self._compute_sigTA = False
        self.chopper_data = None
        self._chopper_finished = False
        self._background_dark = 0
        self._background_fluo = 0
        self._sigBackground_fluo = 0
        self._sigBackground_dark = 0
        self.controller= None
        self.x_axis = None
        #for timing scans
        self.t1 = 0
        self.t2 = 0
        
    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        ## TODO for your custom plugin
        if param.name() == "take_background":
            if param.value():
                self.take_background()
                param.setValue(False)   #Calls commit_settings again, not ideal.
        elif param.name() == 'compute_sigTA':
            self._compute_sigTA = param.value()
#        elif ...
        else:
            for parent_cls in DAQ_1DViewer_tadetector.__bases__:
                parent_cls.commit_settings(self, param)
            #identical to :
            # DAQ_1DViewer_lightfield.commit_settings(self, param)
            # DAQ_0DViewer_chopper.commit_settings(self, param)
            # DAQ_0DViewer_shutters.commit_settings(self, param)

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """
        for parent_cls in DAQ_1DViewer_tadetector.__bases__:
            parent_cls.ini_detector(self, self.controller)
        #identical to :
        # DAQ_1DViewer_lightfield.ini_detector(self, self.controller)
        # DAQ_0DViewer_chopper.ini_detector(self, self.controller)
        # DAQ_0DViewer_shutters.ini_detector(self, self.controller)

        data_x_axis = self.get_x_axis() 
        self._background_dark = np.zeros_like(data_x_axis)
        self._background_fluo = np.zeros_like(data_x_axis)
        self._sigBackground_dark = np.zeros_like(data_x_axis)
        self._sigBackground_fluo = np.zeros_like(data_x_axis)
        self.x_axis = Axis(data=data_x_axis, label='Wavelength', units='pixels', index=0)
        self.dte_signal_temp.emit(DataToExport('tadetector',
                                          data=[DataFromPlugins(name='I_ON', data= [data_x_axis],
                                                                dim='Data1D', labels=['I_ON'], 
                                                                axes=[self.x_axis],
                                                                plot = False, save = True), 
                                                DataFromPlugins(name='I_OFF', data=[data_x_axis],
                                                                dim='Data1D', labels=['I_OFF'],
                                                                axes=[self.x_axis],
                                                                plot = False, save = True),
                                                DataFromPlugins(name='TA', data=[data_x_axis],
                                                                dim='Data1D', labels=['TA'],
                                                                axes=[self.x_axis],
                                                                plot = True, save = False), 
                                                DataFromPlugins(name='WL', data=[data_x_axis],
                                                                dim='Data1D', labels=['WL'],
                                                                axes=[self.x_axis],
                                                                plot = True, save = False),
                                                ]))
        info = "Whatever info you want to log"
        initialized = True
        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        #call the close method of every parent plugin
        for parent_cls in DAQ_1DViewer_tadetector.__bases__:
            parent_cls.close(self)
    
    def get_x_axis(self):
        return np.array(range(1024))
    
    def grab_data(self, Naverage=1, **kwargs):
        """Start a grab from the detector

        Parameters
        ----------
        Naverage: int
            Number of hardware averaging (if hardware averaging is possible, self.hardware_averaging should be set to
            True in class preamble and you should code this implementation)
        kwargs: dict
            others optionals arguments
        """
        also_do_bkg = False
        try:
            #Detect if we need to also take a background. happens during the first step of each scan.
            also_do_bkg = kwargs['also_do_bkg']
        except KeyError:
            pass
        
        if also_do_bkg:
            self.take_background()
            
        I_ON, I_OFF = self._my_grab_data(Naverage, **kwargs) 

        if self._compute_sigTA:
            TA, sigTA = self.compute_TA(I_ON, I_OFF, bg_dark = self._background_dark, bg_fluo = self._background_fluo,
                                      stds = [self._sigBackground_dark, self._sigBackground_fluo])
            #WL = self.compute_WL_spectrum(I_OFF, bg_dark = self._background_dark)
            dte = DataToExport('tadetector',
                                data=[DataFromPlugins(name='I_ON', data=[np.mean(I_ON, axis = 1)],
                                                      dim='Data1D', labels=['I_ON'],
                                                      axes=[self.x_axis],
                                                      errors = [np.std(I_ON, axis = 1)],
                                                      plot = False, save = True), 
                                      DataFromPlugins(name='I_OFF', data=[np.mean(I_OFF, axis = 1)],
                                                      dim='Data1D', labels=['I_OFF'],
                                                      axes=[self.x_axis],
                                                      errors = [np.std(I_OFF, axis = 1)],
                                                      plot = False, save = True),
                                      DataFromPlugins(name='TA', data=[TA],
                                                      dim='Data1D', labels=['TA'],
                                                      axes=[self.x_axis],
                                                      errors = [sigTA],
                                                      plot = True, save = True)
                                      ])
        else:
            TA = self.compute_TA(I_ON, I_OFF, bg_dark = self._background_dark, bg_fluo = self._background_fluo)
            #WL = self.compute_WL_spectrum(I_OFF, bg_dark = self._background_dark)
            dte = DataToExport('tadetector',
                                data=[DataFromPlugins(name='I_ON', data=[np.mean(I_ON, axis = 1)],
                                                      dim='Data1D', labels=['I_ON'],
                                                      axes=[self.x_axis],
                                                      plot = False, save = True), 
                                      DataFromPlugins(name='I_OFF', data=[np.mean(I_OFF, axis = 1)],
                                                      dim='Data1D', labels=['I_OFF'],
                                                      axes=[self.x_axis],
                                                      plot = False, save = True),
                                      DataFromPlugins(name='TA', data=[TA],
                                                      dim='Data1D', labels=['TA'],
                                                      axes=[self.x_axis], plot = True, save = True)
                                      ])
        
        # dte = DataToExport('tadetector',
        #                    data = [DataFromPlugins(name='I_ON', data=[np.mean(I_ON, axis = 1)],
        #                                            dim='Data1D', labels=['I_ON'],
        #                                            axes=[self.x_axis],
        #                                            plot = True, save = True),
        #                            DataFromPlugins(name='I_OFF', data=[np.mean(I_OFF, axis = 1)],
        #                                            dim='Data1D', labels=['I_OFF'],
        #                                            axes=[self.x_axis],
        #                                            plot = True, save = True)])
        
        if also_do_bkg:
            dte.append(self._bg_dte)

        self.dte_signal.emit(dte)
        self.t1 = perf_counter_ns()
        print('detector step time :',self.t1-self.t2)
        self.t2 = self.t1
        ##asynchrone version (non-blocking function with callback)
        #self.controller.your_method_to_start_a_grab_snap(self.callback)
        #########################################################
    
    def _my_grab_data(self, Naverage=1, **kwargs):
        """My method for grabbing without emitting"""
        #necessary for the chopper
        update = False

        # if 'live' in kwargs:
        #     if kwargs['live'] != self.live:
        #         update = True
        #     self.live = kwargs['live']

        if Naverage != self.Naverage:
            self.Naverage = Naverage
            update = True

        if update:
            self.update_tasks()
        
        #call acquisition methods from the chopper and the camera plugins
        self._chopper_finished = False
        self.read_chopper(callback = self.chopper_callback)
        #QThread.msleep(100) #Does this fix spectra flipping ? no
        self.controller['chopper'].isTaskDone() #should be false
        image_array = self.capture_spectra()
        while not self._chopper_finished:
            #Be sure to wait for the chopper to finish his acquisition
            #Probably not needed since reading a few points from the chopper is much faster than acquiring all the spectra
            pass
        #decide if the first spectrum was pump ON or pump OFF by looking at the 5th (set by the param 'decision_index') point from the chopper
        started_ON = self.chopper_data[self.settings.child('decision_index').value()] > self.settings.child('threshold_voltage').value()
        if started_ON:
            I_OFF = image_array[:, ::2]
            I_ON = image_array[:, 1::2]
        else:
            I_ON = image_array[:, ::2]
            I_OFF = image_array[:, 1::2]
        #Lightfield returns values as uint16, convert to the standard int64
        I_ON, I_OFF = I_ON.astype(np.int64), I_OFF.astype(np.int64) 
        
        return I_ON, I_OFF
    
    def compute_TA(self, I_ON, I_OFF, bg_dark = 0, bg_fluo = 0, stds = None):
        """Computes the TA signal from I_ON, I_OFF and their backgrounds. 
        if standard deviations are specified as a list : [sigbg_dark, sigbg_fluo], also computes sigTA"""
        TA = -1e3*np.mean(np.log10((I_ON-bg_fluo[:,None])/(I_OFF-bg_dark[:,None])), axis = 1)
        if stds is not None:
            sigBg_dark, sigBg_fluo = stds
            sigI_ON, sigI_OFF = np.std(I_ON, axis = 1)/np.sqrt(500),  np.std(I_OFF, axis = 1)/np.sqrt(1000) #500 and 1000 needs to be changed based on the numer of spectra taken
            sigTA = 1e3*np.sqrt((((sigI_ON**2+sigBg_fluo**2)/((np.mean(I_ON, axis = 1)-bg_fluo)**2)) +
                                 ((sigI_OFF**2+sigBg_dark**2)/((np.mean(I_OFF, axis = 1)-bg_dark)**2))))
            return TA, sigTA
        else:
            return TA
    
    def compute_WL_spectrum(self, I_OFF, bg_dark = 0):
        WL_spectrum = np.mean(I_OFF-bg_dark[:,None], axis = 1)
        return WL_spectrum
    
    def chopper_callback(self, taskhandle, status, samples=0, callbackdata=None):
        self.chopper_data = self.controller['chopper'].readAnalog(len(self.channels), self.clock_settings)
        DAQ_0DViewer_chopper.stop(self)
        self._chopper_finished = True
        return 0  #mandatory for the PyDAQmx callback

    def take_background(self, param = None):
        self.emit_status(ThreadCommand('Update_Status', ['Taking Backgrounds...']))
        
        #close both shutters
        self.activateShutter('Probe', False)
        self.activateShutter('Pump', False)
        QThread.msleep(500)  #wait for the shutters to close. Is 500ms enough ?
        
        I_ON, I_OFF = self._my_grab_data(Naverage=1)
        self._background_dark = 0.5*np.mean(I_ON + I_OFF, axis = 1)
        self._sigBackground_dark = 0.5*np.std(I_ON + I_OFF, axis = 1)
        
        #close probe, open pump
        self.activateShutter('Probe', False)
        self.activateShutter('Pump', True)
        QThread.msleep(500)  #wait for the shutters to close. Is 500ms enough ?
        
        I_ON, I_OFF = self._my_grab_data(Naverage=1)
        self._background_fluo = np.mean(I_ON, axis = 1) #only I_ON counts, I_OFF has no pump
        self._sigBackground_fluo = np.std(I_ON, axis = 1)
        self._bg_dte = DataToExport('tadetector',
                                          data=[DataFromPlugins(name='Bg_fluo', data=[self._background_fluo],
                                                                dim='Data1D', labels=['Bg_fluo'],
                                                                axes=[self.x_axis],
                                                                errors = [self._sigBackground_fluo],
                                                                plot = True, save = False),
                                                DataFromPlugins(name='Bg_dark', data=[self._background_dark],
                                                                dim='Data1D', labels=['Bg_dark'],
                                                                axes=[self.x_axis],
                                                                errors = [self._sigBackground_dark],
                                                                plot = True, save = False)])
        #self.dte_signal.emit(self._bg_dte)
        
        #self._background_fluo = np.zeros_like(self._background_fluo)
        #self._background_dark = np.zeros_like(self._background_dark)
        
        #Put the shutters back in their intial state
        self.activateShutter('Probe', self.shutter_probe)
        self.activateShutter('Pump', self.shutter_pump)
        
        self.emit_status(ThreadCommand('Update_Status', ['Backgrounds taken !']))
    
    def get_bg_as_dte(self):
        return DataToExport('tadetector',
                                          data=[DataFromPlugins(name='Bg_fluo', data=self._background_fluo,
                                                                dim='Data1D', labels=['Bg_fluo'],
                                                                axes=[self.x_axis]), 
                                                DataFromPlugins(name='Bg_dark', data=self._background_dark,
                                                                dim='Data1D', labels=['Bg_dark'],
                                                                axes=[self.x_axis])])    
    
    def callback(self):
        """optional asynchrone method called when the detector has finished its acquisition of data"""
        raise NotImplementedError
        data_tot = self.controller.your_method_to_get_data_from_buffer()
        self.dte_signal.emit(DataToExport('myplugin',
                                          data=[DataFromPlugins(name='Mock1', data=data_tot,
                                                                dim='Data1D', labels=['dat0', 'data1'])]))

    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        #call the stop method of every parent plugin EXCEPT THE SHUTTERS
        DAQ_1DViewer_lightfield.stop(self)
        DAQ_0DViewer_chopper.stop(self)
        #for parent_cls in DAQ_1DViewer_tadetector.__bases__:
        #    parent_cls.stop(self)
        self.emit_status(ThreadCommand('Update_Status', ['Main plugin stopped']))
        ##############################
        return ''


if __name__ == '__main__':
    main(__file__)
