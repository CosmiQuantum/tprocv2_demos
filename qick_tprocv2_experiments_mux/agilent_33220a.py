import pyvisa
import time

class Agilent33220A:
    def __init__(self,ip_address,timeout=5000):
        """
        ip_adress:string,e.g. '192.168.1.198'
        timeout: VISA timeout in ms
        """
        self.ip=ip_address
        self.resource_str=f"TCPIP0::{ip_address}::inst0::INSTR"
        
        self.rm=pyvisa.ResourceManager()
        self.inst=self.rm.open_resource(self.resource_str)
        self.inst.timeout=timeout
        
        #Optional:cleaner communication
        self.inst.write_termination = '\n'
        self.inst.read_termination= '\n'
        
    def idn(self):
        """Query instrument identification"""
        return self.inst.query("*IDN?")
       
    def set_voltage(self,voltage):
    	self.inst.write(f"VOLT {voltage}")
    	
    def output_on(self):
        self.inst.write("OUTP ON")
	
    def output_off(self):
        """" Turn output OFF"""
        self.inst.write("OUTP OFF")
       
    def set_pulse_high(self, high):
    	#if high > 2.5:
    	    #raise ValueError("Voltage exceeds limit")
    	self.inst.write("FUNC PULS")
    	self.inst.write(f"VOLT:HIGH {high}")
        
    def close(self):
    	"""" Close VISA session"""
    	self.inst.close()
    	self.rm.close()	     
        
        
