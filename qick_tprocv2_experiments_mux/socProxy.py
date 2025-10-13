import Pyro4
from qick import QickConfig

device = "hBN_resonator" ##switch between devices here! updates throughout code

def makeProxy():
    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

    if device == "control_resonator":
        ns_host = "rfsoc216-loud0.dhcp.fnal.gov" # loud1 is the QICK RF board, loud2 is the "old QICK" board
    elif device == "hBN_resonator":
        ns_host = "rfsoc216-loud2.dhcp.fnal.gov" ## check this host address

    ns_port = 8888
    proxy_name = "myqick"

    ns = Pyro4.locateNS(host=ns_host, port=ns_port)
    soc = Pyro4.Proxy(ns.lookup(proxy_name))
    soccfg = QickConfig(soc.get_cfg())
    print(soc)
    print(soccfg)
    return(soc,soccfg)