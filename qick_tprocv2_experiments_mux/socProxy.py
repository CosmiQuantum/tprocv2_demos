import Pyro4
from qick import QickConfig

def makeProxy():
    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

    ns_host = "rfsoc216-cosmiq01.dhcp.fnal.gov" # original QICK box (runs 8-9): "rfsoc216-loud1.dhcp.fnal.gov". New box: rfsoc216-cosmiq01.dhcp.fnal.gov
    ns_port = 8888
    proxy_name = "myqick"

    ns = Pyro4.locateNS(host=ns_host, port=ns_port)
    soc = Pyro4.Proxy(ns.lookup(proxy_name))
    soccfg = QickConfig(soc.get_cfg())
    #print(soccfg)
    return(soc,soccfg)