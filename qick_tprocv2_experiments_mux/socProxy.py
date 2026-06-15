import Pyro4
from qick import QickConfig

def makeProxy():
    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

    ns_host = "pynq216-3.dhcp.fnal.gov" #nexus board
    ns_port = 8889
    proxy_name = "myqick"

    ns = Pyro4.locateNS(host=ns_host, port=ns_port)
    soc = Pyro4.Proxy(ns.lookup(proxy_name))
    soccfg = QickConfig(soc.get_cfg())
    print(soccfg)
    return(soc,soccfg)