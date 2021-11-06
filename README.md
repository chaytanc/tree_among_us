# tree among us

# Connecting the Muse  
1) see these instructions in the README.md https://github.com/synaptech-uw/docs/tree/master/Muse/muse_python  
2) Plug in USB Bluetooth thingy (BLED112)  
3) Type   
muselsl list -b bgapi  
in the terminal to find your Muse's ip address (00:55:DA:B5:74:8A for today) 
3.5) Put Muse in pairing mode by holding down power button until lights are blinking or something??
4) Type  
muselsl stream -b bgapi -a 00:55:DA:B5:74:8A 
to connect (or replace w your specific ip address)  
5) Download and install LabRecorder (https://github.com/labstreaminglayer/App-LabRecorder/releases)  
6) Open LabRecorder (navigate to where you installed it and type  
open LabRecorder  
)
7) Stream, capture to a xdf file and process  
