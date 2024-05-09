clear all
close all
clc

%% settings
% Get the list of all current BLE connections
bleConnections = blelist;
% Display the available Bluetooth device names
disp(bleConnections);

device = "Senseback";
b = ble("D238532C0635");
% % Define the service and characteristic UUIDs
serviceUUID_write = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
characteristicUUID_write = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";
global c_write
% Find the characteristic object
c_write = characteristic(b, serviceUUID_write, characteristicUUID_write);

% Define the service and characteristic UUIDs for reading
serviceUUID_read = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
CharacteristicUUID_read = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";
global c_read
% Find the characteristic object for reading data
c_read = characteristic(b, serviceUUID_read, CharacteristicUUID_read);
subscribe(c_read);
%% define variables

global finished
global first
global elapsedTime
global characteristicData
global count
global count_last
global numBytesReceived
global BUFFER_LENGTH
first=1;
characteristicData=[];
finished=0;
numBytesReceived=0;
count=0;
count_last=0;
BUFFER_LENGTH=244;
global datagenerated
datagenerated=[];
global datacounter
datacounter=0;
finished=0;
%% write START
val = input('Comando: ','s');
% Convert string to uint8 array
val_bytes = uint8(val);
% command start
write(c_write,val_bytes,'WithoutResponse');

%% read data
% Set the callback function for the characteristic
if finished==0
    % Set the callback function for the characteristic
    c_read.DataAvailableFcn = @bleDataReceived;
elseif finished==1
    c_read.DataAvailableFcn = [];
end
%% write STOP
val = input('Comando: ','s');
% Convert string to uint8 array
val_bytes = uint8(val);
% command start
write(c_write,val_bytes,'WithoutResponse');


    






     