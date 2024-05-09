% Define a callback function to handle incoming data
function bleDataReceived(src, ~)
    
    global finished
    global first
    global elapsedTime
    global characteristicData
    global count
    global count_last
    global numBytesReceived
    global c_write
    global c_read
    BUFFER_LENGTH=244; 
    datacounter=0;
    datagenerated=[];
    last_value1=0;
    first_send=1;
    if (first==1)
        tic;
        first=0;
    end
    if finished==0
        newData = read(src, 'latest');
        last_value1=newData(end);
        last_value2=newData(end-1);
        numBytesReceived = numBytesReceived + size(newData,2);
        if size(newData,2)<244
           paddingSize = 244 - size(newData,2);
           zerorow=zeros(1,paddingSize);
           newData = [newData, zerorow]; 
        end
        characteristicData = [characteristicData; newData];
    end
    
    % Check for the end of transmission
    if numBytesReceived >= 2 && last_value1 == 255 && last_value2 == 255
        finished=1;
        disp('Data collection finished.');
        unsubscribe(c_read);
        elapsedTime = toc;
        % compute throughput
        numBitsReceived = numBytesReceived * 8; % Assuming each value is 1 byte (8 bits)
        throughput_receive = numBitsReceived / (elapsedTime*1000) % Bits per milliseconds
        NUMBER_OF_CYCLES = floor(numBytesReceived/ BUFFER_LENGTH);
        REMAINING_8BITS_PACKETS = mod(numBytesReceived, BUFFER_LENGTH);
        data_received=zeros(1, (NUMBER_OF_CYCLES+1)*194+153);
        for i = 1:NUMBER_OF_CYCLES
            for j = 1:5:236
                data_received(count + 1) = bitor(bitshift(uint16(characteristicData(i,j)), 2), bitshift(uint16(characteristicData(i,j+1)), -6));
                data_received(count + 2) = bitor(bitshift(bitand(uint16(characteristicData(i,j+1)), uint16(0x3F)), 4), bitshift(bitand(uint16(characteristicData(i,j+2)), uint16(0xF0)), -4));
                data_received(count + 3) = bitor(bitshift(bitand(uint16(characteristicData(i,j+2)), uint16(0x0F)), 6), bitshift(bitand(uint16(characteristicData(i,j+3)), uint16(0xFC)), -2));
                data_received(count + 4) = bitor(bitshift(bitand(uint16(characteristicData(i,j+3)), uint16(0x03)), 8), uint16(characteristicData(i,j+4)));
                count = count + 4;
            end
            for j = 241:2:BUFFER_LENGTH
                count = count + 1;
                data_received(count) = bitor(bitshift(uint16(characteristicData(i,j+1)), 8), uint16(characteristicData(i,j)));
            end
        end
        i = NUMBER_OF_CYCLES+1;
        for j = 1:5:(REMAINING_8BITS_PACKETS-6)
            data_received(count + 1) = bitor(bitshift(uint16(characteristicData(i,j)), 2), bitshift(uint16(characteristicData(i,j+1)), -6));
            data_received(count + 2) = bitor(bitshift(bitand(uint16(characteristicData(i,j+1)), uint16(0x3F)), 4), bitshift(bitand(uint16(characteristicData(i,j+2)), uint16(0xF0)), -4));
            data_received(count + 3) = bitor(bitshift(bitand(uint16(characteristicData(i,j+2)), uint16(0x0F)), 6), bitshift(bitand(uint16(characteristicData(i,j+3)), uint16(0xFC)), -2));
            data_received(count + 4) = bitor(bitshift(bitand(uint16(characteristicData(i,j+3)), uint16(0x03)), 8), uint16(characteristicData(i,j+4)));
            count = count + 4;
            count_last = j+4;
        end
        count_last = count_last + 1;
        for j = count_last:2:REMAINING_8BITS_PACKETS+1
            count = count + 1;
            data_received(count) = bitor(bitshift(uint16(characteristicData(i,j+1)), 8), uint16(characteristicData(i,j)));
        end
        assignin('base', 'exported_data_received', data_received);
          %% send data
        for i=1:32400
            if datacounter>253
                datacounter=0;
            else 
                datacounter=datacounter+1;
            end
            if i==32400-1 || i==32400
                datacounter=255;
            end
            datagenerated=[datagenerated, datacounter];
            if size(datagenerated, 2)==BUFFER_LENGTH || (i==32400)
                write(c_write, datagenerated,'WithoutResponse');
                if first_send==1
                   first_send=0;
                   tic
                end
                if i==32400
                   elapsedTime_send=toc;
                   throughput_send=i*8/(elapsedTime_send*1000)
                end
                datagenerated=[];
           end
        end
        first=1;
        finished=0;
        numBytesReceived=0;
        subscribe(c_read);
        c_read.DataAvailableFcn = @bleDataReceived;
    end 
end

