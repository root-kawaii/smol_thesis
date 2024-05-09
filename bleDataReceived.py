import time
from pygatt import BLEDevice, Characteristic, exceptions

# settings
serviceUUID_write = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
characteristicUUID_write = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
serviceUUID_read = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
characteristicUUID_read = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# # Connect to BLE device
# try:
#     device = BLEDevice("D238532C0635")
#     device.connect()
#     print("Connected to BLE device")
# except exceptions.NotConnectedError:
#     print("Failed to connect to BLE device")

# # Find the characteristic objects for writing and reading data
# c_write = device.char_write(serviceUUID_write, characteristicUUID_write)
# c_read = device.char_read(serviceUUID_read, characteristicUUID_read)


# Define callback function for handling received data
def ble_data_received(
    handle,
    value,
    first,
    finished,
    numBytesReceived,
    characteristicData,
    c_read,
    device,
    c_write,
):
    # global finished, first, elapsedTime, characteristicData, count, count_last, numBytesReceived, c_write, c_read
    BUFFER_LENGTH = 244
    datacounter = 0
    datagenerated = []
    last_value1 = 0
    first_send = 1

    if first == 1:
        start_time = time.time()
        first = 0

    if finished == 0:
        newData = value
        last_value1 = newData[-1]
        last_value2 = newData[-2]
        numBytesReceived += len(newData)

        if len(newData) < 244:
            paddingSize = 244 - len(newData)
            zerorow = [0] * paddingSize
            newData += zerorow

        characteristicData.append(newData)

    # Check for the end of transmission
    if numBytesReceived >= 2 and last_value1 == 255 and last_value2 == 255:
        finished = 1
        print("Data collection finished.")
        elapsedTime = time.time() - start_time
        throughput_receive = (
            numBytesReceived * 8 / (elapsedTime * 1000)
        )  # Bits per milliseconds

        NUMBER_OF_CYCLES = numBytesReceived // BUFFER_LENGTH
        REMAINING_8BITS_PACKETS = numBytesReceived % BUFFER_LENGTH
        data_received = []

        for i in range(NUMBER_OF_CYCLES):
            for j in range(0, 236, 5):
                data_received.append(
                    (characteristicData[i][j] << 2)
                    | (characteristicData[i][j + 1] >> 6)
                )
                data_received.append(
                    ((characteristicData[i][j + 1] & 0x3F) << 4)
                    | (characteristicData[i][j + 2] >> 4)
                )
                data_received.append(
                    ((characteristicData[i][j + 2] & 0x0F) << 6)
                    | (characteristicData[i][j + 3] >> 2)
                )
                data_received.append(
                    ((characteristicData[i][j + 3] & 0x03) << 8)
                    | characteristicData[i][j + 4]
                )

            for j in range(241, BUFFER_LENGTH, 2):
                data_received.append(
                    (characteristicData[i][j + 1] << 8) | characteristicData[i][j]
                )

        for j in range(0, REMAINING_8BITS_PACKETS - 6, 5):
            data_received.append(
                (characteristicData[NUMBER_OF_CYCLES][j] << 2)
                | (characteristicData[NUMBER_OF_CYCLES][j + 1] >> 6)
            )
            data_received.append(
                ((characteristicData[NUMBER_OF_CYCLES][j + 1] & 0x3F) << 4)
                | (characteristicData[NUMBER_OF_CYCLES][j + 2] >> 4)
            )
            data_received.append(
                ((characteristicData[NUMBER_OF_CYCLES][j + 2] & 0x0F) << 6)
                | (characteristicData[NUMBER_OF_CYCLES][j + 3] >> 2)
            )
            data_received.append(
                ((characteristicData[NUMBER_OF_CYCLES][j + 3] & 0x03) << 8)
                | characteristicData[NUMBER_OF_CYCLES][j + 4]
            )

        for j in range(REMAINING_8BITS_PACKETS - 6, REMAINING_8BITS_PACKETS + 1, 2):
            data_received.append(
                (characteristicData[NUMBER_OF_CYCLES][j + 1] << 8)
                | characteristicData[NUMBER_OF_CYCLES][j]
            )

        print("Throughput (receive):", throughput_receive)
        print("Data received:", data_received)

        # device.unsubscribe(characteristicUUID_read)
        characteristicData = []
        count = 0
        count_last = 0
        numBytesReceived = 0
        # subscribe(c_read)
        # device.unsubscribe(characteristicUUID_read)
        # device.disconnect()
        c_read.char_read_callback = ble_data_received
