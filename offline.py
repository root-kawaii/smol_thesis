import time
from pygatt import BLEDevice, exceptions
from bleDataReceived import ble_data_received

# settings
serviceUUID_write = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
characteristicUUID_write = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
serviceUUID_read = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
characteristicUUID_read = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# Connect to BLE device
try:
    device = BLEDevice("D238532C0635")
    device.connect()
    print("Connected to BLE device")
except exceptions.NotConnectedError:
    print("Failed to connect to BLE device")

# Find the characteristic objects for writing and reading data
c_write = device.char_write(serviceUUID_write, characteristicUUID_write)
c_read = device.char_write(serviceUUID_read, characteristicUUID_read)


# Subscribe to notifications for reading data
device.subscribe(characteristicUUID_read, callback=ble_data_received(7, 1))

# Write command to start
command_start = input("Command (start): ")
c_write.write_value(command_start.encode(), wait_for_response=False)

# Read data
time.sleep(2)  # Wait for data to be received

# Write command to stop
command_stop = input("Command (stop): ")
c_write.write_value(command_stop.encode(), wait_for_response=False)

# Unsubscribe from notifications
device.unsubscribe(characteristicUUID_read)
device.disconnect()
