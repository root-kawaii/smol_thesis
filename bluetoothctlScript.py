import time
import pexpect
import subprocess
import sys


class BluetoothctlError(Exception):
    """This exception is raised, when bluetoothctl fails to start."""

    pass


class Bluetoothctl:
    """A wrapper for bluetoothctl utility."""

    def __init__(self):
        out = subprocess.check_output("rfkill unblock bluetooth", shell=True)
        self.child = pexpect.spawn("bluetoothctl", echo=False)

    def get_output(self, command, pause=1):
        """Run a command in bluetoothctl prompt, return output as a list of lines."""
        self.child.send(command + "\n")
        time.sleep(pause)
        start_failed = self.child.expect(["bluetooth", pexpect.EOF])

        if start_failed:
            raise BluetoothctlError("Bluetoothctl failed after running " + command)

        return self.child.before.split("\r\n")

    def start_scan(self):
        """Start bluetooth scanning process."""
        try:
            out = self.get_output("scan on")
        except BluetoothctlError as e:
            print(e)
            return None

    def make_discoverable(self):
        """Make device discoverable."""
        try:
            out = self.get_output("discoverable on")
        except BluetoothctlError as e:
            print(e)
            return None

    def parse_device_info(self, info_string):
        """Parse a string corresponding to a device."""
        device = {}
        block_list = ["[\x1b[0;", "removed"]
        string_valid = not any(keyword in info_string for keyword in block_list)

        if string_valid:
            try:
                device_position = info_string.index("Device")
            except ValueError:
                pass
            else:
                if device_position > -1:
                    attribute_list = info_string[device_position:].split(" ", 2)
                    device = {
                        "mac_address": attribute_list[1],
                        "name": attribute_list[2],
                    }

        return device

    def get_available_devices(self):
        """Return a list of tuples of paired and discoverable devices."""
        try:
            out = self.get_output("devices")
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            available_devices = []
            for line in out:
                device = self.parse_device_info(line)
                if device:
                    available_devices.append(device)

            return available_devices

    def get_paired_devices(self):
        """Return a list of tuples of paired devices."""
        try:
            out = self.get_output("paired-devices")
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            paired_devices = []
            for line in out:
                device = self.parse_device_info(line)
                if device:
                    paired_devices.append(device)

            return paired_devices

    def get_discoverable_devices(self):
        """Filter paired devices out of available."""
        available = self.get_available_devices()
        paired = self.get_paired_devices()

        return [d for d in available if d not in paired]

    def get_device_info(self, mac_address):
        """Get device info by mac address."""
        try:
            out = self.get_output("info " + mac_address)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            return out

    def pair(self, mac_address):
        """Try to pair with a device by mac address."""
        try:
            out = self.get_output("pair " + mac_address, 4)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            res = self.child.expect(
                ["Failed to pair", "Pairing successful", pexpect.EOF]
            )
            success = True if res == 1 else False
            return success

    def remove(self, mac_address):
        """Remove paired device by mac address, return success of the operation."""
        try:
            out = self.get_output("remove " + mac_address, 3)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            res = self.child.expect(
                ["not available", "Device has been removed", pexpect.EOF]
            )
            success = True if res == 1 else False
            return success

    def connect(self, mac_address):
        """Try to connect to a device by mac address."""
        try:
            out = self.get_output("connect " + mac_address, 2)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            res = self.child.expect(
                ["Failed to connect", "Connection successful", pexpect.EOF]
            )
            success = True if res == 1 else False
            return success

    def disconnect(self, mac_address):
        """Try to disconnect to a device by mac address."""
        try:
            out = self.get_output("disconnect " + mac_address, 2)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            res = self.child.expect(
                ["Failed to disconnect", "Successful disconnected", pexpect.EOF]
            )
            success = True if res == 1 else False
            return success

    def menu_gatt(self):
        """Menu Gatt"""
        try:
            out = self.get_output("menu gatt")
        except BluetoothctlError as e:
            print(e)
            return None

    def selec_characteristic(self, char):
        """Select characteristic"""
        try:
            out = self.get_output("select-attribute asdasasasdasdasdasdasd" + char)
        except BluetoothctlError as e:
            print(e)
            return None

    def acquire_write(self):
        """acquire write"""
        try:
            out = self.get_output("acquire-write")
        except BluetoothctlError as e:
            print(e)
            return None

    def acquire_notify(self):
        """acquire write"""
        try:
            out = self.get_output("acquire-notify")
        except BluetoothctlError as e:
            print(e)
            return None

    def write(self, stri):
        """write"""
        try:
            out = self.get_output("write " + '"' + stri + '"')
        except BluetoothctlError as e:
            print(e)
            return None


if __name__ == "__main__":

    print("Init bluetooth...")
    bl = Bluetoothctl()
    print("Ready!")
    bl.start_scan()
    print("Scanning for 10 seconds...")
    for i in range(0, 10):
        print(i)
        time.sleep(1)

    print(bl.get_discoverable_devices())
    bl.connect("D2")
    bl.menu_gatt()
    bl.selec_characteristic("c")
    bl.acquire_write()
    for j in range(20):
        start_time = time.time()
        bl.write(
            "0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77 0x77"
        )
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"The function took {elapsed_time} seconds to execute.")
