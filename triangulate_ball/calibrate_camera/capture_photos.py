import os
import requests
import keyboard
import time

class CameraCapture:
    def __init__(self, camera1_ip, camera2_ip, camera1_folder, camera2_folder):
        self.camera1_ip = camera1_ip
        self.camera2_ip = camera2_ip
        self.camera1_folder = camera1_folder
        self.camera2_folder = camera2_folder
        self.photo_number = 1

    def capture_photo(self, ip_address, folder_name, photo_number):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        url = f"http://{ip_address}:8080/photo.jpg"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                file_name = os.path.join(folder_name, f"photo{photo_number}.jpg")
                with open(file_name, 'wb') as file:
                    file.write(response.content)
                print(f"Photo saved to {file_name}")
            else:
                print(f"Failed to capture photo from {ip_address}")
        except Exception as e:
            print(f"Error capturing photo from {ip_address}: {e}")

    def main(self):
        print("Press 'p' to capture photos from both cameras. Press 'q' to quit.")
        while True:
            if keyboard.is_pressed('p'):
                print(f"Capturing photo {self.photo_number} from both cameras...")
                self.capture_photo(self.camera1_ip, self.camera1_folder, self.photo_number)
                self.capture_photo(self.camera2_ip, self.camera2_folder, self.photo_number)
                self.photo_number += 1
                time.sleep(1)
            if keyboard.is_pressed('q'):
                print("Exiting program...")
                break

if __name__ == "__main__":
    camera_capture = CameraCapture('192.168.0.11', '192.168.0.11', 'camera1', 'camera2')
    camera_capture.main()