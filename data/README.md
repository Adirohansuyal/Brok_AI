# Image Steganography using OpenCV

## 📌 Overview
This project hides a secret message inside an image by modifying pixel values and then retrieves the hidden message during decryption. It uses OpenCV to read and modify image pixels, storing message characters as pixel values.

## 🚀 Features
- Encode a secret message into an image.
- Retrieve the hidden message from the image.
- Requires a passcode for decryption.
- Uses OpenCV for image processing.
- Works on Windows, macOS, and Linux.

## 🛠️ Technologies Used
- Python
- OpenCV
- NumPy
- OS Module

## 📂 Project Structure
```
📦 Image-Steganography
 ┣ 📜 datahiding.py   # Main Python script
 ┣ 🖼️ adi.jpeg        # Input image (replace with your own)
 ┣ 🖼️ encryptedImage.jpg  # Output image with hidden message
 ┗ 📜 README.md       # Documentation
```

## 🔧 Installation & Setup
### 1️⃣ Prerequisites
Ensure you have **Python 3** installed and the required dependencies:
```sh
pip install opencv-python numpy
```

### 2️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/Image-Steganography.git
cd Image-Steganography
```

### 3️⃣ Run the Script
```sh
python datahiding.py
```

## 📌 How It Works
### **1️⃣ Encoding (Hiding a Message)**
1. The script loads an image (`adi.jpeg`).
2. The user enters a **secret message** and **passcode**.
3. The message is converted to ASCII values and stored in pixel values.
4. The modified image is saved as `encryptedImage.jpg`.

### **2️⃣ Decoding (Retrieving the Message)**
1. The script asks for a **passcode**.
2. If correct, it extracts ASCII values from the modified image pixels.
3. Converts ASCII values back to characters to reveal the **secret message**.
4. If the passcode is incorrect, access is denied.

## 🔍 Example Usage
```sh
Enter secret message: HelloWorld
Enter a passcode: 1234
```
- Saves `encryptedImage.jpg` with hidden text.

```sh
Enter passcode for decryption: 1234
Decryption message: HelloWorld
```

## ⚠️ Limitations
- Only small text messages can be hidden (depends on image size).
- Not a secure encryption method (basic pixel modification).

## 📌 Future Improvements
✅ Implement **LSB Steganography** for better security.  
✅ Add **encryption** before hiding the message.  
✅ Develop a GUI for easy usage.

## 📜 License
This project is **open-source** and free to use.

---
Made with ❤️ by Nisha Thakur

