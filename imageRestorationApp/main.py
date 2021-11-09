# ---------------------------------------------------------------#
# __name__ = "ImageRestoration"
# __author__ = "Chung Duc Nguyen Dang"
# __version__ = "1.0"
# __email__ = "duc.ndc172484@sis.hust.edu.vn"
# __status__ = "Development"
# ---------------------------------------------------------------#

# main.py contains the code for initializing and running the code for GUI
import sys
# PyQt5 libraries are used for GUI
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QPushButton, QFileDialog, QDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt                                                                                              

# OpenCV2 library is used for reading/ writing of images
import cv2
# All array operations are performed using numpy library
import numpy as np

# The GUI structure definition is provided in gui.py
from gui import *
# Image restoration logic is defined in imageRestorationFns.py
import imageRestorationFns as ir


# class ImageEditorClass implements the GUI main window class
class ImageRestorationClass(QtWidgets.QMainWindow):

    # stores a copy of original image for use in Undo All functionality
    originalImage = [0]
    # stores the current image being displayed/ processed
    currentImage = [0]
    # stores the ground truth image for psnr/ ssim calculations
    trueImage = [0]

    # stores current image height and width
    imageWidth = 0
    imageHeight = 0

    # GUI initialization
    def __init__(self, parent=None):
        super(ImageRestorationClass, self).__init__()
        QWidget.__init__(self, parent)
        self.ui = ImageRestorationGuiClass()
        self.ui.setupUi(self)

        # Assigning functions to be called on all button clicked events and
        # combo box change events
        self.ui.buttonOpen.clicked.connect(lambda: self.open_image())
        self.ui.buttonSave.clicked.connect(lambda: self.save_image())
        self.ui.buttonFullInv.clicked.connect(lambda: self.call_full_inverse())
        self.ui.buttonInv.clicked.connect(lambda: self.call_truncated_inverse_filter())
        self.ui.buttonWeiner.clicked.connect(lambda: self.call_weiner_filter())
        self.ui.buttonCLS.clicked.connect(lambda: self.call_constrained_ls_filter())

        self.ui.comboBoxKernel.currentIndexChanged.connect(lambda: self.displayKernel())

        # disable all buttons initially, except open image button
        self.disableAll()

    # calls the full inverse function
    def call_full_inverse(self):
        if not np.array_equal(self.originalImage, np.array([0])):
            # read the selected blur kernel
            blur_kernel = self.get_blur_kernel()
            self.currentImage = ir.full_inverse_filter(self.originalImage, blur_kernel)
            self.displayOutputImage()

            # compute psnr and ssim for output if true image is available
            if not np.array_equal(self.trueImage, np.array([0])):
                self.calculate_psnr()
                self.calculate_ssim()

    # calls the truncated inverse function
    def call_truncated_inverse_filter(self):
        self.ui.input_radius.setStyleSheet("background-color: white;")
        if not np.array_equal(self.originalImage, np.array([0])):
            # read the selected blur kernel
            blur_kernel = self.get_blur_kernel()
            # read the blur kernel radius from line edit input object
            R = self.ui.input_radius.text()
            if R and float(R) > 0:
                radius = float(R)
                self.currentImage = ir.truncated_inverse_filter(self.originalImage, blur_kernel, radius)
                self.displayOutputImage()

                # compute psnr and ssim for output if true image is available
                if not np.array_equal(self.trueImage, np.array([0])):
                    self.calculate_psnr()
                    self.calculate_ssim()

            else:
                self.ui.input_radius.setStyleSheet("background-color: red;")

    # calls the weiner filter function
    def call_weiner_filter(self):
        self.ui.input_K.setStyleSheet("background-color: white;")
        if not np.array_equal(self.originalImage, np.array([0])):
            # read the selected blur kernel
            blur_kernel = self.get_blur_kernel()
            # read the K value from line edit input object
            K_str = self.ui.input_K.text()
            if K_str:
                K = float(K_str)
                self.currentImage = ir.weiner_filter(self.originalImage, blur_kernel, K)
                self.displayOutputImage()

                # compute psnr and ssim for output if true image is available
                if not np.array_equal(self.trueImage, np.array([0])):
                    self.calculate_psnr()
                    self.calculate_ssim()

            else:
                self.ui.input_K.setStyleSheet("background-color: red;")

    # calls the constrained ls function
    def call_constrained_ls_filter(self):
        self.ui.input_gamma.setStyleSheet("background-color: white;")
        if not np.array_equal(self.originalImage, np.array([0])):
            # read the selected blur kernel
            blur_kernel = self.get_blur_kernel()
            # read the gamma value from line edit input object
            Y = self.ui.input_gamma.text()
            if Y:
                gamma = float(Y)
                self.currentImage = ir.constrained_ls_filter(self.originalImage, blur_kernel, gamma)
                self.displayOutputImage()

                # compute psnr and ssim for output if true image is available
                if not np.array_equal(self.trueImage, np.array([0])):
                    self.calculate_psnr()
                    self.calculate_ssim()
            else:
                self.ui.input_gamma.setStyleSheet("background-color: red;")

    # read the selected blur kernel from kernels folder
    def get_blur_kernel(self):
        index = self.ui.comboBoxKernel.currentIndex()
        kernel_filename = 'kernels/' + str(index + 1) + '.bmp'
        kernel = np.array(cv2.imread(kernel_filename, 0))
        return kernel

    # called when Open button is clicked
    def open_image(self):
        # open a new Open Image dialog box and capture path of file selected
        open_image_window = QFileDialog()
        image_path, _ = QFileDialog.getOpenFileName\
            (open_image_window, 'Open Image', '/')

        # check if image path is not null or empty
        if image_path:
            # initialize class variables
            self.currentImage = [0]
            self.trueImage = [0]

            # read image at selected path to a numpy ndarray object as color image
            self.currentImage = cv2.imread(image_path, 1)

            # set image specific class variables based on current image
            self.imageWidth = self.currentImage.shape[1]
            self.imageHeight = self.currentImage.shape[0]

            self.originalImage = self.currentImage.copy()

            # displayInputImage converts original image from ndarry format to
            # pixmap and assigns it to image display label
            self.displayInputImage()

            self.ui.labelOut.clear()

            # Enable all buttons and sliders
            self.enableAll()

    # called when Save button is clicked
    def save_image(self):
        # configure the save image dialog box to use .jpg extension for image if
        # not provided in file name
        dialog = QFileDialog()
        dialog.setDefaultSuffix('jpg')
        dialog.setAcceptMode(QFileDialog.AcceptSave)

        # open the save dialog box and wait until user clicks 'Save'
        # button in the dialog box
        if dialog.exec_() == QDialog.Accepted:
            # select the first path in the selected files list as image save
            # location
            save_image_filename = dialog.selectedFiles()[0]
            # write current image to the file path selected by user
            cv2.imwrite(save_image_filename, self.currentImage)

    # displayInputImage converts original image from ndarry format to pixmap and
    # assigns it to input image display label
    def displayInputImage(self):
        # set display size to size of the image display label
        display_size = self.ui.labelIn.size()
        # copy original image to temporary variable for processing pixmap
        image = np.array(self.originalImage.copy())
        zero = np.array([0])

        # display image if image is not [0] array
        if not np.array_equal(image, zero):
            # convert BGR image to RGB format for display in label
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # ndarray cannot be directly converted to QPixmap format required
            # by image display label
            # so ndarray is first converted to QImage and then QImage to QPixmap
            # convert image ndarray to QImage format
            qImage = QImage(image, self.imageWidth, self.imageHeight,
                            self.imageWidth * 3, QImage.Format_RGB888)

            # convert QImage to QPixmap for loading in image display label
            pixmap = QPixmap()
            QPixmap.convertFromImage(pixmap, qImage)
            pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
            # set pixmap to image display label in GUI
            self.ui.labelIn.setPixmap(pixmap)

    # displayOutputImage converts current image from ndarry format to pixmap and
    # assigns it to output image display label
    def displayOutputImage(self):
        # set display size to size of the image display label
        display_size = self.ui.labelOut.size()
        # copy current image to temporary variable for processing pixmap
        image = np.array(self.currentImage.copy())
        zero = np.array([0])

        # display image if image is not [0] array
        if not np.array_equal(image, zero):
            # convert BGR image to RGB format for display in label
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # ndarray cannot be directly converted to QPixmap format required
            # by image display label
            # so ndarray is first converted to QImage and then QImage to QPixmap
            # convert image ndarray to QImage format
            qImage = QImage(image, self.imageWidth, self.imageHeight,
                            self.imageWidth * 3, QImage.Format_RGB888)

            # convert QImage to QPixmap for loading in image display label
            pixmap = QPixmap()
            QPixmap.convertFromImage(pixmap, qImage)
            pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
            # set pixmap to image display label in GUI
            self.ui.labelOut.setPixmap(pixmap)

    # displayKernel converts selected kernel image from ndarry format to pixmap and
    # assigns it to kernel display label
    def displayKernel(self):
        # set display size to size of the kernel display label
        display_size = self.ui.labelKernelDisplay.size()
        # copy kernel image to temporary variable for processing pixmap
        kernel = np.array(self.get_blur_kernel())
        zero = np.array([0])

        # display image if kernel is not [0] array
        if not np.array_equal(kernel, zero):
            # ndarray cannot be directly converted to QPixmap format required
            # by kernel display label
            # so ndarray is first converted to QImage and then QImage to QPixmap

            # convert kernel ndarray to QImage format
            qImage = QImage(kernel, kernel.shape[1], kernel.shape[0], kernel.shape[1], QImage.Format_Indexed8)

            # convert QImage to QPixmap for loading in image display label
            pixmap = QPixmap()
            QPixmap.convertFromImage(pixmap, qImage)
            pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
            # set pixmap to kernel display label in GUI
            self.ui.labelKernelDisplay.setPixmap(pixmap)

    # Function to enable all buttons and sliders
    def enableAll(self):
        self.ui.buttonSave.setEnabled(True)
        self.ui.buttonFullInv.setEnabled(True)
        self.ui.buttonInv.setEnabled(True)
        self.ui.buttonWeiner.setEnabled(True)
        self.ui.buttonCLS.setEnabled(True)

        self.ui.comboBoxKernel.setEnabled(True)
        self.displayKernel()

        self.ui.input_radius.setEnabled(True)
        self.ui.input_K.setEnabled(True)
        self.ui.input_gamma.setEnabled(True)

        self.ui.input_radius.clear()
        self.ui.input_K.clear()
        self.ui.input_gamma.clear()

    # Function to disable all buttons and sliders
    def disableAll(self):
        self.ui.buttonSave.setEnabled(False)
        self.ui.buttonFullInv.setEnabled(False)
        self.ui.buttonInv.setEnabled(False)
        self.ui.buttonWeiner.setEnabled(False)
        self.ui.buttonCLS.setEnabled(False)
        self.ui.comboBoxKernel.setEnabled(False)

        self.ui.input_radius.setEnabled(False)
        self.ui.input_K.setEnabled(False)
        self.ui.input_gamma.setEnabled(False)

        self.ui.input_radius.clear()
        self.ui.input_K.clear()
        self.ui.input_gamma.clear()


# initialize the ImageEditorClass and run the application
if __name__ == "__main__":
    app = QApplication([])
    myapp = ImageRestorationClass()
    myapp.showMaximized()
    sys.exit(app.exec_())

