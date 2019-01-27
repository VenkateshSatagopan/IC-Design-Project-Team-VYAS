import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout,QFileDialog,QLabel,QLineEdit
from PyQt5.QtGui import QIcon, QPixmap,QImage,QPalette,QBrush
from PyQt5.QtCore import pyqtSlot,QSize,Qt
from Build_dataset import *
from keras.models import load_model
 
class App(QWidget):
 
  def __init__(self):
     super().__init__()
     self.title = 'Automatic Pneumonia detection'
     self.left = 100
     self.top = 100
     self.width = 3000
     self.height = 3000
     #self.file=None
     #self.filename=None
     self.setWindowTitle( self.title )
     self.setGeometry( self.left, self.top, self.width, self.height )
     self.layout = QVBoxLayout()
     self.setAutoFillBackground( True )
     p = self.palette()
     p.setColor(self.backgroundRole(), Qt.white )
     self.setPalette( p )

    # oImage = QImage( "/home/venkatesh/Desktop/QT-tutorial/Tum.jpg" )
     #sImage = oImage.scaled(QSize(1250,1500))  # resize Image to widgets size
     #palette = QPalette()
     #palette.setBrush(10, QBrush( sImage ) )  # 10 = Windowrole
     #self.setPalette( palette )

     #self.label = QLabel( 'Test', self )  # test, if it's really backgroundimage
     #self.label.setGeometry( 50, 50, 200, 50 )
     #self.show()
     self.initUI()

  def changebackground( self ):
      fname = QFileDialog.getOpenFileName( self, 'Select background image', '/home/venkatesh/Desktop/QT-tutorial/Tum.jpg' )
      print(fname)
      self.results.setStyleSheet(
          "background-image: url(fname); background-repeat: no-repeat; background-position: center;" )


  def initUI(self):


    #container=QWidget(self)
    #fname = QFileDialog.getOpenFileName( self, 'Select background image',
    #                                     '/home/venkatesh/Desktop/QT-tutorial/Tum.jpg' )
    #container.setStyleSheet("background-image: url(fname); background-repeat: no-repeat; background-position: center;")

    self.button = QPushButton('Select Model file and Image file',self)
    self.button_1 = QPushButton('Clear',self)
    self.textbox = QLineEdit(self)
    self.textbox1=QLineEdit(self)
    self.textbox.move(200,400)
    self.textbox1.move(200,500)
    self.textbox1.resize(280,40)
    self.textbox.resize(280, 40)
    self.layout.addWidget(self.button)
    self.layout.addWidget(self.button_1)
    self.button.move( 200, 70 )
    self.button_1.move(200,600)

    self.label = QLabel(self)
    self.label.move(200,150)

    #self.label.resize(512,512)
    self.layout.addWidget(self.label)
    #self.textbox_cal = QLineEdit(self)
    #self.textbox_cal.move(200,200)
    #self.textbox_cal.resize(280, 40)
    #self.textbox_cal.hide()
    self.button.clicked.connect(self.openFileNameDialog)

    #self.textbox_cal.hide()
    #button.clicked.connect()
    #self.button.setText("Finished")
    #self.show()
    self.button_1.clicked.connect(self.clear_all)
    self.show()
    #self.show()

  @pyqtSlot()
  def openFileNameDialog(self):
      options = QFileDialog.Options()
      options |= QFileDialog.DontUseNativeDialog
      fileName, _ = QFileDialog.getOpenFileName( QFileDialog(), "QFileDialog.getOpenFileName()", "",
                                                 "All Files (*);;Python Files (*.py)", options=options)
      fileName_1, _ = QFileDialog.getOpenFileName( QFileDialog(), "QFileDialog.getOpenFileName()", "",
                                                 "All Files (*);;Image file(*.jpg)", options=options )

      #self.layout.addWidget(self.textbox_cal)
      #self.textbox_cal.setText("Calculating")
      #self.textbox_cal.show()


      #self.button.setText("Calculating")
      pixmap = QPixmap(fileName_1)
      #myScaledPixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio )
      self.label.setPixmap(pixmap)

      self.label.setScaledContents(True)
      self.label.resize(224,224)
      self.label.show()
      #self.button.clicked.connect( self.calculate )
      predicted_value,Actual_value=run_machine_learning_model(fileName,fileName_1)

      if not predicted_value:
          value='Predicted value is Normal'
      else:
          value='Predicted value is Pneumonia'
      if not Actual_value:
          value_1='Actual value is Normal'
      else:
          value_1='Actual value is Pneumonia'
      #textboxValue = self.textbox.text()
      #textboxValue_1=self.textbox1.text()
      #QMessageBox.question( self, 'The value predicted is ' + value, QMessageBox.Ok,
      #                      QMessageBox.Ok )
      self.textbox.setText( value)
      self.textbox1.setText(value_1)
      self.textbox.show()
      self.textbox1.show()



  def openImageFileNameDialog(self):
          options = QFileDialog.Options()
          options |= QFileDialog.DontUseNativeDialog
          fileName, _ = QFileDialog.getOpenFileName( QFileDialog(), "QFileDialog.getOpenFileName()", "",
                                                          "All Files (*);;Image file(*.jpg)", options=options )
          #label = QLabel(self)
          #pixmap = QPixmap(fileName)
          #label.setPixmap(pixmap)
          #self.resize(pixmap.width(), pixmap.height() )
          #self.show()
          print(fileName)

          return fileName

  def clear_all( self):
      self.textbox.clear()
      self.textbox1.clear()
      self.label.clear()

  def calculate( self ):
      self.textbox_cal.setText("Calculating")
      self.textbox_cal.show()

 
if __name__ == "__main__":
 app = QApplication(sys.argv)
 ex = App()
 file_1=ex.initUI()
 print(file_1)
#button = QPushButton('Browse_files_new',QPushButton)
#file=button.clicked.connect(ex.openFileNameDialog)
#print(file)
#file=but.clicked.connect(ex.openFileNameDialog)
#but.show()
 sys.exit(app.exec_())
