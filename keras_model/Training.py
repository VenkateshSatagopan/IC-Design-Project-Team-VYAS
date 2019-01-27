from Build_dataset import *

# Getting Train and Test data

parser = argparse.ArgumentParser()
parser.add_argument("device_name",help="Enter gpu or cpu to run the program with gpu or cpu",type=str)
args = parser.parse_args()
if args.device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

TRAIN_DIR = "Input/chest_xray/train/"
TEST_DIR =  "Input/chest_xray/test/"

X_train, y_train = get_data(TRAIN_DIR)
X_test , y_test = get_data_test(TEST_DIR)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

print(y_train.shape,'\n',y_test.shape)

# Visulaization of Images

Pimages = os.listdir(TRAIN_DIR + "PNEUMONIA")
Nimages = os.listdir(TRAIN_DIR + "NORMAL")

def plotter(i):
    imagep1 = cv2.imread(TRAIN_DIR+"PNEUMONIA/"+Pimages[i])
    imagep1 = skimage.transform.resize(imagep1, (112, 112, 3) , mode = 'reflect')
    imagen1 = cv2.imread(TRAIN_DIR+"NORMAL/"+Nimages[i])
    imagen1 = skimage.transform.resize(imagen1, (112, 112, 3))
    pair = np.concatenate((imagen1, imagep1), axis=1)
    print("(Left) - No Pneumonia Vs (Right) - Pneumonia")
    print("-----------------------------------------------------------------------------------------------------------------------------------")
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    plt.show()

'''for i in range(0,5):
    plotter(i)'''


#seaborn_plot(y_train)
#X_train =image_generate(X_train)

image_gen_2 = ImageDataGenerator(  # featurewise_center=True,
    # featurewise_std_normalization=True)

    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest" )

image_gen_2.fit(X_train)

filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


X_train=X_train.reshape(7750,112,112,3)
X_test=X_test.reshape(624,112,112,3)

INIT_LR=06e-05
#5.76480058953166e-05
NUM_EPOCHS = 30
epochs=30
batch_size=64
reg=0.00025

def poly_decay(epoch):
# initialize the maximum number of epochs, base learning rate,
# and power of the polynomial
   maxEpochs = NUM_EPOCHS
   baseLR = INIT_LR
   power = 1.0
# compute the new learning rate based on polynomial decay
   alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

# return the new learning rate
   return alpha


def ResNet( input_shape=(150, 150, 3), classes=2 ):
    # Define the input as a tensor with shape input_shape
    X_input = Input( input_shape )

    # Stage 1
    X = Conv2D(32, (1, 5),kernel_regularizer=regularizers.l2(reg),padding="same",data_format='channels_last')(X_input)
    X = Conv2D(32, (5, 1),kernel_regularizer=regularizers.l2(reg),padding="same",data_format='channels_last')(X)
    #X = Conv2D(32, (1, 3),kernel_regularizer=regularizers.l2(reg),padding="same",data_format='channels_last')(X_input)
    #X = Conv2D(32, (3, 1),kernel_regularizer=regularizers.l2(reg),padding="same",data_format='channels_last')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(32, (1, 5),kernel_regularizer=regularizers.l2(reg),padding="same",data_format='channels_last')(X)
    X = Conv2D(32, (5, 1),kernel_regularizer=regularizers.l2(reg),padding="same",data_format='channels_last')(X)
    #X = Conv2D(32, (1, 3),kernel_regularizer=regularizers.l2(reg),padding="same",data_format='channels_last')(X)
    #X = Conv2D(32, (3, 1),kernel_regularizer=regularizers.l2(reg),padding="same",data_format='channels_last')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    #X = Conv2D(32, (1, 3),kernel_regularizer=regularizers.l2(reg),padding="same",data_format='channels_last')(X)
    #X = Conv2D(32, (3, 1),kernel_regularizer=regularizers.l2(reg),padding="same",data_format='channels_last')(X)
    #X = BatchNormalization()(X)
    #X = Activation('relu')(X)
    '''X = Conv2D(16, (1, 3), kernel_regularizer=regularizers.l2( reg ), padding="same", data_format='channels_last' )(
        X)
    X = Conv2D(16, (3, 1), kernel_regularizer=regularizers.l2( reg ), padding="same", data_format='channels_last' )(
        X)
    X = BatchNormalization()( X )
    X = Activation( 'relu' )( X ) '''

    '''X = Conv2D(32, (1, 3), kernel_regularizer=regularizers.l2( reg ), padding="same",
                         data_format='channels_last' )( X )
    X = Conv2D( 32, (3, 1), kernel_regularizer=regularizers.l2( reg ), padding="same",
                data_format='channels_last' )( X )
    X = BatchNormalization()( X )
    X = Activation( 'relu' )( X )'''
    '''X = Conv2D(16, (3, 3),kernel_regularizer=regularizers.l2(reg),padding="same",data_format='channels_last')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)'''
    '''X = Conv2D(16, (3, 3),kernel_regularizer=regularizers.l2(reg), padding="same",data_format='channels_last')(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)'''
    '''X = Conv2D(32, (3, 3),kernel_regularizer=regularizers.l2(reg), padding="same",data_format='channels_last')(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(32, (3, 3),kernel_regularizer=regularizers.l2(reg), padding="same",data_format='channels_last')(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2),data_format='channels_last')(X)'''
    # X = Dropout(0.6)(X)
    X = MaxPooling2D( (2, 2), data_format='channels_last' )( X )

    # Stage 2
    X_shortcut = X
    X = SeparableConv2D(32, (3, 3), strides=(2, 2), kernel_regularizer=regularizers.l2( reg ), padding="same",
                         data_format='channels_last' )( X )
    X = BatchNormalization()( X )
    X = Activation('relu' )( X )
    X = SeparableConv2D( 32, (3, 3), kernel_regularizer=regularizers.l2( reg ), padding="same",
                         data_format='channels_last' )( X )
    X = BatchNormalization()( X )
    X = Activation( 'relu' )( X )
    X = SeparableConv2D( 64, (3, 3), kernel_regularizer=regularizers.l2( reg ), padding="same",
                         data_format='channels_last' )( X )
    X = BatchNormalization()( X )

    # shortcut
    X_shortcut = SeparableConv2D( filters=64, kernel_regularizer=regularizers.l2( reg ), kernel_size=(1, 1),
                                  strides=(2, 2), padding="same", kernel_initializer=glorot_uniform( seed=0 ),
                                  data_format='channels_last' )( X_shortcut )
    X_shortcut = BatchNormalization()( X_shortcut )
    X = Add()( [ X_shortcut, X ] )
    X = Activation( 'relu' )( X )
    # X = Dropout(0.6)(X)

    # Stage 3
    X_shortcut = X
    X = SeparableConv2D( 64, (3, 3), strides=(2, 2), kernel_regularizer=regularizers.l2( reg ), padding="same",
                         data_format='channels_last' )( X )
    X = BatchNormalization()( X )
    X = Activation( 'relu' )( X )
    X = SeparableConv2D( 64, (3, 3), kernel_regularizer=regularizers.l2( reg ), padding="same",
                         data_format='channels_last' )( X )
    X = BatchNormalization()( X )
    X = Activation( 'relu' )( X )
    X = SeparableConv2D( 128, (3, 3), kernel_regularizer=regularizers.l2( reg ), padding="same",
                         data_format='channels_last' )( X )
    X = BatchNormalization()( X )

    # shortcut
    X_shortcut = SeparableConv2D( filters=128, kernel_regularizer=regularizers.l2( reg ), kernel_size=(1, 1),
                                  strides=(2, 2), padding="same", kernel_initializer=glorot_uniform( seed=0 ),
                                  data_format='channels_last' )( X_shortcut )
    X_shortcut = BatchNormalization()( X_shortcut )
    X = Add()([ X_shortcut, X ] )
    X = Activation( 'relu' )( X )
    # X = Dropout(0.6)(X)

    # Stage 4
    X_shortcut = X
    X = SeparableConv2D( 128, (3, 3), strides=(2, 2), kernel_regularizer=regularizers.l2( reg ), padding="same",
                         data_format='channels_last' )( X )
    X = BatchNormalization()( X )
    X = Activation( 'relu' )( X )
    X = SeparableConv2D( 128, (3, 3), kernel_regularizer=regularizers.l2( reg ), padding="same",
                         data_format='channels_last' )( X )
    X = BatchNormalization()( X )
    X = Activation( 'relu' )( X )
    X = SeparableConv2D( 256, (3, 3), kernel_regularizer=regularizers.l2( reg ), padding="same",
                         data_format='channels_last' )( X )
    X = BatchNormalization()( X )

    # shortcut
    X_shortcut = SeparableConv2D( filters=256, kernel_regularizer=regularizers.l2( reg ), kernel_size=(1, 1),
                                  strides=(2, 2), padding="same", kernel_initializer=glorot_uniform( seed=0 ),
                                  data_format='channels_last' )( X_shortcut )
    X_shortcut = BatchNormalization()( X_shortcut )
    X = Add()( [ X_shortcut, X ] )
    X = Activation( 'relu' )( X )

    # X = Dropout(0.6)(X)

    X = AveragePooling2D( (2, 2), data_format='channels_last' )( X )

    X = Flatten()( X )
    X = Dense( classes, activation='softmax', kernel_initializer=glorot_uniform( seed=0 ),
               kernel_regularizer=regularizers.l2( reg ) )( X )

    model = Model( inputs=X_input, outputs=X, name='ResNet' )

    return model


model = ResNet(input_shape = (112, 112, 3), classes = 2)
model.compile(optimizer=Adam(lr=INIT_LR), loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


H = model.fit_generator(image_gen_2.flow(X_train, y_train, batch_size=batch_size),validation_data = (X_test ,y_test) ,callbacks=[checkpoint,LearningRateScheduler(poly_decay)],
steps_per_epoch=len(X_train) / batch_size, validation_steps = len(X_test)/batch_size ,epochs=epochs,shuffle=True)

#Plotting

plt.plot(H.history['acc'])
plt.plot(H.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



#Confusion Matrix
pred = model.predict(X_test)
pred = np.argmax(pred,axis = 1)
y_true = np.argmax(y_test,axis = 1)

CM = confusion_matrix(y_true, pred)
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))
plt.show()



