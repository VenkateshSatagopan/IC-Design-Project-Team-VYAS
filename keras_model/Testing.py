from Build_dataset import *
from keras.models import load_model
from scipy import misc
def get_data_test_new( Dir ):
# Function to prepare test data images
    X = [ ]
    y = [ ]
    for nextDir in os.listdir( Dir ):
        if not nextDir.startswith( '.' ):
            if nextDir in [ 'NORMAL' ]:
                label = 0
            elif nextDir in [ 'PNEUMONIA' ]:
                label = 1
            else:
                label = 2

            temp = Dir + nextDir

            for file in tqdm( os.listdir( temp ) ):
                img = misc.imread( temp + '/' + file )
                if img is not None:
                    img = misc.imresize(img, (112, 112) )
                    if len(img.shape) == 2:
                        img = np.dstack( [ img, img, img ] )
                    img = np.asarray(img)
                    X.append( img )
                    y.append( label )

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y
TEST_DIR =  "Input/chest_xray/test/"
X_test , y_test = get_data_test_new(TEST_DIR)
model=load_model("weights.hdf5")
print(model.summary())
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[ 'accuracy' ])
a=model.predict(X_test)
y_classes = a.argmax( axis=-1)
print(y_classes)
print(y_test)
