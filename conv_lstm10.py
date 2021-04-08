"""
#This script demonstrates the use of a convolutional LSTM network.

This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling3D
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import model_from_json 

n_samples =12000
n_frames = 10
# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                  input_shape=(None, 60, 48, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
#seq.add(MaxPooling3D(pool_size = (1,2,2),padding = 'same'))

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
#seq.add(MaxPooling3D(pool_size = (1,2,2),padding = 'same'))

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
#seq.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same'))

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
#seq.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same'))

seq.add(Conv3D(filters=3, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')


# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.

def generate_movies(n_samples, n_frames):
    import os
    from PIL import Image
    import numpy as np
    folder = "/incois_ncmrwfx/incois/hycom/ARVIND_SINGH/HYCOM_TS/CONVLSTM/sst.day.mean.1981-2019.Figures/"
    images = sorted(os.listdir(folder))
    input_movie=[]
    label_movie=[]
    movie_array_input = []
    movie_array_label = []
    for sample in range(n_samples):
        for frame in range(sample,sample+n_frames):
            # reading images for input data and labels
            im_i = Image.open(folder + images[sample])
            im_l = Image.open(folder + images[sample+1])
            
            #im_i = im_i.getdata()
            #im_l = im_l.getdata()
            # converting images into nparray and append
            movie_array_input.append(np.array(im_i)) #.transpose(1, 0, 2))
            movie_array_label.append(np.array(im_l))
            # converting list of image array to array 
        movie_array_input = np.array(movie_array_input)
        movie_array_label = np.array(movie_array_label)

        input_movie.append(movie_array_input)
        label_movie.append(movie_array_label) 
        movie_array_input = []
        movie_array_label = []
    input_movie = np.array(input_movie)
    label_movie = np.array(label_movie)
    print(input_movie.shape)
    print(label_movie.shape)
    #(75, 50, 100, 3)
    return input_movie, label_movie

# Train the network
noisy_movies, shifted_movies = generate_movies(n_samples, n_frames)
seq.fit(noisy_movies[:int(0.8*n_samples)], shifted_movies[:int(0.8*n_samples)], batch_size=128,
        epochs=300, validation_split=0.33)

# save model
# serialize model to JSON
model_json = seq.to_json()
with open("seq.json", "w") as json_file:
     json_file.write(model_json)
# serialize weights to HDF5
seq.save_weights("model.h5")
print("Saved model to disk")
       
# later...
       
# load json and create model
# json_file = open('seq.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("seq.h5")
# print("Loaded model from disk")

# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = int(0.8*n_samples+10)
track = noisy_movies[which][:int(n_frames/2), ::, ::, ::]

for j in range(n_frames+1):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)


# And then compare the predictions
# to the ground truth
track2 = noisy_movies[which][::, ::, ::, ::]
for i in range(n_frames):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= int(n_frames/2):
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%i_sst_field.png' % (i + 1))
#Testing the network on test movie 
