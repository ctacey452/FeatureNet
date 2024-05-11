import shutil
import os
import trimesh
import numpy as np
import tensorflow as tf


def read_cifar10(data_dir, is_train, batch_size, shuffle):
    img_width = 64
    img_height = 64
    img_depth = 64
    label_bytes = 1
    image_bytes = img_width * img_height * img_depth

    filenames = []
    if is_train:
        for ii in np.arange(0, 24000):
            filenames.append(os.path.join(data_dir, 'train{}.bin'.format(ii)))
    else:
        for jj in np.arange(0, 1):
            filenames.append(os.path.join(data_dir, 'test{}.bin'.format(jj)))

    dataset = tf.data.FixedLengthRecordDataset(filenames, label_bytes + image_bytes)
    
    def parse_record(record):
        record_bytes = tf.io.decode_raw(record, tf.uint8)

        label = tf.slice(record_bytes, [0], [label_bytes])
        label = tf.cast(label, tf.int32)

        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
        image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width, 1])
        image = tf.transpose(image_raw, (1, 2, 0, 3))
        image = tf.cast(image, tf.float32)
        image = (image - 0.5) * 2
        
        return image, label

    dataset = dataset.map(parse_record)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filenames))

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    n_classes = 24
    def one_hot(label):
        return tf.one_hot(label, depth=n_classes)
    
    dataset = dataset.map(lambda image, label: (image, tf.reshape(one_hot(label), (batch_size, n_classes))))
    return dataset

def stl_to_voxels(input_file, output_file, type, index, symbol):
    # Read the STL file
    mesh = trimesh.load_mesh(input_file)

    # Determine the dimensions of the bounding box
    bounding_box = mesh.bounding_box_oriented.bounds

    # Calculate the size of the voxel
    mesh_size = np.max(bounding_box, axis=0) - np.min(bounding_box, axis=0)
    target_voxel_dimensions = (63, 63, 63)
    voxel_size = mesh_size / target_voxel_dimensions

    # Create a binary voxel grid from the mesh
    # Voxelization
    voxelized = mesh.voxelized(voxel_size)

    # If need to check
    #mesh1 = voxelized.marching_cubes
    # Сохраняем объект Trimesh в формате STL
    #mesh1.export('output.stl', file_type='stl')

    # Convert the voxel grid to raw binary format
    raw_voxel_data = voxelized.matrix.astype(np.uint8).tobytes()

    # Save the raw binary voxel data to a file
    with open(output_file, 'wb') as file:
        file.write(symbol.encode('utf-8'))
        file.write(raw_voxel_data)

    directory = os.path.dirname(output_file)
    short_filename = type + '{}.bin'.format(index)
    new_output_path = os.path.join(directory, short_filename)

    shutil.copyfile(output_file, new_output_path)

def process_directory (input_directory, output_directory, type = 'train'):
    # Пройдемся по всем файлам во входной директории и поддиректориях
    index = 0
    for root, dirs, files in os.walk(input_directory):
        for file in files:
           # Формируем полное имя входного файла
            input_file_path = os.path.join(root, file)

            # Выполняем преобразование имени файла
            # Например, можно взять имя файла и добавить к нему некий суффикс
            new_file_name = file + '.binvox'

            # Формируем полное имя выходного файла, используя новое имя
            output_file_path = os.path.join(output_directory, new_file_name)
            if (type == 'train'):
                symbol = chr (int (new_file_name.split('_')[0]))
            else:
                symbol = 0
            stl_to_voxels (input_file_path, output_file_path, type, index, symbol)
            index = index + 1

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same',
                                            activation='relu', kernel_initializer='truncated_normal')
        self.batch_norm1 = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding='same',
                                            activation='relu', kernel_initializer='truncated_normal')

        self.conv3 = tf.keras.layers.Conv3D(filters=64, kernel_size=(4, 4, 4), strides=(1, 1, 1), padding='same',
                                            activation='relu', kernel_initializer='truncated_normal')

        self.conv4 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                            activation='relu', kernel_initializer='truncated_normal')

        self.pool2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='truncated_normal')

        self.softmax_linear = tf.keras.layers.Dense(units=24, kernel_initializer='truncated_normal')

    def call(self, inputs):
        conv1_out = self.conv1(inputs)
        batch_norm1_out = self.batch_norm1(conv1_out)
        conv2_out = self.conv2(batch_norm1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        pool2_out = self.pool2(conv4_out)
        flatten_out = self.flatten(pool2_out)
        dense1_out = self.dense1(flatten_out)
        softmax_linear_out = self.softmax_linear(dense1_out)
        return softmax_linear_out


def losses(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)
    return loss

def evaluation(logits, labels):
    labels = tf.argmax(labels, axis=1)  # Convert one-hot encoded labels to class indices
    predictions = tf.argmax(logits, axis=1)  # Convert logits to class indices
    correct = tf.equal(predictions, labels)  # Check if predictions match labels
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # Calculate accuracy
    return accuracy

def train(data_dir, log_dir):
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    BATCH_SIZE = 40  # Adjust batch size as needed
    MAX_STEP = 20000  # Adjust max step as needed
    learning_rate = 0.001  # Adjust learning rate as needed

    train_dataset = read_cifar10(data_dir=data_dir, is_train=True, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model = Model()
    # Define the saver object to save checkpoints
    saver = tf.train.Checkpoint(optimizer=optimizer, model=model, optimizer_step=my_global_step)

    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images)
            loss_value = losses(logits, labels)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy = evaluation(logits, labels)

        return loss_value, train_accuracy
    
    #for images, labels in train_dataset.take(MAX_STEP):
    for step, (images, labels) in enumerate(train_dataset.take(MAX_STEP)):
        loss_value, train_accuracy = train_step(images, labels)

        if step % 50 == 0:
            print('Step: %d, loss: %.4f,train accuracy = %.2f%%' %(step, loss_value,train_accuracy * 100.0))

        if step % 2000 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(log_dir, 'model_%s.ckpt'%step)
            saver.save(checkpoint_path)

        tf.print("Step:", step, "Loss:", loss_value)

def result(log_dir, test_dir):
        n_test = 1
        # reading test data
        test_dataset = read_cifar10(data_dir=test_dir,
                                      is_train=False,
                                      batch_size=1,
                                      shuffle=False)
    
        
        names = ["oring", "through_hole", "blind_hole", "triangular_passage",
                 "rectangular_passage", "circular_through_hole", "triangular_through_hole", "rectangular_through_hole",
                 "rectangular_blind_spot", "triangular_pocket", "rectangular_pocket", "circular_end_pocket",
                 "triangular_blind_step", "circular_blind_step", "rectangular_blind_step", "rectangular_through_step",
                 "2sides_through_step", "slanted_through_step", "chamfer", "round",
                 "v_circular_end_blind_slot", "h_circular_end_blind_slot", "6sides_passage", "6sides_pocket"]

        model = Model()
        checkpoint = tf.train.Checkpoint(model=model)
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint.restore(tf.train.latest_checkpoint(log_dir))
            print('Loading success')
        else:
            print('No checkpoint file found')
            return

        for step, (images, labels) in enumerate(test_dataset.take(1)):

            logits = model(images)
            numclassesinfo = 5

            # Get top k predictions
            predict_label = tf.math.top_k(logits, k=numclassesinfo).indices.numpy()
            true_label = labels.numpy()

            # Compute softmax probabilities
            softmax_logits = tf.nn.softmax(logits)
            softmax_values = softmax_logits.numpy()

            # Print predictions
        for value in predict_label[0]:
            print(names[value] + ': probability = %.3f' % softmax_values[0][value])


result('logn', 'test')
#train ('d:\\6\\dataset\\bin', 'd:\\6\\dataset\\logn')
#read_cifar10('bin', True, 40, True)
#process_directory ('d:\\6\\dataset\\test\\in', 'd:\\6\\dataset\\test\\out', 'test')