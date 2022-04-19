import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import psutil

TRAIN_SIZE = 267838
VAL_SIZE = 2085
TEST_SIZE = 3653

def print_mem():
    total, available, percent = psutil.virtual_memory()[:3]
    giga = 1024 ** 3
    total /= giga
    available /= giga
    print('total: ' + str(total) + ' available: ' + str(available) + ' (' + str(percent) + '%)')

def read_npy_files_as_matrix(data_split):
    """
    Data must be in ./mimic_cxr/raw_embeddings
    """
    data_path = 'mimic_cxr/raw_embeddings/' + data_split + '/'
    if data_split == 'train':
        data_size = TRAIN_SIZE
        n_split = 17
    elif data_split == 'val':
        data_size = VAL_SIZE
        n_split = 1
    elif data_split == 'test':
        data_size = TEST_SIZE
        n_split = 1
    else:
        raise ValueError('Invalid value: ' + str(data_split))

    print('Allocating feature map array of size (' + str(data_size) + ', 65536)')
    image_embeddings = np.zeros((data_size, 65536), dtype=np.float32)
    print('Allocating caption of size (' + str(data_size) + ', 402)')
    captions = np.zeros((data_size, 402), dtype=np.float32)
    print_mem()

    start = 0
    for file_index in tqdm(range(n_split)):
        feat_maps = np.load(data_path + 'feature_maps_' + str(file_index) + '.npy')
        caps = np.load(data_path + 'captions_' + str(file_index) + '.npy')
        print_mem()
        stop = start + feat_maps.shape[0]
        image_embeddings[start:stop, :] = feat_maps
        print_mem()
        captions[start:stop, :] = caps
        start = stop
        print_mem()
        del feat_maps
        del caps
    
    return image_embeddings, captions

def predict(embeddings,train_captions, one_nn, batch_size=64):
    captions = []
    total_len = embeddings.shape[0]
    data_loader = np.array_split(embeddings, range(batch_size, total_len, batch_size))

    for j, batch in enumerate(tqdm(data_loader)):
        dists, indices = one_nn.kneighbors(batch)
        captions.extend([train_captions[i] for i in indices])
    captions = np.array(captions).reshape(embeddings.shape[0], -1)

    return captions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_threads', '-nt', type=int, help='Number of threads to run this script')
    args = parser.parse_args()
    num_threads = args.num_threads

    print('Number of threads: ' + str(num_threads))

    # Set number of threads
    try:
        os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    except Exception as e:
        print('Exception raised: \n')
        print(str(e))

    try:
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
    except Exception as e:
        print('Exception raised: \n')
        print(str(e))

    print('Loading train...')
    train_image_embeddings, train_captions = read_npy_files_as_matrix('train')
    print('Train size: ' + str(train_image_embeddings.shape))
    print('Train caption size: ' + str(train_captions.shape))
    print('Loading val...')
    val_image_embeddings, _ = read_npy_files_as_matrix('val')
    print('Val size: ' + str(val_image_embeddings.shape))
    print('Loading test...')
    test_image_embeddings, _ = read_npy_files_as_matrix('test')
    print('Test size: ' + str(test_image_embeddings.shape))

    indices = [*range(train_image_embeddings.shape[0])]
    one_nn = KNeighborsClassifier(n_neighbors=1)
    one_nn.fit(train_image_embeddings, indices)

    # Can free train_image_embeddings

    predicted_val_captions = predict(val_image_embeddings, train_captions, one_nn)
    np.save('predicted_val_captions.npy', predicted_val_captions)
    print('Val caption finished with size' + str(predicted_val_captions.shape))
    predicted_test_captions = predict(test_image_embeddings, train_captions, one_nn)
    np.save('predicted_test_captions.npy', predicted_test_captions)
    print('Test caption finished with size' + str(predicted_test_captions.shape))

if __name__ == '__main__':
    main()