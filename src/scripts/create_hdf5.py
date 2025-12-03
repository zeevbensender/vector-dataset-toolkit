import argparse
import h5py
import numpy


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='', help='groundtruth file')
    parser.add_argument('--base', type=str, default='', help='base file')
    parser.add_argument('--query', type=str, default='', help='query file')
    parser.add_argument('--hdf5', type=str, default='', help='output hdf5 file')
    args = parser.parse_args()
    gnd_file_name = args.gt
    fbin_base_file_name = args.base
    query_base_file_name = args.query
    hdf5_file_name = args.hdf5
    gnd_file = open(gnd_file_name, 'rb')
    attrs = numpy.fromfile(gnd_file, count=2, dtype=numpy.int32)
    print(attrs)
    sz_query, max_k = attrs[0], attrs[1]
    data = numpy.fromfile(gnd_file, count=sz_query * max_k, dtype=numpy.int32)
    arr_neighbors = numpy.reshape(data, [sz_query, max_k])
    data = numpy.fromfile(gnd_file, count=sz_query * max_k, dtype=numpy.float32)
    arr_distances = numpy.reshape(data, [sz_query, max_k]) # this is actually the square distances -  (arr_distances**0.5) should apply on write

    query_file = open(query_base_file_name, 'rb')
    attrs = numpy.fromfile(query_file, count=2, dtype=numpy.int32)
    print(attrs)
    num_query_vectors, dimension = attrs[0], attrs[1]
    query_data = numpy.fromfile(query_file, count=num_query_vectors * dimension, dtype=numpy.float32)
    arr_queries = numpy.reshape(query_data, [num_query_vectors, dimension])

    base_file = open(fbin_base_file_name, 'rb')
    attrs = numpy.fromfile(base_file, count=2, dtype=numpy.int32)
    print(attrs)
    num_base_vectors, dimension = attrs[0], attrs[1]
    print('arr_queries:' + str(len(arr_queries)))
    print('-------------------------------------------')
    # print(arr_queries)
    # print('arr_base:' + str(len(arr_base)))
    # print('-------------------------------------------')
    # print(arr_base)

    fout = h5py.File(hdf5_file_name, "w")
    fout.create_dataset('distances', data=arr_distances**0.5)  # squared of the real distances
    fout.create_dataset('neighbors', data=arr_neighbors)
    fout.create_dataset('test', data=numpy.float32(arr_queries))
    # fout.create_dataset('train', data=numpy.float32(arr_base))
    print('create_dataset num_base_vectors: ' + str(num_base_vectors))
    if num_base_vectors >= 1e6:
        sz_b = int(1e6)
    else:
        sz_b = num_base_vectors
    print('sz_b: ' + str(sz_b))
    fout.create_dataset("train", (num_base_vectors, dimension), chunks=(sz_b, dimension))
    num_of_chunks = int(num_base_vectors / sz_b)
    print('num_of_chunks:' + str(num_of_chunks))
    for i in range(num_of_chunks):
        data = numpy.fromfile(base_file, count=sz_b * dimension, dtype=numpy.float32)
        arr_base = numpy.reshape(data, [sz_b, dimension])
        block_to_write = numpy.float32(arr_base)
        fout["train"][(i * sz_b):((i + 1) * sz_b), :] = block_to_write
        print('finished chunk ' + str(i))

    fout.attrs.create('distance', 'euclidean')
    fout.close()
    print('hdf5 file created: ' + hdf5_file_name)
    print('END')


if __name__ == "__main__":
    main()
