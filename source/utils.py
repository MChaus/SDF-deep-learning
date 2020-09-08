
import os 
import zipfile
import urllib.request

import torch
import skimage.measure

from .deep_sdf import Decoder

def marching_cubes(
    decoder,
    latent_vec,
    level =0.0,
    N=256,
    max_batch=32 ** 3
):
    ''' Return verts, faces, normals, values for given representation.
    '''
    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        print('computing verices {:.2f} %'.format(head / num_samples * 100), end='\r')

        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3]

        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    pytorch_3d_sdf_tensor = sdf_values.data.cpu()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()    

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )
    print('\nvertices are ready')

    return verts, faces, normals, values


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]
    latent_repeat = latent_vector.expand(num_samples, -1)
    
    inputs = torch.cat([latent_repeat, queries], 1)
    sdf = decoder(inputs)

    return sdf


def download_models(source_url, target_dir, target_file):
    print('Downloading ...')
    urllib.request.urlretrieve(source_url, filename=target_file)

    print('Unzipping ...')
    zip_ref = zipfile.ZipFile(target_file, 'r')
    zip_ref.extractall(target_dir)
    zip_ref.close()
    os.remove(target_file)

    print('Models were downloaded to {}'.format(target_dir))

def data_loader(path, specs, epochs: list, obj_id=0):
    for epoch in epochs:
        net_path = os.path.join(path, 'model_{}.pt'.format(epoch))
        lat_path = os.path.join(path, 'latent_vecs_{}.pt'.format(epoch))
        
        if not os.path.exists(net_path):
            continue
            
        deep_sdf = Decoder(specs['CodeLength'], **specs['NetworkSpecs'])
        data = torch.load(net_path)
        deep_sdf.load_state_dict(data["model_state_dict"])
        
        latent = torch.load(lat_path)
        latent_vec = latent['latent_codes']['weight'][obj_id]

        yield deep_sdf, latent_vec