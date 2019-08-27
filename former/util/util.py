import torch, os, errno

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'former' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)

def profile_parameters(module):
    params = 0
    for p in module.parameters():
        params += p.numel()
    return params

def profile_model_weights(model):
    
    return (
        """
            Model params ------------------
            Model total: {}M
            Classification layer: {}M
            Transformers: {}M
            Single Transformer: {}M
            Self attention module: {}M
            MLP module: {}M

        """.format(*[round(1e-6*profile_parameters(x), 4) for x in [
            model, model.toprobs, model.tblocks, model.tblocks[0],
             model.tblocks[0].attention, model.tblocks[0].ff]]))

def estimate_memory_usage(b, t, k, h, nlayers, mlp_z, bytes_per_param=4):

    BACKWARD_MUL = 2

    btk = b*t*k

    mlp_w = 2*(mlp_z*k*k)
    mlp_z = 3*(mlp_z*btk)
    ln_z = btk
    att_w = 4*(h*k*k)
    att_z = 4*(h*btk) + 2*(btk) + b*h*t*t

    block_w = mlp_w+att_w
    block_z = mlp_z+2*ln_z+att_z

    return (
        """
            Estimated Model Size ------------------
            Self attention weights: {}MB
            Self attention activations: {}MB
            MLP weights: {}MB
            MLP activations: {}MB
            LayerNorm activations {}MB

            Transformer Block weights: {}MB
            Transformer Block activations: {}MB

            Total weights: {}MB
            Total activations: {}MB
        """.format(*[round(1e-6*bytes_per_param*BACKWARD_MUL*x, 4) for x in [
            att_w, att_z, mlp_w, mlp_z, ln_z, block_w, block_z,
             block_w*nlayers, block_z*nlayers]]))

def fprint(content, fpath):
    if not os.path.exists(os.path.dirname(fpath)):
        try:
            os.makedirs(os.path.dirname(fpath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(fpath, "w") as f:
       f.write(content)
    print(content)

class Bunch(object):
    def __init__(self, dict=None, **kwargs):
        if dict is not None:
            self.__dict__.update(dict)
        else:
            for name in kwargs:
                setattr(self, name, kwargs[name])
