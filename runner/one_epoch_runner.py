import tensorflow as tf
import datetime
import numpy as np
import zutils.tf_math_funcs as tmf

from zutils.py_utils import *
from scipy.io import savemat


def unnest_dict(dict_, join_subkeys=True, subkey_sep="_"):
    """Takes a nested dictionary structure and unpacks the inner dictionaries into the topmost dictionary.
    
    This is useful when working with recursively nested
    dict structures and the goal is to convert them to a DataFrame.

    Parameters
    ----------
    dict_ : dict
        the nested dictionary structure to transform
    join_subkeys : bool, optional
        if the key from the partent level should be joined to the child level key.
        For example : {"test" : {"1" : 2, "3" : 4}} will result in {"test_1" : 2, "test_3" : 4} if join_subkeys is True.
        If False, the keys from the children will be taken. This could overwrite parrent elements in the worst case.
        By default True. 
    subkey_sep : str, optional
        the seperating character to use for joining subkeys. By default "_".
    
    Returns
    -------
    dict
        new dictionary with unnested structure

    Examples
    --------
        dict_ = {1 : 2, 3 : {4 : 5, 6 : 7}}
        dict_["test"] = dict_.copy()
        new_dict = unnest_dict(dict_)
        dict_, new_dict

        # create dataframe from old dict results in dict becoming a value in the table
        df1 = pd.DataFrame(dict_)
        
        # creating dataframe from new dict results a nice tabular dataframe
        df = pd.DataFrame(listify_dict(new_dict))
    """

    new_dict = {}
    A = {}  # subdict that contains only non-dict values
    B = {}  # subdict that contains only dict values
    for k, v in dict_.items():
        if isinstance(v, dict):
            B[k] = v
        else:
            A[k] = v

    new_dict.update(A)
    for k, b in B.items():
        if join_subkeys:
            new_b = {subkey_sep.join([str(k), str(bk)]): bv for bk, bv in b.items()}
        else:
            new_b = b
        new_b = unnest_dict(
            new_b, join_subkeys=join_subkeys, subkey_sep=subkey_sep
        )  # recurse into subdictionary
        new_dict.update(new_b)
    return new_dict

class OneEpochRunner:

    def __init__(
            self, data_module, output_list=None,
            net_func=None, batch_axis=0, num_samples=None, disp_time_interval=2,
            output_fn=None, is_large=False):

        self.data_module = data_module
        self.num_samples = self.data_module.num_samples()
        self.batch_axis = batch_axis
        self.disp_time_interval = disp_time_interval
        self.output_fn = output_fn
        self.is_large = is_large

        if num_samples is not None:
            if self.num_samples < num_samples:
                print("specified number_samples is larger than one epoch")
            else:
                self.num_samples = num_samples

        self.use_net_func = output_list is None  # otherwise use net_func
        if self.use_net_func:
            assert net_func is not None, \
                "output_list and net_func should not be both specified"
            self.net_func = net_func
            # remark: net_func(sess)
        else:
            assert net_func is None, \
                "one of output_list and net_func must be specified"
            self.output_list = output_list
            [self.flatten_output_list, self.output_wrap_func] = \
                recursive_flatten_with_wrap_func(
                    lambda x: tmf.is_tf_data(x), self.output_list)

        self.data_module.reset()
        self.cur_sample_end = 0

    def run_single_batch(self, sess):

        if self.cur_sample_end >= self.num_samples:
            return None

        if self.use_net_func:
            output_val = self.net_func(sess)
        else:
            output_val = sess.run(self.flatten_output_list, {})
            output_val = self.output_wrap_func(output_val)

        batch_size = first_element_apply(
            lambda x: isinstance(x, np.ndarray),
            lambda x: x.shape[self.batch_axis], output_val)
        self.batch_size = batch_size

        new_end = self.cur_sample_end + batch_size
        if new_end > self.num_samples:
            effective_batch_size = \
                batch_size - (new_end-self.num_samples)
            slice_indexes = (slice(None),)*self.batch_axis + (slice(effective_batch_size),)
            output_val = recursive_apply(
                lambda x: isinstance(x, np.ndarray),
                lambda x: x[slice_indexes], output_val)
        self.cur_sample_end = new_end
        return output_val

    def run(self, sess):
        disp_countdown = IfTimeout(self.disp_time_interval)
        num_samples_total = self.num_samples

        output_val_single = self.run_single_batch(sess)
        output_val = []

        while output_val_single is not None:
            output_val += [output_val_single]

            iter = self.data_module.iter()
            if self.data_module.epoch() == 0:
                num_samples_finished = self.data_module.num_samples_finished()
            else:
                num_samples_finished = self.num_samples

            if disp_countdown.is_timeout():
                epoch_percentage = num_samples_finished / num_samples_total * 100
                print("%s] Iter %d (%4.1f%% = %d / %d)" %
                      (datetime.datetime.now().strftime('%Y-%m/%d-%H:%M:%S.%f'),
                       iter, epoch_percentage, num_samples_finished, num_samples_total))
                disp_countdown = IfTimeout(self.disp_time_interval)

            
            if self.is_large and (num_samples_finished % (100*self.batch_size) == 0 or num_samples_finished == self.num_samples):
                output_val = recursive_apply(
                    lambda *args: isinstance(args[0], np.ndarray),
                    lambda *args: np.concatenate(args, axis=self.batch_axis),
                    *output_val)
                self.dir_path = os.path.dirname(self.output_fn+'_'+'%06d'%num_samples_finished)
                if not os.path.exists(self.dir_path):
                    os.makedirs(self.dir_path)
                savemat(self.output_fn+'_'+'%06d'%num_samples_finished+'.mat',output_val)
                np.savez_compressed(self.output_fn+'_'+'%06d'%num_samples_finished+'.npz', **unnest_dict(output_val))
                print('Saving part of output to '+ self.output_fn+'_'+'%06d'%num_samples_finished+'.mat')
                output_val = []
            output_val_single = self.run_single_batch(sess)
 
        if not self.is_large:
            output_val = recursive_apply(
                lambda *args: isinstance(args[0], np.ndarray),
                lambda *args: np.concatenate(args, axis=self.batch_axis),
                *output_val)
            savemat(self.output_fn + ".mat", output_val)
            np.savez_compressed(self.output_fn + ".npz", **unnest_dict(output_val))
            print('Saving output to ' + self.output_fn + ".mat")
