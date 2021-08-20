
import pandas as pd
import os, sys, shutil

class BatchResultsWriter():
    def __init__(self, split_num, output_dir, output_file_name_prefix, batch_size=20):
        self._split_num = split_num
        self._curr_batch_num = 0
        self._batch_size = batch_size
        self._output_dir = output_dir
        self._output_file_name_prefix = output_file_name_prefix
        self._buffer = []
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)


    def _clear_output_dir(self):
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        else: 
            for filename in os.listdir(self._output_dir):
                file_path = os.path.join(self._output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))    


    def append_and_maybe_write(self, item):
        self._buffer.append(item)
        if len(self._buffer) == self._batch_size:
            self.write_results()


    def write_results(self):        
        if len(self._buffer):   
            
            df = pd.concat(self._buffer) 

            file_name = self._output_dir + \
                f"{self._output_file_name_prefix}_split_{self._split_num}_batch_{self._curr_batch_num}.csv"
            
            df.to_csv(file_name, index=False, float_format='%.3f')

            self._buffer = []
            self._curr_batch_num += 1
            
    def __len__(self):
        return len(self._buffer)


    def __str__(self):
        return (f'ForecastResultsWriter\n'\
            f'  output_dir: {self._output_dir}\n'\
            f'  output_file_name:{self._output_file_name_prefix}\n'\
            f'  split_num: {self._split_num}\n' )

        