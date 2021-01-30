import os
import time
from collections import deque, OrderedDict, defaultdict
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter


class RecordDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __setitem__(self, key, value):
        if key in self:
            self[key].record_value(value)
        else:
#             print(type(value), isinstance(value, _BaseRec), value.__class__)
            assert isinstance(value, _BaseRec), 'unregistered value assignment'
            super().__setitem__(key, value)
            

class Recorder:
    def __init__(self, default_win_size='inf', **kwargs):
        self._record = defaultdict(RecordDict)
        self.default_win_size = default_win_size
        self.__dict__.update(kwargs)

    def _add_record_key(self, key, _win_size, _mode, _group):
        assert _win_size == 'inf' or (type(_win_size) == int and _win_size > 0), \
            'invalid window size {}'.format(_win_size)
        assert _mode in {'max', 'min', 'sum', 'mean', 'plain'}, \
            'invalid summary mode {}'.format(_mode)
        _recorder = {
            'max': _MaxRec, 'min': _MinRec, 'sum': _SumRec, 
            'mean': _MeanRec, 'plain': _PlainRec,
            }[_mode]
        self._record[_group][key] = _recorder(_win_size)
#         print(self._record[_group][key])

    def register_key(self, key, window_size=None, mode='plain', group='default'):
        """
        method for recording values
        Args:
            key: key to be recorded
            window_size: window size for smoothing, using self.default_win_size if None
            mode: sepcify recording max, min, sum, or mean
            group: for which group the key should be placed
        """
        if group not in self._record or key not in self._record[group]:
            _win_size = self.default_win_size if window_size is None else window_size
            self._add_record_key(key, _win_size, mode, group)
        _recorder = self._record[group][key]
        assert window_size is None or _recorder.win_size == window_size, 'window size mismatch {} {}'.format(_recorder.win_size, window_size)
        assert mode is None or _recorder.mode == mode, 'summary mode mismatch'
    
    def __getitem__(self, group):
        return self._record[group]
    
    def __iter__(self):
        return self._record.__iter__()

    @contextmanager
    def register_time(self, key, window_size=None, mode='plain', group='default'):
        self.register_key(key, window_size, mode, group)
        _tick = time.time()
        yield
        self._record[group][key] = time.time() - _tick

    def summary(self, key, group='default'):
        if group not in self._record:
            raise KeyError('group {} not been recorded'.format(group))
        if key not in self._record[group]:
            raise KeyError('key {} not been recorded in group {}'.format(key, group))
        return self._record[group][key].summary()

    def groups(self):
        return list(self._record.keys())

    def keys(self, group='default'):
        return list(self._record[group].keys())

    def values(self, group='default'):
        return list(_rec.summary() for _rec in self._record[group].values())

    def items(self, group='default'):
        return list(zip(self.keys(group), self.values(group)))

    def pop(self, key, group='default', nonexist_ret=None):
        return self._record[group].pop(key, nonexist_ret)

    def reset_group(self, group='default'):
        self._record[group].clear()

    def reset_all(self):
        self._record.clear()
        
        
class _BaseRec:
    def __init__(self, _mode, _win_size):
        self.mode = _mode
        self.win_size = _win_size
        
    def record_value(self, value):
        raise NotImplementedError
    
    def summery(self):
        raise NotImplementedError
        
    def __str__(self):
        return 'Rec Object ' + \
        '[mode: {}, win_size: {}]'.format(
            self.mode, self.win_size,)

        
class _PlainRec(_BaseRec):
    def __init__(self, _win_size):
        super().__init__('plain', _win_size)
        self.window = [] if _win_size == 'inf' else deque(maxlen=_win_size)
    
    def record_value(self, value):
        self.window.append(value)
        
    def summary(self):
        return self.window
    
        
class _MeanRec(_BaseRec):
    def __init__(self, _win_size):
        super().__init__('mean', _win_size)
        self.window = [] if _win_size == 'inf' else deque(maxlen=_win_size)

    def record_value(self, value):
        self.window.append(value)

    def summary(self):
        return sum(self.window) / max(len(self.window), 1)


class _SumRec(_BaseRec):
    def __init__(self, _win_size):
        super().__init__('sum', _win_size)
        self.value = 0 if _win_size == 'inf' else deque(maxlen=_win_size)

    def record_value(self, value):
        if self.win_size == 'inf':
            self.value += value
        else:
            self.value.append(value)

    def summary(self):
        if self.win_size == 'inf':
            return self.value
        else:
            return sum(self.value)


class _PolarizedRec(_BaseRec):
    def __init__(self, _mode, _win_size):
        super().__init__(_mode, _win_size)
        self.queue = None if _win_size == 'inf' else deque(maxlen=_win_size)
        self.value = None
        self.polarize = eval(_mode)

    def record_value(self, value):
        if self.queue is not None:
            self.queue.append(value)
        elif self.value is not None:
            self.value = self.polarize(self.value, value)
        else:
            self.value = value

    def summary(self):
        if self.queue is not None: return self.polarize(self.queue)
        return self.value


class _MinRec(_PolarizedRec):
    def __init__(self, _win_size):
        super().__init__('min', _win_size)

class _MaxRec(_PolarizedRec):
    def __init__(self, _win_size):
        super().__init__('max', _win_size)
    


GlobalRecorder = Recorder()



class TensorboardWriter(SummaryWriter):
    """SummaryWriter wrapper
    https://pytorch.org/docs/stable/tensorboard.html
    """
    def __init__(self, log_dir, **kwargs):
        super(TensorboardWriter, self).__init__(log_dir=log_dir, **kwargs)
        if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)

    def add_textfile(self, tag, filename):
        assert type(tag) == str
        with open(filename, 'r') as f:
            # since 'list' object has no attribute 'encode'
            content = '\n'.join(f.readlines())
        self.add_text(tag, content)
        
    def add_dict(self, tag, dict_obj):
        assert type(tag) == str
        content = '\n'.join('{}: {}'.format(k, v) for k, v in dict_obj.items())
        self.add_text(tag, content)
        
        