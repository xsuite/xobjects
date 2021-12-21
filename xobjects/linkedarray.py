
class BaseLinkedArray():

    container = None
    container_setitem_name=None
    mode = None

    @classmethod
    def from_array(cls, a, mode=None, container=None,
                   container_setitem_name=None):
        assert len(a.shape) == 1 # TODO: To be generalized
        assert mode in (None, 'readonly', 'setitem_from_container')
        self = cls._build_view(a)
        self.mode=mode
        self.container = container
        self.container_setitem_name = container_setitem_name
        return self

    def __setitem__(self, indx, val):
        if (self.mode is None or
            (hasattr(self.container, '_flag_bypass_linked')
                         and self.container._flag_bypass_linked)):
            super().__setitem__(indx, val)
        elif self.mode == 'setitem_from_container':
            getattr(self.container, self.container_setitem_name)(indx, val)
        elif self.mode == 'readonly':
            raise ValueError('This array is read only')

class BypassLinked:
    def __init__(self, container):
        self.container = container

    def __enter__(self):
        self.container._flag_bypass_linked = True

    def __exit__(self, *args, **kwargs):
        del(self.container._flag_bypass_linked)