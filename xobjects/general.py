class Print:
    suppress = False

    def __call__(self, *args, **kwargs):
        if not self.suppress:
            print(*args, **kwargs)


_print = Print()
