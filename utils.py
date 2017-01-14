class SliceWindow(object):
    def __init__(self, data, samples, hop):
        self.data = data
        self.samples = samples
        self.hop = hop
        self.off = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.off + self.samples > len(self.data):
            raise StopIteration()

        else:
            window = self.data[self.off : self.off + self.samples]
            self.off += self.hop
            return window
