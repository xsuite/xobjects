import numpy as np

from xobjects import CLContext, ByteArrayContext

class Check:
    def __init__(self,ctx,capacity):
        self.ctx=ctx
        self.buffer=ctx.new_buffer(capacity=capacity)
        self.state={}

    def random_string(self,maxlength=100):
        size = np.random.randint(1, maxlength)
        self.new_string(size)

    def new_string(self,size=100):
        if size>0:
             data=bytes(np.random.randint(65, 90, size, dtype="u1"))

             offset=self.buffer.allocate(len(data))
             self.buffer.write(offset,data)
             self.state[offset]=data
             return offset
        else:
            raise ValueError("size must be >0")

    def free_string(self,offset):
        size=len(self.state[offset])
        self.buffer.free(offset,size)
        del self.state[offset]

    def random_free(self):
        ii=np.random.randint(1, len(self.state))
        offset=list(self.state.keys())[ii]
        self.free_string(offset)

    def check(self):
        for offset, value in self.state.items():
            assert self.buffer.read(offset, len(value)) == value


def test_cl_print_devices():
    CLContext.print_devices()

def test_cl_init():
    ctx=CLContext(device="0.0")

def test_cl_new_buffer():
    ctx=CLContext(device="0.0")
    buff1=ctx.new_buffer()
    buff2=ctx.new_buffer(capacity=100)

def test_new_buffer():
    ctx=ByteArrayContext()
    buff1=ctx.new_buffer()
    buff2=ctx.new_buffer(capacity=200)

def test_read_write():
    buff=ByteArrayContext().new_buffer()
    bb = b"asdfasdfafsdf"
    buff.write(23, bb)
    assert buff.read(23, len(bb)) == bb

def test_cl_read_write():
    ctx=CLContext(device="0.0")
    buff = ctx.new_buffer(100)
    bb = b"asdfasdfafsdf"
    buff.write(23, bb)
    assert buff.read(23, len(bb)) == bb

def test_allocate_simple():
    ctx=CLContext(device="0.0")
    ch=Check(ctx,200)
    ch.new_string(30)
    ch.check()

def test_free_simple():
    ctx=CLContext(device="0.0")
    ch=Check(ctx,200)
    offsets=  [ ch.new_string(ii*2+1) for ii in range(10)]
    print(offsets)
    for offset in offsets:
        print(offset)
        ch.free_string(offset)
        ch.check()

def test_grow():
    ctx=CLContext(device="0.0")
    ch=Check(ctx,200)
    ch.new_string(150)
    ch.new_string(60)
    ch.check()
    assert ch.buffer.capacity==400
    assert ch.buffer.chunks[0].start==210
    assert ch.buffer.chunks[0].end==400
    ch.new_string(500)
    assert ch.buffer.capacity==900
    ch.check()



def test_random_string():

    ctx = CLContext(device="0.0")
    ch=Check(ctx,200)

    for i in range(50):
        ch.random_string(maxlength=2000)
        ch.check()

    for i in range(50):
        ch.random_string(maxlength=2000)
        ch.check()
        ch.random_free()
        ch.check()

