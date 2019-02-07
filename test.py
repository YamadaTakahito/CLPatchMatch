import patchmatch as pm
import imageio
import os

import pyopencl as cl

import numpy as np

img = imageio.imread('bike_a.png') / 255
os.environ['PYOPENCL_NO_CACHE'] = '1'
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

filename = 'patchmatch.c'
PATCH_SIZE = 8


HEIGHT, WIDTH, _ = img.shape
ctx = cl.create_some_context(False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
program = cl.Program(ctx, open(filename).read()).build()

NNF_HEIGHT = HEIGHT - (HEIGHT % PATCH_SIZE)
NNF_WIDTH = WIDTH - (WIDTH % PATCH_SIZE)
# NNF_HEIGHT = HEIGHT - PATCH_SIZE
# NNF_WIDTH = WIDTH - PATCH_SIZE
nnf = np.empty((NNF_HEIGHT, NNF_WIDTH, 3))
inputBuf = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                      hostbuf=img) for _ in range(2)]
outputBuf = cl.Buffer(ctx, mf.WRITE_ONLY |
                      mf.COPY_HOST_PTR, nnf.nbytes, hostbuf=nnf)

GLOBAL_SIZE = (NNF_HEIGHT, NNF_WIDTH)
program.randomfill(queue, GLOBAL_SIZE, None,
                   np.int32(PATCH_SIZE),  # patchHeight
                   np.int32(PATCH_SIZE),  # patchWidth
                   np.int32(HEIGHT),  # height
                   np.int32(WIDTH),  # width
                   inputBuf[0],
                   inputBuf[1],
                   outputBuf)


init_nnf = np.empty_like(nnf)
cl.enqueue_copy(queue, init_nnf, outputBuf)
