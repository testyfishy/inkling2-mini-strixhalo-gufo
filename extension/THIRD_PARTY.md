# Third-party work this extension builds on

## headbouyJB/inkling2-gufo

`AttentionFlashKernel` in `src/hip/decode.hip` follows the design of the prefill attention
kernel in https://github.com/headbouyJB/inkling2-gufo (`lab/attn/attn_kernels.h`,
`attn_prefill`): one block per group of rows and key head, each wave keeping its queries, its
running softmax and its output in registers, the weights exchanged between lane pairs to form
the operand of the second product, and the rows of the staged keys and values padded so that a
wave's lanes read different banks of shared memory. The code here is written for this
engine's caches and arithmetic; the design and those two ideas are theirs.

MIT License

Copyright (c) 2026 headbouyJB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
