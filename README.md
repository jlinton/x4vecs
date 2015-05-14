x4vecs
======

C++ wrapper class for SSE/AVX/NEON/Portable x4 float vectors, with a 4x4 matrix class and a wireframe renderer unit test


Summary
=======

This project originally was started (somewhere in the mid 2000's) simply to mess around with some of the intel SSE intrinsics. 
Over time I piled a few other things into it, NEON support, AVX x8, a basic C++ reimplementation without any intrinsic usage, a 4x4 
matrix class, and last year (2013) a little wireframe "renderer" using SDL and pugixml to read .DAE files. 

No real effort has been made to optimize for one case or another, and benchmarking against the portable version
shows there are definitly cases where the portable version is faster (especially if a lot of insert/extract operations are being used)

If you need a library for multiple hardware platforms, this might be useful, otherwise there are probably better choices, 
especially for SSE.  

Simple code examples
====================

This is not "documentation" see the class definitions and the unit tests for actual examples!
<pre>
Construction:
	NEONx4 w(7,8,9,10);
	float y[4]={1,2,3,4};
	NEONx4 x(y); // loads floats into hardware vector
	NEONx4 z(5); // loads a single value into first elements, sets remaining to 0
	NEONx4 y; // Uninitialized

Extraction/Insertion of vector elements:
        x[y]      // returns element y from vector x as float
        x.Get(y)  // returns element y from vector x as vector 
	x.Set(y,value) //sets element y of vector x to value

Operations:
	Pretty much everything you would expect, including overloading the global float
	.Dot() // dot product 
	.ElementSum() //the elements of the vector summed and returned as a float
</pre>
Legal Stuff
==========
Copyright (c) 2012,2014 Jeremy Linton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

