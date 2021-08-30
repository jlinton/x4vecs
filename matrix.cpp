// Copyright (c) 2012,2014 Jeremy Linton
//
// matrix.cpp
//	 C++ wrapper which creates a 4x4 matrix out of the x4 vector classes
//	 This matrix class works with the NEON,SSE and PORT classes
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.



// This class wraps one of the vector types to create a basic 4x4 matrix class
// Its pretty "clean" except for the Transpose operation which requires us to be
// friended by the Vec4 class so that we can access the individual vector members in a
// somewhat efficient manner.


template<int Size> class vMatrix
{
  public:
	vMatrix(void):Vecs() {}
	vMatrix(const vMatrix &orig);
	vMatrix(Vec4 Vecs_prm[4]);
	vMatrix(const Vec4 &V1,const Vec4 &V2,const Vec4 &V3,const Vec4 &V4);
	Vec4 operator*(const Vec4 &Mult);
	vMatrix operator*(const vMatrix &Mult);

	void Transpose();
	std::string as_str(const char *format_prm="%6.2f %6.2f %6.2f %6.2f");
  private:
	Vec4 Vecs[Size];

};

template<int Size> vMatrix<Size>::vMatrix(const vMatrix &orig):Vecs()
{
	for (int x=0; x<Size; x++)
	{
		Vecs[x]=orig.Vecs[x];
	}
}

template<int Size> vMatrix<Size>::vMatrix(const Vec4 &V1,const Vec4 &V2,const Vec4 &V3,const Vec4 &V4):Vecs()
{
	Vecs[0]=V1;
	Vecs[1]=V2;
	Vecs[2]=V3;
	Vecs[3]=V4;
}

template<int Size> vMatrix<Size>::vMatrix(Vec4 Vecs_prm[Size]):Vecs()
{
	for (int x=0; x<Size; x++)
	{
		Vecs[x]=Vecs_prm[x];
	}
}

template<int Size> Vec4 vMatrix<Size>::operator*(const Vec4 &Mult)
{
	Vec4 ret;
	// The generated code here isn't as nice as one would hope
	// for NEON this should be a vmula
	/*ret.Set(0,(Vecs[0]*Mult).ElementSum());
	ret.Set(1,(Vecs[1]*Mult).ElementSum());
	ret.Set(2,(Vecs[2]*Mult).ElementSum());
	ret.Set(3,(Vecs[3]*Mult).ElementSum());*/
	for (int x=0; x<Size; x++)
	{
		ret.Set(x,(Vecs[x]*Mult).ElementSum());
	}
	return ret;
}

template<int Size> inline vMatrix<Size> vMatrix<Size>::operator*(const vMatrix<Size> &Mult)
{
	vMatrix<Size> Trans(Mult);
	Trans.Transpose();

	vMatrix<Size> ret;
	for (int x=0; x<Size; x++)
	{
		for (int y=0; y<Size; y++)
		{
			ret.Vecs[y].Set(x,(Vecs[y]*Trans.Vecs[x]).ElementSum());
		}
	}
	/*ret.Vecs[0].Set(0,(Vecs[0]*Trans.Vecs[0]).ElementSum());
	ret.Vecs[1].Set(0,(Vecs[1]*Trans.Vecs[0]).ElementSum());
	ret.Vecs[2].Set(0,(Vecs[2]*Trans.Vecs[0]).ElementSum());
	ret.Vecs[3].Set(0,(Vecs[3]*Trans.Vecs[0]).ElementSum());


	ret.Vecs[0].Set(1,(Vecs[0]*Trans.Vecs[1]).ElementSum());
	ret.Vecs[1].Set(1,(Vecs[1]*Trans.Vecs[1]).ElementSum());
	ret.Vecs[2].Set(1,(Vecs[2]*Trans.Vecs[1]).ElementSum());
	ret.Vecs[3].Set(1,(Vecs[3]*Trans.Vecs[1]).ElementSum());

	ret.Vecs[0].Set(2,(Vecs[0]*Trans.Vecs[2]).ElementSum());
	ret.Vecs[1].Set(2,(Vecs[1]*Trans.Vecs[2]).ElementSum());
	ret.Vecs[2].Set(2,(Vecs[2]*Trans.Vecs[2]).ElementSum());
	ret.Vecs[3].Set(2,(Vecs[3]*Trans.Vecs[2]).ElementSum());

	ret.Vecs[0].Set(3,(Vecs[0]*Trans.Vecs[3]).ElementSum());
	ret.Vecs[1].Set(3,(Vecs[1]*Trans.Vecs[3]).ElementSum());
	ret.Vecs[2].Set(3,(Vecs[2]*Trans.Vecs[3]).ElementSum());
	ret.Vecs[3].Set(3,(Vecs[3]*Trans.Vecs[3]).ElementSum());
	*/

	return ret;
}

template<int Size> void vMatrix<Size>::Transpose(void)
{
	// Think about gcc's __builtin_shuffle() to replace this mess
	// this method is also why we need to friend the matrix class
	//
#ifdef __SSE3__
	_MM_TRANSPOSE4_PS(Vecs[0].quad_floats, Vecs[1].quad_floats, Vecs[2].quad_floats, Vecs[3].quad_floats);
#elif defined(__ARM_NEON__)
	// this path should be
	// vtrn.32 vec0,vec1
	// vtrn.32 vec2,vec3
	// vswp vec0low,vec2low
	// vswp vec1high,vec3high
	// or something..
	// fight with the compiler here.
	// TODO: its probably easier to do this with inline assembly (if I can figure out the gcc in/out/clobber syntax for neon registers)
	// The "undocumented" register information is @ http://hardwarebug.org/2010/07/06/arm-inline-asm-secrets based on constraints.md
	// in the gcc source.

	float32x4x2_t tmp=vtrnq_f32 (Vecs[0].quad_floats, Vecs[1].quad_floats);
	Vecs[0].quad_floats=((float32x4_t*)&tmp)[0];
	Vecs[1].quad_floats=((float32x4_t*)&tmp)[1];
	tmp=vtrnq_f32 (Vecs[2].quad_floats, Vecs[3].quad_floats);
	Vecs[2].quad_floats=((float32x4_t*)&tmp)[0];
	Vecs[3].quad_floats=((float32x4_t*)&tmp)[1];
	//now i need a vswp but there isn't an intrinisc!
	// this should collapse to vswp if we are lucky!
	float32x2_t tmp2=vget_high_f32(Vecs[0].quad_floats);
	Vecs[0].quad_floats=vcombine_f32(vget_low_f32(Vecs[0].quad_floats),vget_low_f32(Vecs[2].quad_floats));
	Vecs[2].quad_floats=vcombine_f32(tmp2,vget_high_f32(Vecs[2].quad_floats));

	tmp2=vget_high_f32(Vecs[1].quad_floats);
	Vecs[1].quad_floats=vcombine_f32(vget_low_f32(Vecs[1].quad_floats),vget_low_f32(Vecs[3].quad_floats));
	Vecs[3].quad_floats=vcombine_f32(tmp2,vget_high_f32(Vecs[3].quad_floats));
	// with my version of GCC this is being generated as:
	//vtrn.32 q10, q11	// Oh look! We got lucky, the compiler figured out the ugly ptr copy syntax I used!
	//vtrn.32 q8, q9
	// Sigh, or maybe not, and it decided to generate this mess
	// of mov's instead of the vswp's I wanted?
	//vmov	  d26, d20	@ v2sf
	//vmov	  d27, d16	@ v2sf
	//vmov	  d24, d21	@ v2sf
	//vmov	  d25, d17	@ v2sf
	//vmov	  d20, d22	@ v2sf
	//vmov	  d21, d18	@ v2sf
	//vmov	  d16, d23	@ v2sf
	//vmov	  d17, d19	@ v2sf
#elif defined(PORTABLE_VEC)
	// just do a basic set of swaps
	for (int x=1;x<Size;x++)
	{
		for (int y=0;y<x;y++)
		{
			float tmp=Vecs[x].elements[y];
			Vecs[x].elements[y]=Vecs[y].elements[x];
			Vecs[y].elements[x]=tmp;
		}
	}
#endif
}


template<int Size> std::string vMatrix<Size>::as_str(const char *format_prm)
{
	std::string ret; //= Vecs[0].as_str(format_prm)+std::string("\n");+Vecs[1].as_str(format_prm)+std::string("\n")+Vecs[2].as_str(format_prm)+std::string("\n")+Vecs[3].as_str(format_prm)+std::string("\n");
	for (int x=0;x<Size;x++)
	{
		ret+=Vecs[x].as_str(format_prm)+std::string("\n");
	}
	return ret;
}

// Ok lets just use the FPU for transendentals for now....
// there are some nice approximation libs on the interwebs..
#include <math.h>

typedef vMatrix<4> SSEx4Matrix;

class Translate: public SSEx4Matrix
{
  public:
	Translate(const float x,const float y,const float z):SSEx4Matrix(Vec4(1,0,0,x),Vec4(0,1,0,y),Vec4(0,0,1,z),Vec4(0,0,0,1)) {}
};

class Scale: public SSEx4Matrix
{
  public:
	Scale(const float x,const float y,const float z):SSEx4Matrix(Vec4(x,0,0,0),Vec4(0,y,0,0),Vec4(0,0,z,0),Vec4(0,0,0,1)) {}
};

// right handed (incase we eventually love GL)
class RotateXRad: public SSEx4Matrix
{
  public:
	RotateXRad(float Rad):SSEx4Matrix(Vec4(1,0,0,0),Vec4(0,cos(Rad),sin(Rad),0),Vec4(0,-sin(Rad),-cos(Rad),0),Vec4(0,0,0,1)) {}
};

class RotateYRad: public SSEx4Matrix
{
  public:
	RotateYRad(float Rad):SSEx4Matrix(Vec4(cos(Rad),0,-sin(Rad),0),Vec4(0,1,0,0),Vec4(-sin(Rad),0,-cos(Rad),0),Vec4(0,0,0,1)) {}
};


class RotateZRad: public SSEx4Matrix
{
  public:
	RotateZRad(float Rad):SSEx4Matrix(Vec4(cos(Rad),sin(Rad),0,0),Vec4(-sin(Rad),cos(Rad),0,0),Vec4(0,0,1,0),Vec4(0,0,0,1)) {}
};

class Projection: public SSEx4Matrix
{
  public:
	Projection(int w,int h):SSEx4Matrix(Vec4(1,0,0,0),Vec4(0,1,0,0),Vec4(0,0,-1,-1),Vec4(0,0,0,0)) {}
};
