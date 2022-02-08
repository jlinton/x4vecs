// Copyright (c) 2012,2014 Jeremy Linton
//
// sse_objs2.cpp
//	 C++ wrapper around the SSE3 x4 float vector intrinsics
//	 with a matching class for NEON and a "portable" implementation
//	 There is also a AVX x8 float class implmented here
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




// To build some basic unit tests for this module do
// SSEx4 version
//	  g++ -msse3 -g -O3 sse_objs.cpp
// AVXx8 version
//	  g++ -mavx -g -O3 sse_objs.cpp
//
// This has been tested with
//gcc (SUSE Linux) 4.7.1 20120723 [gcc-4_7-branch revision 189773]

// suppress const string to sprintf usage warnings
#pragma GCC diagnostic ignored "-Wwrite-strings"

#include <string>
#include <stdio.h>

template<int Size> class vMatrix;

#include <pmmintrin.h>
//#include <mmintrin.h>

// Use cpp -dM too see lists of defined macros
#ifndef	 __SSE3__
#error "You must compile with -msse3"
#endif

#ifdef __SSE4_1__
#pragma message "Building SSE4"
#include <smmintrin.h>
#endif

#ifdef __AVX__
#pragma message "Building AVX"
#include <immintrin.h>
//#include <avxintrin.h>
#define _mm256_load1_ps(x) _mm256_set_ps(x, x, x, x, x, x, x, x)


// well should we use loadu or load?
// seems that although intel says 16 byte aligment is sometimes ok, its not
// really, and GCC seems to have issues doing proper placment
// force it with alignas, although depending on compiler version this
// isn't right 100% of the time, with newer compilers try c++17/20 instead
// of c++11
class alignas(32) SSEx8
{
  public:
	// initial construction
	SSEx8(){}; //construct unititalized
	SSEx8(float src) {octo_floats=_mm256_castps128_ps256(_mm_load_ss(&src));}; //load single value, clear remaining
	SSEx8(float *src) {octo_floats=_mm256_loadu_ps(src);}; //load 8 floats
	SSEx8(float a,float b,float c,float d,float e,float f,float g,float h) { float tmp[8]; tmp[0]=a;tmp[1]=b;tmp[2]=c;tmp[3]=d;tmp[4]=e;tmp[5]=f;tmp[6]=g;tmp[7]=h; octo_floats=_mm256_loadu_ps(tmp); }; //load 8 floats (same as mm_set_ps)

	// copy constructor
	//SSEx8(const SSEx8 &src_prm) { octo_floats=_mm_shuffle_ps(src_prm.octo_floats,src_prm.octo_floats,_MM_SHUFFLE(3,2,1,0)); };

	// assignment
//	SSEx8 & operator=(const SSEx8 &src_prm) { octo_floats=_mm_shuffle_ps(src_prm.octo_floats,src_prm.octo_floats,_MM_SHUFFLE(3,2,1,0)); }
	SSEx8 & operator=(const SSEx8 &src_prm) { octo_floats=src_prm.octo_floats; return *this;}
	SSEx8 & operator=(const float x[8]) { octo_floats=_mm256_loadu_ps(x); return *this;};
	void Set(int index,float value);

	// reads
	float operator[](int index_prm) const;
	SSEx8 Get(int index_prm);
	void save(float *dst_prm) { _mm256_store_ps(dst_prm,octo_floats);};
	std::string as_str(const char *format_prm="%1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f");

	// adds urinary
	SSEx8 & operator+=(const SSEx8 &src2) { octo_floats=_mm256_add_ps(octo_floats,src2.octo_floats); return *this;}
	SSEx8 & operator-=(const SSEx8 &src2) { octo_floats=_mm256_sub_ps(octo_floats,src2.octo_floats); return *this;}
	SSEx8 & operator+=(const float src2) { __m256 tmp=_mm256_load1_ps(src2); octo_floats=_mm256_add_ps(octo_floats,tmp); return *this;}
	SSEx8 & operator-=(const float src2) { __m256 tmp=_mm256_load1_ps(src2); octo_floats=_mm256_sub_ps(octo_floats,tmp); return *this;}
	// adds binary
	SSEx8 const operator+ (const SSEx8 &src2) { SSEx8 tmp; tmp.octo_floats=_mm256_add_ps(octo_floats,src2.octo_floats); return tmp;}
	SSEx8 const operator- (const SSEx8 &src2) { SSEx8 tmp; tmp.octo_floats=_mm256_sub_ps(octo_floats,src2.octo_floats); return tmp;}
	SSEx8 const operator+ (const float src2)  { SSEx8 tmp; tmp.octo_floats=_mm256_load1_ps(src2); tmp.octo_floats=_mm256_add_ps(octo_floats,tmp.octo_floats); return tmp;}
	SSEx8 const operator- (const float src2)  { SSEx8 tmp; tmp.octo_floats=_mm256_load1_ps(src2); tmp.octo_floats=_mm256_sub_ps(octo_floats,tmp.octo_floats); return tmp;}


	// muls urinary
	SSEx8 & operator*=(const SSEx8 &src2) { octo_floats=_mm256_mul_ps(octo_floats,src2.octo_floats); return *this;}
	SSEx8 & operator/=(const SSEx8 &src2) { octo_floats=_mm256_div_ps(octo_floats,src2.octo_floats); return *this;}
	SSEx8 & operator*=(const float src2) { __m256 tmp=_mm256_load1_ps(src2); octo_floats=_mm256_mul_ps(octo_floats,tmp); return *this;}
	SSEx8 & operator/=(const float src2) { __m256 tmp=_mm256_load1_ps(src2); octo_floats=_mm256_div_ps(octo_floats,tmp); return *this;}
	// muls binary
	SSEx8 const operator* (const SSEx8 &src2) { SSEx8 tmp; tmp.octo_floats=_mm256_mul_ps(octo_floats,src2.octo_floats); return tmp;}
	SSEx8 const operator/ (const SSEx8 &src2) { SSEx8 tmp; tmp.octo_floats=_mm256_div_ps(octo_floats,src2.octo_floats); return tmp;}
	SSEx8 const operator* (const float src2) { SSEx8 tmp; tmp.octo_floats=_mm256_load1_ps(src2); tmp.octo_floats=_mm256_mul_ps(octo_floats,tmp.octo_floats); return tmp;}
	SSEx8 const operator/ (const float src2) { SSEx8 tmp; tmp.octo_floats=_mm256_load1_ps(src2); tmp.octo_floats=_mm256_div_ps(octo_floats,tmp.octo_floats); return tmp;}


	// other manipulations
	// floatFMAC(const SSEx8 &src2) {}
	float ElementSum(void) const { __m256 tmp = _mm256_permute2f128_ps(octo_floats , octo_floats , 1); tmp = _mm256_add_ps(tmp, octo_floats); tmp = _mm256_hadd_ps(tmp, tmp); tmp = _mm256_hadd_ps(tmp, tmp); __m128 tmp2 = _mm256_extractf128_ps (tmp,0);return _mm_cvtss_f32(tmp2); }
//	float Dot(const SSEx8 &src2) { __m256 tmp=_mm256_mul_ps(octo_floats,src2.octo_floats); tmp=_mm256_hadd_ps(tmp,tmp); tmp=_mm256_hadd_ps(tmp,tmp);float ret; _mm256_store_ss(&ret,tmp); return ret;}

  private:
	__m256 octo_floats;
};


SSEx8 SSEx8::Get(int index_prm)
{
	SSEx8 tmp;
	tmp.octo_floats=_mm256_sub_ps(octo_floats,octo_floats); //clear tmp to 0;
	switch (index_prm)
	{
		case 0:
			tmp.octo_floats=_mm256_blend_ps(tmp.octo_floats,octo_floats,1);
			break;
		case 1:
			tmp.octo_floats=_mm256_blend_ps(tmp.octo_floats,octo_floats,2);
			tmp.octo_floats=_mm256_permute_ps(tmp.octo_floats,1);
			break;
		case 2:
			tmp.octo_floats=_mm256_blend_ps(tmp.octo_floats,octo_floats,4);
			tmp.octo_floats=_mm256_permute_ps(tmp.octo_floats,2);
			break;
		case 3:
			tmp.octo_floats=_mm256_blend_ps(tmp.octo_floats,octo_floats,8);
			tmp.octo_floats=_mm256_permute_ps(tmp.octo_floats,3);
			break;
		case 4:
			tmp.octo_floats=_mm256_blend_ps(tmp.octo_floats,_mm256_castps128_ps256(_mm256_extractf128_ps(octo_floats,1)),1);

			break;
		case 5:
			tmp.octo_floats=_mm256_blend_ps(tmp.octo_floats,_mm256_castps128_ps256(_mm256_extractf128_ps(octo_floats,1)),2);
			tmp.octo_floats=_mm256_permute_ps(tmp.octo_floats,1);
			break;
		case 6:
			tmp.octo_floats=_mm256_blend_ps(tmp.octo_floats,_mm256_castps128_ps256(_mm256_extractf128_ps(octo_floats,1)),4);
			tmp.octo_floats=_mm256_permute_ps(tmp.octo_floats,2);
			break;
		case 7:
			tmp.octo_floats=_mm256_blend_ps(tmp.octo_floats,_mm256_castps128_ps256(_mm256_extractf128_ps(octo_floats,1)),8);
			tmp.octo_floats=_mm256_permute_ps(tmp.octo_floats,3);
			break;
	}
	return tmp;

}


extern __inline float __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm256_extract_ps (__m256 __X, const int __N)
{
	register __m128 __Y = _mm256_extractf128_ps (__X, __N >> 2);
	register float __RET;
	_MM_EXTRACT_FLOAT(__RET,__Y,__N % 4);
	return __RET;
//	return (float)_mm_extract_ps (__Y, __N % 4); //doesn't do what we want...
}

float SSEx8::operator[](int index_prm) const
{
	// _ISN'T_ this what compilers are for????????
	switch (index_prm)
	{
	  case 0:
		return _mm256_extract_ps (octo_floats, 0);
	  case 1:
		return _mm256_extract_ps (octo_floats, 1);
	  case 2:
		return _mm256_extract_ps (octo_floats, 2);
	  case 3:
		return _mm256_extract_ps (octo_floats, 3);
	  case 4:
		return _mm256_extract_ps (octo_floats, 4);
	  case 5:
		return _mm256_extract_ps (octo_floats, 5);
	  case 6:
		return _mm256_extract_ps (octo_floats, 6);
	  case 7:
		return _mm256_extract_ps (octo_floats, 7);
	}
	return 0.0; //dummy
}

// still ugly in AVX...
void SSEx8::Set(int index,float value)
{
	// this is sucky for now just store to memory and reload.
	// various incantations of shuffles can replace this
	float tmp[8];
	save(tmp);
	tmp[index]=value;
	octo_floats=_mm256_loadu_ps(tmp);

}

std::string SSEx8::as_str(const char *format_prm)
{
	char outstring[255];
	float x[8];
	save(x);
	sprintf(outstring,format_prm,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
	return std::string(outstring);
}

// global operators (overload float) for interoperability with float
SSEx8 operator+(const float &a,const SSEx8 &b)
{
	SSEx8 tmp(a,a,a,a,a,a,a,a);
	tmp+=b;
	return tmp;
}

SSEx8 operator-(const float &a,const SSEx8 &b)
{
	SSEx8 tmp(a,a,a,a,a,a,a,a);
	tmp-=b;
	return tmp;
}

SSEx8 operator*(const float &a,const SSEx8 &b)
{
	SSEx8 tmp(a,a,a,a,a,a,a,a);
	tmp*=b;
	return tmp;
}

SSEx8 operator/(const float &a,const SSEx8 &b)
{
	SSEx8 tmp(a,a,a,a,a,a,a,a);
	tmp/=b;
	return tmp;
}


// bogus class used for assignment
class SSEx8_field
{
	SSEx8 parent;
};


#endif //__AVX__

class alignas(32) SSEx4
{
  public:
	// initial construction
	SSEx4(){}; //construct unititalized
	SSEx4(float src) {quad_floats=_mm_load_ss(&src);}; //load single value, clear remaining
	SSEx4(float *src) {quad_floats=_mm_load_ps(src);}; //load 4 floats
	SSEx4(float a,float b,float c,float d) { float tmp[4]; tmp[0]=a;tmp[1]=b;tmp[2]=c;tmp[3]=d; quad_floats=_mm_load_ps(tmp); }; //load 4 floats (same as mm_set_ps)

	// copy constructor
	//SSEx4(const SSEx4 &src_prm) { quad_floats=_mm_shuffle_ps(src_prm.quad_floats,src_prm.quad_floats,_MM_SHUFFLE(3,2,1,0)); };

	// assignment
	SSEx4 & operator=(const SSEx4 &src_prm) { quad_floats=src_prm.quad_floats; return *this;}
	SSEx4 & operator=(const float x[4]) { quad_floats=_mm_load_ps(x); return *this;};
	void Set(int index,float value);

	// reads
	float operator[](int index_prm) const;
	SSEx4 Get(int index_prm);
	void save(float *dst_prm) { _mm_store_ps(dst_prm,quad_floats);};
	std::string as_str(const char *format_prm="%1.2f %1.2f %1.2f %1.2f");

	// adds urinary
	SSEx4 & operator+=(const SSEx4 &src2) { quad_floats=_mm_add_ps(quad_floats,src2.quad_floats); return *this;}
	SSEx4 & operator-=(const SSEx4 &src2) { quad_floats=_mm_sub_ps(quad_floats,src2.quad_floats); return *this;}
	SSEx4 & operator+=(const float src2) { __m128 tmp=_mm_load1_ps(&src2); quad_floats=_mm_add_ps(quad_floats,tmp); return *this;}
	SSEx4 & operator-=(const float src2) { __m128 tmp=_mm_load1_ps(&src2); quad_floats=_mm_sub_ps(quad_floats,tmp); return *this;}
	// adds binary
	SSEx4 const operator+ (const SSEx4 &src2) const { SSEx4 tmp; tmp.quad_floats=_mm_add_ps(quad_floats,src2.quad_floats); return tmp;}
	SSEx4 const operator- (const SSEx4 &src2) const { SSEx4 tmp; tmp.quad_floats=_mm_sub_ps(quad_floats,src2.quad_floats); return tmp;}
	SSEx4 const operator+ (const float src2)  const { SSEx4 tmp; tmp.quad_floats=_mm_load1_ps(&src2); tmp.quad_floats=_mm_add_ps(quad_floats,tmp.quad_floats); return tmp;}
	SSEx4 const operator- (const float src2)  const { SSEx4 tmp; tmp.quad_floats=_mm_load1_ps(&src2); tmp.quad_floats=_mm_sub_ps(quad_floats,tmp.quad_floats); return tmp;}


	// muls urinary
	SSEx4 & operator*=(const SSEx4 &src2) { quad_floats=_mm_mul_ps(quad_floats,src2.quad_floats); return *this;}
	SSEx4 & operator/=(const SSEx4 &src2) { quad_floats=_mm_div_ps(quad_floats,src2.quad_floats); return *this;}
	SSEx4 & operator*=(const float src2) { __m128 tmp=_mm_load1_ps(&src2); quad_floats=_mm_mul_ps(quad_floats,tmp); return *this;}
	SSEx4 & operator/=(const float src2) { __m128 tmp=_mm_load1_ps(&src2); quad_floats=_mm_div_ps(quad_floats,tmp); return *this;}
	// muls binary
	SSEx4 const operator* (const SSEx4 &src2) const { SSEx4 tmp; tmp.quad_floats=_mm_mul_ps(quad_floats,src2.quad_floats); return tmp;}
	SSEx4 const operator/ (const SSEx4 &src2) const { SSEx4 tmp; tmp.quad_floats=_mm_div_ps(quad_floats,src2.quad_floats); return tmp;}
	SSEx4 const operator* (const float src2)  const { SSEx4 tmp; tmp.quad_floats=_mm_load1_ps(&src2); tmp.quad_floats=_mm_mul_ps(quad_floats,tmp.quad_floats); return tmp;}
	SSEx4 const operator/ (const float src2)  const { SSEx4 tmp; tmp.quad_floats=_mm_load1_ps(&src2); tmp.quad_floats=_mm_div_ps(quad_floats,tmp.quad_floats); return tmp;}


	// other manipulations
	float ElementSum(void) const { float ret; __m128 tmp=_mm_hadd_ps(quad_floats,quad_floats); tmp=_mm_hadd_ps(tmp,tmp); _mm_store_ss(&ret,tmp); return ret; }
	float Dot(const SSEx4 &src2) { __m128 tmp=_mm_mul_ps(quad_floats,src2.quad_floats); tmp=_mm_hadd_ps(tmp,tmp); tmp=_mm_hadd_ps(tmp,tmp);float ret; _mm_store_ss(&ret,tmp); return ret;}
//	float Dot(const float &src2) { // use the extract notation to create a vector with only the lower element set? otherwise we end up with a load }

  private:
	__m128 quad_floats;
	friend class vMatrix<4>;
};

SSEx4 SSEx4::Get(int index_prm)
{
	SSEx4 tmp;
	tmp.quad_floats=_mm_sub_ps(quad_floats,quad_floats); //clear tmp to 0;
	switch (index_prm)
	{
		case 0:
			tmp.quad_floats=_mm_unpacklo_ps(quad_floats,tmp.quad_floats);
			tmp.quad_floats=_mm_shuffle_ps(tmp.quad_floats,tmp.quad_floats,_MM_SHUFFLE(1,1,1,0));
			break;
		case 1:
			tmp.quad_floats=_mm_unpacklo_ps(quad_floats,tmp.quad_floats);
			tmp.quad_floats=_mm_shuffle_ps(tmp.quad_floats,tmp.quad_floats,_MM_SHUFFLE(1,1,1,2));
			break;
		case 2:
			tmp.quad_floats=_mm_unpackhi_ps(quad_floats,tmp.quad_floats);
			tmp.quad_floats=_mm_shuffle_ps(tmp.quad_floats,tmp.quad_floats,_MM_SHUFFLE(1,1,1,0));
			break;
		case 3:
			tmp.quad_floats=_mm_unpackhi_ps(quad_floats,tmp.quad_floats);
			tmp.quad_floats=_mm_shuffle_ps(tmp.quad_floats,tmp.quad_floats,_MM_SHUFFLE(1,1,1,2));
			break;
	}
	return tmp;
}

float SSEx4::operator[](int index_prm) const
{
	register float ret;
	//(could be _mm_extract_ps(quad_floats,index_prm) for sse4) (but that rounds to int!)
//	switch (index_prm&0x03) // don't pass index >3
//#ifdef  __SSE4_1__

	// Its quite possible that its faster to just call get...
	switch (index_prm)
	{
		case 0:
			ret=_mm_cvtss_f32(_mm_shuffle_ps(quad_floats,quad_floats,_MM_SHUFFLE(0,0,0,0)));
			break;
		case 1:
			ret=_mm_cvtss_f32(_mm_shuffle_ps(quad_floats,quad_floats,_MM_SHUFFLE(0,0,0,1)));
			break;
		case 2:
			ret=_mm_cvtss_f32(_mm_shuffle_ps(quad_floats,quad_floats,_MM_SHUFFLE(0,0,0,2)));
			break;
		case 3:
			ret=_mm_cvtss_f32(_mm_shuffle_ps(quad_floats,quad_floats,_MM_SHUFFLE(0,0,0,3)));
			break;
	}
#ifdef NONE
	switch (index_prm)
	{
		case 0:
			_mm_store_ss(&ret,_mm_shuffle_ps(quad_floats,quad_floats,_MM_SHUFFLE(0,0,0,0)));
			break;
		case 1:
			_mm_store_ss(&ret,_mm_shuffle_ps(quad_floats,quad_floats,_MM_SHUFFLE(0,0,0,1)));
			break;
		case 2:
			_mm_store_ss(&ret,_mm_shuffle_ps(quad_floats,quad_floats,_MM_SHUFFLE(0,0,0,2)));
			break;
		case 3:
			_mm_store_ss(&ret,_mm_shuffle_ps(quad_floats,quad_floats,_MM_SHUFFLE(0,0,0,3)));
			break;
	}
#endif
	return ret;
}

// and intel invented sse4 (_mm_blend_ps for this too)
// before 4 you have very ugly code to do simple things like this.
void SSEx4::Set(int index,float value)
{
	if (index==0)
	{
		__m128 tmp;
		tmp=_mm_load_ss(&value);
		quad_floats=_mm_move_ss(quad_floats,tmp);
	}
	else
	{
		// this is sucky for now just store to memory and reload.
		// various incantations of shuffles can replace this
		float tmp[4];
		save(tmp);
		tmp[index]=value;
		quad_floats=_mm_load_ps(tmp);
	}
}

std::string SSEx4::as_str(const char *format_prm)
{
	char outstring[255];
	float x[4];
	save(x);
	snprintf(outstring,254,format_prm,x[0],x[1],x[2],x[3]);
	outstring[254]=0;
	return std::string(outstring);
}

// global operators (overload float) for interoperability with float
SSEx4 operator+(const float &a,const SSEx4 &b)
{
	SSEx4 tmp(a,a,a,a);
	tmp+=b;
	return tmp;
}

SSEx4 operator-(const float &a,const SSEx4 &b)
{
	SSEx4 tmp(a,a,a,a);
	tmp-=b;
	return tmp;
}

SSEx4 operator*(const float &a,const SSEx4 &b)
{
	SSEx4 tmp(a,a,a,a);
	tmp*=b;
	return tmp;
}

SSEx4 operator/(const float &a,const SSEx4 &b)
{
	SSEx4 tmp(a,a,a,a);
	tmp/=b;
	return tmp;
}

typedef SSEx4 Vec4;
#include "matrix.cpp"

// bogus class used for assignment
class SSEx4_field
{
	SSEx4 parent;
};


#ifndef NO_MAIN

#include <sys/time.h>

#ifdef __AVX__
int main(int argc,char *argv[])
{
	int xc,xd;
	struct timeval start,end, toffset;
	float y[8]={1,2,3,4,5,6,7,8};
	SSEx8 x(y);
	SSEx8 z(5);
//	SSEx4 w= (float *) {7.0,8.0,9.0,0.0};
//	SSEx4 w= y;
	SSEx8 w(7,8,9,10,11,12,13,14);
	SSEx8 v;
	v.Set(0,20);
	v.Set(1,21);
	v.Set(2,22);
	v.Set(3,23);
	v.Set(4,24);
	v.Set(5,25);
	v.Set(6,26);
	v.Set(7,27);
	SSEx8 u=v+z;
	SSEx8 t=y;
	SSEx8 s=t;

	printf("sizeof SSEx8=%d\n",sizeof(x));
	printf("x=%s\n",x.as_str().c_str());
	printf("z=%s\n",z.as_str().c_str());
	printf("w=%s\n",w.as_str().c_str());
	printf("v=%s\n",v.as_str().c_str());
	printf("u=(v+z)=%s\n",u.as_str().c_str());
	printf("t=%s\n",t.as_str().c_str());
	printf("s=%s\n",s.as_str().c_str());
	printf("%1.2f\n",x[0]);
	printf("%1.2f\n",x[1]);
	printf("%1.2f\n",x[2]);
	printf("%1.2f\n",x[3]);
	printf("%1.2f\n",x[4]);
	printf("%1.2f\n",x[5]);
	printf("%1.2f\n",x[6]);
	printf("%1.2f\n",x[7]);
	printf("%s\n",x.as_str().c_str());
	x+=v;
	printf("x+=v %s\n",x.as_str().c_str());
	x-=v;
	printf("x-=v %s\n",x.as_str().c_str());
	x+=5;
	printf("x+=5 %s\n",x.as_str().c_str());
	x*=10;
	printf("x*=10 %s\n",x.as_str().c_str());
	printf("x.ElementSum()=%1.2f\n",x.ElementSum());

	SSEx8 lowb=x.Get(0);
	printf("lowb[0]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(1);
	printf("lowb[1]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(2);
	printf("lowb[2]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(3);
	printf("lowb[3]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(4);
	printf("lowb[4]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(5);
	printf("lowb[5]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(6);
	printf("lowb[6]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(7);
	printf("lowb[7]=%s\n",lowb.as_str().c_str());

	SSEx8 a=30.0+x;

	printf("30+x=%s\n",a.as_str().c_str());
	printf("x=%s\n",x.as_str().c_str());
	a=x+30;
	printf("30+x=%s\n",a.as_str().c_str());
	printf("x=%s\n",x.as_str().c_str());

	int items=1024*1024;
	SSEx8 *b_init=new SSEx8[items+1];

	// 32byte align the array
	SSEx8 *b=(SSEx8*)((((unsigned long long)&b_init[1])/sizeof(SSEx8))*sizeof(SSEx8));
	for (xc=0;xc<items;xc++)
	{
		SSEx8 ass(rand(),rand(),rand(),rand(),rand(),rand(),rand(),rand());
		b[xc]=ass;
	}



	gettimeofday(&start,NULL);

	for (xd=0;xd<100;xd++)
	{
		for (xc=0;xc<items;xc++)
		{
			b[xc]*=b[items-xc];
			b[items-xc]+=b[xc];
		}
	}

	gettimeofday(&end,NULL);
	timersub(&end,&start,&toffset);
	{

		float totalsecs=toffset.tv_sec+toffset.tv_usec/1000000.0;
		printf("Did %d ops in in %f seconds\n",xd*xc*16,totalsecs);
	}
/*	for (int xc=0;xc<items;xc++)
	{
		printf("b[%d]=%s\n",xc,b[xc].as_str().c_str());
	}*/
	delete[] b_init;


}

#else
int main(int argc,char *argv[])
{
	int xc,xd;
	struct timeval start,end, toffset;
	float y[4]={1,2,3,4};
	SSEx4 x(y);
	SSEx4 z(5);
//	SSEx4 w= (float *) {7.0,8.0,9.0,0.0};
//	SSEx4 w= y;
	SSEx4 w(7,8,9,10);
	SSEx4 v;
	v.Set(0,20);
	v.Set(1,21);
	v.Set(2,22);
	v.Set(3,23);
	SSEx4 u=v+z;
	SSEx4 t=y;
	SSEx4 s=t;

	printf("sizeof SSEx4=%d\n",sizeof(x));

	printf("x=%s\n",x.as_str().c_str());
	printf("w=%s\n",w.as_str().c_str());
	printf("v=%s\n",v.as_str().c_str());
	printf("u=%s\n",u.as_str().c_str());
	printf("t=%s\n",t.as_str().c_str());
	printf("s=%s\n",s.as_str().c_str());
	printf("%1.2f\n",x[0]);
	printf("%1.2f\n",x[1]);
	printf("%1.2f\n",x[2]);
	printf("%1.2f\n",x[3]);
	printf("%s\n",x.as_str().c_str());
	x+=v;
	printf("x+=v %s\n",x.as_str().c_str());
	x-=v;
	printf("x-=v %s\n",x.as_str().c_str());
	x+=5;
	printf("x+=5 %s\n",x.as_str().c_str());
	x*=10;
	printf("x*=10 %s\n",x.as_str().c_str());
	printf("x.ElementSum()=%1.2f\n",x.ElementSum());


	SSEx4 lowb=x.Get(0);
	printf("lowb[0]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(1);
	printf("lowb[1]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(2);
	printf("lowb[2]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(3);
	printf("lowb[3]=%s\n",lowb.as_str().c_str());

	SSEx4 a=30.0+x;

	printf("30+x=%s\n",a.as_str().c_str());
	printf("x=%s\n",x.as_str().c_str());
	a=x+30;
	printf("30+x=%s\n",a.as_str().c_str());
	printf("x=%s\n",x.as_str().c_str());

	int items=1024*1024;
	SSEx4 *b=new SSEx4[items];
	for (xc=0;xc<items;xc++)
	{
		SSEx4 ass(rand(),rand(),rand(),rand());
		b[xc]=ass;
	}

	gettimeofday(&start,NULL);

	for (xd=0;xd<1;xd++) //do the loop 100 times
	{
		for (xc=1;xc<items;xc++)
		{
			b[xc]*=b[items-xc];
			b[items-xc]+=b[xc];
		}
	}

	gettimeofday(&end,NULL);
	timersub(&end,&start,&toffset);
	{
		float totalsecs=toffset.tv_sec+toffset.tv_usec/1000000.0;
		printf("Did %d ops in in %f seconds\n",xd*xc*8,totalsecs);
	}

	printf("b[%d]=%s\n",0,b[0].as_str().c_str());
	printf("b[%d]=%s\n",1,b[1].as_str().c_str());
/*	for (int xc=0;xc<items;xc++)
	{
		printf("b[%d]=%s\n",xc,b[xc].as_str().c_str());
	}*/
	delete[] b;

	SSEx4Matrix TranslateX(SSEx4(1,0,0,10),SSEx4(0,1,0,0),SSEx4(0,0,1,0),SSEx4(0,0,0,1));
	SSEx4 Point(10,10,10,1);
	printf("New position %s\n",(TranslateX*Point).as_str().c_str());
	Translate tx(10,0,0);
	printf("New position %s\n",(tx*Point).as_str().c_str());
	Scale sa(2,2,2);
	printf("New scale %s\n",(sa*Point).as_str().c_str());


/* this should print
Matrix
  250.00   260.00	270.00	 280.00
  618.00   644.00	670.00	 696.00
  986.00  1028.00  1070.00	1112.00
 1354.00  1412.00  1470.00	1528.00
*/
	SSEx4Matrix MatM(SSEx4(1,2,3,4),SSEx4(5,6,7,8),SSEx4(9,10,11,12),SSEx4(13,14,15,16));
	SSEx4Matrix Mat2(SSEx4(17,18,19,20),SSEx4(21,22,23,24),SSEx4(25,26,27,28),SSEx4(29,30,31,32));
	printf("Matrix\n%s\n",MatM.as_str().c_str());
	printf("Matrix\n%s\n",Mat2.as_str().c_str());
	SSEx4Matrix Mat3=MatM*Mat2;
	printf("Matrix\n%s\n",Mat3.as_str("%8.2f %8.2f %8.2f %8.2f").c_str());

	RotateXRad rot(3.1415);
	printf("Matrix\n%s\n",rot.as_str("%8.2f %8.2f %8.2f %8.2f").c_str());

}
#endif

#endif //NO_MAIN
