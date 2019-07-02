// Copyright (c) 2012,2014 Jeremy Linton
//
// neon_objs.cpp
//   C++ wrapper around the ARM NEON x4 float vector intrinsics
//   with a matching class for SSE and a "portable" implementation
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




// To build some basic unit tests for this module do:
//   g++ -mfloat-abi=softfp -mfpu=neon -g -O3 neon_objs.cpp
//
// seems there is a generation problem?
// the alignment prefix from the compiler appears to be using a : when it should be using a @
// hack up the assembly output from the compile phase, then assemble and link
// /usr/local/gcc48/bin/g++ -mfloat-abi=softfp -mfpu=neon -O3 -g neon_objs.cpp -S; sed -i "s/\(\[r[0-9]\):64/\\1/g" neon_objs.s;sed -i "s/\(\[ip\):64/\\1/g" neon_objs.s; as neon_objs.s -o neon_objs.o ; g++ -g neon_objs.o
// This is obviously a custom built ARM version of the release gcc 4.8.3 


#pragma GCC diagnostic ignored "-Wwrite-strings"

#include <string>
#include <stdio.h>

#ifdef __ARM_NEON
#define __ARM_NEON__
#endif

#ifdef __ARM_NEON__
#pragma message "Building ARM NEON"
#include <arm_neon.h>

class NEONx4
{
  public:
	NEONx4(){}; //construct uninitialized
	NEONx4(float src) { quad_floats=vmovq_n_f32(0.0);  quad_floats=vsetq_lane_f32(src, quad_floats, 0); } //load single value, clear remaining
	NEONx4(float *src) {quad_floats=vld1q_f32(src); }
	NEONx4(float a,float b,float c,float d) { quad_floats=vld1q_lane_f32(&a,quad_floats, 0); quad_floats=vld1q_lane_f32(&b,quad_floats, 1); quad_floats=vld1q_lane_f32(&c,quad_floats, 2); quad_floats=vld1q_lane_f32(&d,quad_floats, 3); }

	// assignment
	NEONx4 & operator=(const NEONx4 &src_prm) {quad_floats=src_prm.quad_floats; return *this;}
	NEONx4 & operator=(const float x[4]) {quad_floats=vld1q_f32(x); return *this;}
	void Set(const int index,float value) 
	{ 
		switch (index)
		{
		case 0:
			quad_floats=vld1q_lane_f32 (&value,quad_floats, 0); 
			break;
		case 1:
			quad_floats=vld1q_lane_f32 (&value,quad_floats, 1); 
			break;
		case 2:
			quad_floats=vld1q_lane_f32 (&value,quad_floats, 2); 
			break;
		case 3:
			quad_floats=vld1q_lane_f32 (&value,quad_floats, 3); 
			break;
		}
	}

	// reads
	//float operator[](int index_prm) { float ret; vst1q_lane_f32 (&ret, quad_floats,index_prm); return ret;}
	float operator[](const int index_prm) 
	{
		switch (index_prm)
		{
			case 0:	
				return vgetq_lane_f32(quad_floats,0);
			case 1:	
				return vgetq_lane_f32(quad_floats,1);
			case 2:	
				return vgetq_lane_f32(quad_floats,2);
			case 3:	
				return vgetq_lane_f32(quad_floats,3);
		}
		return 0.0;
	}
	//return indexed value as element 0 in new vector
	NEONx4 Get(const int index_prm)
	{	
		float tmp;
		NEONx4 ret;
		// It would seem that all the insert/store/fun/etc in neon
		// would make this easy, but its still a PITA for an variable index
		switch (index_prm)
		{
		  case 0:		
			tmp=vgetq_lane_f32(quad_floats,0);
			break;
		  case 1:
			tmp=vgetq_lane_f32(quad_floats,1);
			break;
		  case 2:
			tmp=vgetq_lane_f32(quad_floats,2);
			break;
		  case 3:
			tmp=vgetq_lane_f32(quad_floats,3);
			break;
				
		}
		ret.quad_floats=vsetq_lane_f32(tmp,ret.quad_floats, 0);
		return ret;
	}

	void save(float *dst_prm) {  vst1q_f32 (dst_prm, quad_floats); }
	std::string as_str(char *format_prm="%1.2f %1.2f %1.2f %1.2f");

	// adds urinary
	NEONx4 & operator+=(const NEONx4 &src2) { quad_floats=vaddq_f32 (quad_floats, src2.quad_floats); return *this;} 
	NEONx4 & operator-=(const NEONx4 &src2) { quad_floats=vsubq_f32 (quad_floats, src2.quad_floats); return *this;} 
	NEONx4 & operator+=(const float src2)   { quad_floats=vaddq_f32 (quad_floats, vdupq_n_f32(src2)); return *this;} 
	NEONx4 & operator-=(const float src2)   { quad_floats=vsubq_f32 (quad_floats, vdupq_n_f32(src2)); return *this;} 
	// adds binary
	NEONx4 const operator+ (const NEONx4 &src2) const { NEONx4 tmp; tmp.quad_floats=vaddq_f32(quad_floats, src2.quad_floats); return tmp; }
	NEONx4 const operator- (const NEONx4 &src2) const { NEONx4 tmp; tmp.quad_floats=vsubq_f32(quad_floats, src2.quad_floats); return tmp; }
	NEONx4 const operator+ (const float src2) const { NEONx4 tmp; tmp.quad_floats=vaddq_f32(quad_floats, vdupq_n_f32(src2)); return tmp; }
	NEONx4 const operator- (const float src2) const { NEONx4 tmp; tmp.quad_floats=vsubq_f32(quad_floats, vdupq_n_f32(src2)); return tmp; }

	// muls urinary
	NEONx4 & operator*=(const NEONx4 &src2) { quad_floats=vmulq_f32 (quad_floats, src2.quad_floats); return *this;} 
	NEONx4 & operator/=(const NEONx4 &src2) { quad_floats=vmulq_f32 (quad_floats, vrecpeq_f32 (src2.quad_floats)); return *this;} 
	NEONx4 & operator*=(const float src2) { quad_floats=vmulq_f32 (quad_floats, vdupq_n_f32(src2)); return *this;} 
	NEONx4 & operator/=(const float src2) { quad_floats=vmulq_f32 (quad_floats, vrecpeq_f32 (vdupq_n_f32(src2))); return *this;} 
	
	// muls binary
	NEONx4 const operator* (const NEONx4 &src2) const { NEONx4 tmp; tmp.quad_floats=vmulq_f32(quad_floats,src2.quad_floats); return tmp;}
	NEONx4 const operator/ (const NEONx4 &src2) const { NEONx4 tmp; tmp.quad_floats=vmulq_f32(quad_floats,vrecpeq_f32(src2.quad_floats)); return tmp;}
	NEONx4 const operator* (const float src2)  const { NEONx4 tmp;  tmp.quad_floats=vmulq_f32(quad_floats,vdupq_n_f32(src2)); return tmp;}
	NEONx4 const operator/ (const float src2)  const { NEONx4 tmp;  tmp.quad_floats=vmulq_f32(quad_floats,vrecpeq_f32(vdupq_n_f32(src2))); return tmp;}

	// other
    float ElementSum(void) const { float32x2_t tmp = vadd_f32(vget_high_f32(quad_floats), vget_low_f32(quad_floats)); return vget_lane_f32(vpadd_f32(tmp, tmp), 0); }
	float Dot(const NEONx4 &src2) {	float32x4_t prod = vmulq_f32(quad_floats, src2.quad_floats); float32x4_t sum = vaddq_f32(prod, vrev64q_f32(prod));	return vgetq_lane_f32(vaddq_f32(sum, vcombine_f32(vget_high_f32(sum), vget_low_f32(sum))),0); }
  private:
	float32x4_t quad_floats;
    friend class SSEx4Matrix;
};


std::string NEONx4::as_str(char *format_prm)
{
	char outstring[255];
	float x[4];
	save(x);
	snprintf(outstring,254,format_prm,x[0],x[1],x[2],x[3]);
	outstring[254]=0;
	return std::string(outstring);
}

// Here are the global operators for interoperability with float
static inline NEONx4 operator+(const float &a,const NEONx4 &b)
{
	NEONx4 tmp(a,a,a,a);
	tmp+=b;
	return tmp;
}

static inline NEONx4 operator-(const float &a,const NEONx4 &b)
{
	NEONx4 tmp(a,a,a,a);
	tmp-=b;
	return tmp;
}

static inline NEONx4 operator*(const float &a,const NEONx4 &b)
{
	NEONx4 tmp(a,a,a,a);
	tmp*=b;
	return tmp;
}

static inline NEONx4 operator/(const float &a,const NEONx4 &b)
{
	NEONx4 tmp(a,a,a,a);
	tmp/=b;
	return tmp;
}


#else //!__ARM_NEON__
#pragma message "NEON not being built"
#endif //__ARM_NEON__

typedef NEONx4 Vec4;
#include "matrix.cpp"


// Unit test here...
#ifndef NO_MAIN

#include <sys/time.h>
#include <stdlib.h>
int main(int argc,char *argv[])
{
	int xc,xd;
	struct timeval start,end, toffset;
	printf("Neon testing\n");

	float y[4]={1,2,3,4};
	NEONx4 x(y);
	NEONx4 z(5);
//	NEONx4 w= (float *) {7.0,8.0,9.0,0.0};
//	NEONx4 w= y;
	NEONx4 w(7,8,9,10);
	NEONx4 v;
	v.Set(0,20);
	v.Set(1,21);
	v.Set(2,22);
	v.Set(3,23);
	NEONx4 u=v+z;
	NEONx4 t=y;
	NEONx4 s=t;
	
	printf("sizeof NEONx4=%d\n",sizeof(x));

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


	NEONx4 lowb=x.Get(0);
	printf("lowb[0]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(1);
	printf("lowb[1]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(2);
	printf("lowb[2]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(3);
	printf("lowb[3]=%s\n",lowb.as_str().c_str());

	NEONx4 a=30.0+x;
	
	printf("30+x=%s\n",a.as_str().c_str());
	printf("x=%s\n",x.as_str().c_str());
	a=x+30;
	printf("30+x=%s\n",a.as_str().c_str());
	printf("x=%s\n",x.as_str().c_str());

	int items=1024*1024;
	NEONx4 *b=new NEONx4[items];
	for (xc=0;xc<items;xc++)
	{
		NEONx4 ass(rand(),rand(),rand(),rand());
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

	delete[] b;


    SSEx4Matrix MatM(NEONx4(1,2,3,4),NEONx4(5,6,7,8),NEONx4(9,10,11,12),NEONx4(13,14,15,16));
    printf("Matrix\n%s\n",MatM.as_str().c_str());
    MatM.Transpose();
    printf("Matrix Transpose\n%s\n",MatM.as_str().c_str());
	return 0;
}

#endif
