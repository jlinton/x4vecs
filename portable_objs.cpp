// Copyright (c) 2012,2014 Jeremy Linton
//
// portable_objs2.cpp
//   C++ wrapper for a generic CPU target that mimmics
//   the SSE and NEON classes. 
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
//   g++ -g -O3 portable_objs.cpp

// This version just uses normal CPU instructions/autovectorization (if possible)
// rather than the intrinsics, its used sort of as a benchmark, and 
// for unsupported arches. 


// most of the methods here are written with direct array indexes rather than using a small loop
// this seems to help the old version of GCC this was initially written for, not sure about more recent versions which might
// find the loops utilizing a constant to be easier to optimize.. This should be benchmarked for a given compiler/CPU if anyone really cares
// definitely using a loop would allow a further template parameter specifying the number of elements in the vector. 

#include <string>
#include <stdio.h>

#define PORTABLE_VEC

template<class ELEM_T> class PORTx4
{
  public:
	// initial construction
	PORTx4(){}; //construct unititalized
	PORTx4(ELEM_T *src) {elements[0]=src[0];elements[1]=src[1]; elements[2]=src[2]; elements[3]=src[3];};
	PORTx4(ELEM_T a,ELEM_T b=0,ELEM_T c=0,ELEM_T d=0) { elements[0]=a;elements[1]=b; elements[2]=c; elements[3]=d; }; 

	// copy constructor
	//PORTx4<ELEM_T>(const PORTx4<ELEM_T> &src_prm) { quad_ELEM_Ts=_mm_shuffle_ps(src_prm.quad_ELEM_Ts,src_prm.quad_ELEM_Ts,_MM_SHUFFLE(3,2,1,0)); }; 

	// assignment
	PORTx4<ELEM_T> & operator=(const PORTx4<ELEM_T> &src_prm) { elements[0]=src_prm.elements[0];elements[1]=src_prm.elements[1]; elements[2]=src_prm.elements[2]; elements[3]=src_prm.elements[3]; }
	PORTx4<ELEM_T> & operator=(const ELEM_T x[4]) { elements[0]=x[0];elements[1]=x[1]; elements[2]=x[2]; elements[3]=x[3]; }; 
	void Set(int index,ELEM_T value) { elements[index]=value; }

	// reads 
	ELEM_T operator[](int index_prm) { return elements[index_prm]; }
	PORTx4<ELEM_T> Get(int index_prm) { PORTx4<ELEM_T> tmp; tmp.elements[0]=elements[index_prm]; tmp.elements[1]=0;  tmp.elements[2]=0;  tmp.elements[3]=0;  tmp.elements[4]=0; return tmp; }
	void save(ELEM_T *dst_prm) { dst_prm[0]=elements[0];dst_prm[1]=elements[1]; dst_prm[2]=elements[2]; dst_prm[3]=elements[3]; };
	std::string as_str(char *format_prm="%1.2f %1.2f %1.2f %1.2f")
	{
		char outstring[255];
		snprintf(outstring,254,format_prm,elements[0],elements[1],elements[2],elements[3]);
		outstring[254]=0;
		return std::string(outstring);
	}

	// adds urinary
	PORTx4<ELEM_T> & operator+=(const PORTx4<ELEM_T> &src2) { elements[0]=+src2.elements[0];elements[1]+=src2.elements[1]; elements[2]+=src2.elements[2]; elements[3]+=src2.elements[3]; return *this;}
	PORTx4<ELEM_T> & operator-=(const PORTx4<ELEM_T> &src2) { elements[0]=-src2.elements[0];elements[1]-=src2.elements[1]; elements[2]-=src2.elements[2]; elements[3]-=src2.elements[3]; return *this;}
	PORTx4<ELEM_T> & operator+=(const ELEM_T src2) { elements[0]=+src2;elements[1]+=src2; elements[2]+=src2; elements[3]+=src2; return *this;}
	PORTx4<ELEM_T> & operator-=(const ELEM_T src2) { elements[0]=-src2;elements[1]-=src2; elements[2]-=src2; elements[3]-=src2; return *this;}
	// adds binary
	PORTx4<ELEM_T> const operator+ (const PORTx4<ELEM_T> &src2) const { PORTx4<ELEM_T> tmp; tmp.elements[0]=elements[0]+src2.elements[0]; tmp.elements[1]=elements[1]+src2.elements[1]; tmp.elements[2]=elements[2]+src2.elements[2]; tmp.elements[3]=elements[3]+src2.elements[3]; return tmp;}
	PORTx4<ELEM_T> const operator- (const PORTx4<ELEM_T> &src2) const { PORTx4<ELEM_T> tmp; tmp.elements[0]=elements[0]-src2.elements[0]; tmp.elements[1]=elements[1]-src2.elements[1]; tmp.elements[2]=elements[2]-src2.elements[2]; tmp.elements[3]=elements[3]-src2.elements[3]; return tmp;}
	PORTx4<ELEM_T> const operator+ (const ELEM_T src2)  const { PORTx4<ELEM_T> tmp; tmp.elements[0]=elements[0]+src2; tmp.elements[1]=elements[1]+src2; tmp.elements[2]=elements[2]+src2; tmp.elements[3]=elements[3]+src2; return tmp;}
	PORTx4<ELEM_T> const operator- (const ELEM_T src2)  const { PORTx4<ELEM_T> tmp; tmp.elements[0]=elements[0]-src2; tmp.elements[1]=elements[1]-src2; tmp.elements[2]=elements[2]-src2; tmp.elements[3]=elements[3]-src2;  return tmp;}


	// muls urinary
	PORTx4<ELEM_T> & operator*=(const PORTx4<ELEM_T> &src2) { elements[0]=*src2.elements[0]; elements[1]*=src2.elements[1]; elements[2]*=src2.elements[2]; elements[3]*=src2.elements[3]; return *this; }
	PORTx4<ELEM_T> & operator/=(const PORTx4<ELEM_T> &src2) { elements[0]/=src2.elements[0]; elements[1]/=src2.elements[1]; elements[2]/=src2.elements[2]; elements[3]/=src2.elements[3]; return *this; }
	PORTx4<ELEM_T> & operator*=(const ELEM_T src2) { elements[0]*=src2;elements[1]*=src2; elements[2]*=src2; elements[3]*=src2; return *this;}
	PORTx4<ELEM_T> & operator/=(const ELEM_T src2) { elements[0]/=src2;elements[1]/=src2; elements[2]/=src2; elements[3]/=src2; return *this;}
	// muls binary
	PORTx4<ELEM_T> const operator* (const PORTx4<ELEM_T> &src2) const { PORTx4<ELEM_T> tmp; tmp.elements[0]=elements[0]*src2.elements[0]; tmp.elements[1]=elements[1]*src2.elements[1]; tmp.elements[2]=elements[2]*src2.elements[2]; tmp.elements[3]=elements[3]*src2.elements[3]; return tmp;}
	PORTx4<ELEM_T> const operator/ (const PORTx4<ELEM_T> &src2) const { PORTx4<ELEM_T> tmp; tmp.elements[0]=elements[0]/src2.elements[0]; tmp.elements[1]=elements[1]/src2.elements[1]; tmp.elements[2]=elements[2]/src2.elements[2]; tmp.elements[3]=elements[3]/src2.elements[3]; return tmp;}
	PORTx4<ELEM_T> const operator* (const ELEM_T src2)  const { PORTx4<ELEM_T> tmp; tmp.elements[0]=elements[0]*src2; tmp.elements[1]=elements[1]*src2; tmp.elements[2]=elements[2]*src2; tmp.elements[3]=elements[3]*src2; return tmp;}
	PORTx4<ELEM_T> const operator/ (const ELEM_T src2)  const { PORTx4<ELEM_T> tmp; tmp.elements[0]=elements[0]/src2; tmp.elements[1]=elements[1]/src2; tmp.elements[2]=elements[2]/src2; tmp.elements[3]=elements[3]/src2; return tmp;}


	// other manipulations
	ELEM_T ElementSum(void) const { return elements[0]+elements[1]+elements[2]+elements[3]; }
	ELEM_T Dot(const PORTx4<ELEM_T> &src2) { return elements[0]*src2.elements[0]+elements[1]*src2.elements[1]+elements[2]*src2.elements[2]+elements[3]*src2.elements[3]; }

  private:
	 ELEM_T elements[4];
     friend class SSEx4Matrix;
};


// global operators (overload float)
template<class ELEM_T> PORTx4<ELEM_T> operator+(const float &a,const PORTx4<ELEM_T> &b)
{
	PORTx4<ELEM_T> tmp(a,a,a,a);
	tmp+=b;
	return tmp;
}

template<class ELEM_T> PORTx4<ELEM_T> operator-(const float &a,const PORTx4<ELEM_T> &b)
{
	PORTx4<ELEM_T> tmp(a,a,a,a);
	tmp-=b;
	return tmp;
}

template<class ELEM_T> PORTx4<ELEM_T> operator*(const float &a,const PORTx4<ELEM_T> &b)
{
	PORTx4<ELEM_T> tmp(a,a,a,a);
	tmp*=b;
	return tmp;
}

template<class ELEM_T> PORTx4<ELEM_T> operator/(const float &a,const PORTx4<ELEM_T> &b)
{
	PORTx4<ELEM_T> tmp(a,a,a,a);
	tmp/=b;
	return tmp;
}

typedef PORTx4<float> Vec4;

// Unit test here...
#ifndef NO_MAIN

#include <sys/time.h>
#include <stdlib.h>
#include "matrix.cpp"
int main(int argc,char *argv[])
{
	int xc,xd;
	struct timeval start,end, toffset;
	printf("Native testing\n");

	float y[4]={1,2,3,4};
    Vec4 x(y);
    Vec4 z(5);
//	Vec4 w= (float *) {7.0,8.0,9.0,0.0};
//	Vec4 w= y;
    Vec4 w(7,8,9,10);
    Vec4 v;
	v.Set(0,20);
	v.Set(1,21);
	v.Set(2,22);
	v.Set(3,23);
    Vec4 u=v+z;
    Vec4 t=y;
    Vec4 s=t;
	
	printf("sizeof NATIVx4=%d\n",sizeof(x));

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


    Vec4 lowb=x.Get(0);
	printf("lowb[0]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(1);
	printf("lowb[1]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(2);
	printf("lowb[2]=%s\n",lowb.as_str().c_str());
	lowb=x.Get(3);
	printf("lowb[3]=%s\n",lowb.as_str().c_str());

    Vec4 a=30.0+x;
	
	printf("30+x=%s\n",a.as_str().c_str());
	printf("x=%s\n",x.as_str().c_str());
	a=x+30;
	printf("30+x=%s\n",a.as_str().c_str());
	printf("x=%s\n",x.as_str().c_str());

	int items=1024*1024;
    Vec4 *b=new Vec4[items];
	for (xc=0;xc<items;xc++)
	{
        Vec4 ass(rand(),rand(),rand(),rand());
		b[xc]=ass;
	}

	gettimeofday(&start,NULL);

	for (xd=0;xd<1;xd++) //do the loop 100 times
	{
		for (xc=1;xc<items;xc++)
		{
			//b[xc]*=b[items-xc];
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

    SSEx4Matrix MatM(Vec4(1,2,3,4),Vec4(5,6,7,8),Vec4(9,10,11,12),Vec4(13,14,15,16));
    printf("Matrix\n%s\n",MatM.as_str().c_str());
    MatM.Transpose();
    printf("Matrix Transpose\n%s\n",MatM.as_str().c_str());

}

#endif
