// Copyright (c) 2021 Jeremy Linton
//
// big_vec.cpp
//	 C++ wrapper around the $PLAT_objs.cpp vector to extend its len
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
//	 g++ -mfloat-abi=softfp -mfpu=neon -g -O3 -std=c++11 big_vec.cpp
// or
//	  g++ -msse3 -g -O3 -std=c++11 big_vec.cpp
// AVXx8 version
//	  g++ -mavx -g -O3 -std=c++11 big_vec.cpp

#define NO_MAIN
#ifdef __ARM_NEON__
#include "neon_objs.cpp"
#elif defined(__SSE3__)
#include "sse_objs2.cpp"
#else
#include "portable_objs.cpp"
#endif

// big allocs might need something like	 -Wl,-z,stack-size=XXXX; ulimit -s XXXXX

// define arbitrary len vector in relation to HW vector class
//
#define VECS ((ELEM_CNT/VEC_CNT)?(ELEM_CNT/VEC_CNT):1)

template<int VEC_CNT, int ELEM_CNT, class VEC_T> class BigVector
{
  public:
	BigVector():vec() {}; //construct unititalized
	BigVector(BigVector *src) { for (int x=0;x<VECS;x++) vec[x]=src->vec[x]; };
	BigVector(BigVector &src) { for (int x=0;x<VECS;x++) vec[x]=src.vec[x]; };
	BigVector(VEC_T *src,int src_cnt) { VEC_T tmp(0.0); for (int x=0;x<VECS;x++) vec[x]=tmp; for (int x=0;x<src_cnt;x++) vec[x]=src[x]; }; //allows a resize op
	//BigVector(double *fill) {VEC_T tmp(fill); for (int x=0;x<VECS;x++) vec[x]=tmp;}
	template <typename... Args> BigVector(Args... args) {VEC_T tmp(0.0); for (int x=0;x<VECS;x++) vec[x]=tmp; Set(0, args...); }; // if this aborts, verify vec is properly aligned (32 bytes for x8)

	// POD copy construct

	// assign
	BigVector & operator=(const BigVector &src_prm) { for (int x=0;x<VECS;x++) vec[x]=src_prm.vec[x]; return *this;}
	BigVector & operator=(const float src[ELEM_CNT]) { for (int x=0;x<ELEM_CNT;x++) vec[x/VEC_CNT].Set(x-(x/VEC_CNT)*VEC_CNT,src[x]); return *this;};
	void Set(int off, float Val) {const int n=off/VEC_CNT; vec[n].Set(off-n*VEC_CNT, Val); }
	void Fill(float Val) {VEC_T tmp(Val); for (int x=1;x<VEC_CNT;x++) tmp.Set(x,Val); for (int x=0;x<VECS;x++) vec[x]=tmp;}
	template <typename... Args> void Set(int off, float Val, Args... args) { int n=off/VEC_CNT; vec[n].Set(off-n*VEC_CNT, Val); Set(++off,args...); };

	// reads
	float operator[](int index_prm) const {int n=index_prm/VEC_CNT; return vec[n][index_prm-n*VEC_CNT];};
	//BigVector Get(int index_prm);
	//void save(float *dst_prm) { };
	std::string as_str(const char *format_prm="%05.2f ") { std::string fmt; for (int x=0;x<VEC_CNT;x++) fmt+=format_prm; std::string tmp; for (int x=0;x<VECS;x++) tmp+=vec[x].as_str(fmt.c_str()); return tmp; }

	// adds urinary
	BigVector & operator+=(const BigVector &src2) { for (int x=0;x<VECS;x++) vec[x]+=src2.vec[x]; return *this;}
	BigVector & operator-=(const BigVector &src2) { for (int x=0;x<VECS;x++) vec[x]-=src2.vec[x]; return *this;}
	BigVector & operator+=(const float src2) { for (int x=0;x<VECS;x++) vec[x]+=src2; return *this;}
	BigVector & operator-=(const float src2) { for (int x=0;x<VECS;x++) vec[x]-=src2;; return *this;}
	// adds binary
	BigVector const operator+ (const BigVector &src2) { BigVector tmp; for (int x=0;x<VECS;x++) tmp.vec[x]=vec[x]+src2.vec[x]; return tmp;}
	BigVector const operator- (const BigVector &src2) { BigVector tmp; for (int x=0;x<VECS;x++) tmp.vec[x]=vec[x]-src2.vec[x]; return tmp;}
	BigVector const operator+ (const float src2)  { BigVector tmp; for (int x=0;x<VECS;x++) tmp.vec[x]=vec[x]+src2; return tmp;}
	BigVector const operator- (const float src2)  { BigVector tmp; for (int x=0;x<VECS;x++) tmp.vec[x]=vec[x]-src2; return tmp;}

	// muls urinary
	BigVector & operator*=(const BigVector &src2) { for (int x=0;x<VECS;x++) vec[x]*src2.vec[x]; return *this;}
	BigVector & operator/=(const BigVector &src2) { for (int x=0;x<VECS;x++) vec[x]/src2.vec[x]; return *this;}
	BigVector & operator*=(const float src2) { for (int x=0;x<VECS;x++) vec[x]=vec[x]*src2; return *this;}
	BigVector & operator/=(const float src2) { for (int x=0;x<VECS;x++) vec[x]=vec[x]/src2; return *this;}
	// muls binary
	BigVector const operator* (const BigVector &src2) { BigVector tmp; for (int x=0;x<VECS;x++) tmp.vec[x]=vec[x]*src2.vec[x]; return tmp;}
	BigVector const operator/ (const BigVector &src2) { BigVector tmp; for (int x=0;x<VECS;x++) tmp.vec[x]=vec[x]/src2.vec[x]; return tmp;}
	BigVector const operator* (const float src2) { BigVector tmp; for (int x=0;x<VECS;x++) tmp.vec[x]=vec[x]*src2; return tmp;}
	BigVector const operator/ (const float src2) { BigVector tmp; for (int x=0;x<VECS;x++) tmp.vec[x]=vec[x]/src2; return tmp;}


	// other manipulations
	float ElementSum(void) const  { float tmp=0; for (int x=0;x<VECS;x++) tmp+=vec[x].ElementSum(); return tmp; }
	float fma(BigVector &src2) {return 0.0;}//implement fma (which requires the underlying vector to do it too)
	VEC_T *raw() {return vec;}
	int raw_sz() {return VECS;}


  private:
	VEC_T vec[VECS];
};

// square matrix based on big vecs above
// most of these loops should be expanded with openMP if the matrix sizes get sufficiently large
template<int SIZE, class VEC_T> class MatrixS
{
  public:
	MatrixS() {} //construct unititalized
	MatrixS(float val) { VEC_T tmp; tmp.Fill(val); for (int x=0;x<SIZE;x++) vecs[x]=tmp; }
	MatrixS(const MatrixS &orig) {for (int x=0; x<SIZE; x++) vecs[x]=orig.vecs[x]; }

	// assign
	//BigVector & operator=(const BigVector &src_prm) { for (int x=0;x<VECS;x++) vec[x]=src_prm.vec[x]; }
	//BigVector & operator=(const float src[ELEM_CNT]) { for (int x=0;x<ELEM_CNT;x++) vec[x/VEC_CNT].Set(x-(x/VEC_CNT)*VEC_CNT,src[x]);};
	void Set(int col, int row, float Val) { vecs[row].Set(col, Val); }
	//template <typename... Args> void Set(int off, float Val, Args... args) { int n=off/VEC_CNT; vec[n].Set(off-n*VEC_CNT, Val); Set(++off,args...); };

	// reads
	VEC_T &operator[](int index_prm) {return vecs[index_prm];};
	float Get(int row,int col) {return vecs[row][col];}
	//void save(float *dst_prm) { };
	std::string as_str(const char *format_prm="%05.2f,") { std::string ret; for (int x=0;x<SIZE;x++) ret+=vecs[x].as_str(format_prm)+";\n"; return ret; }

	// adds urinary
	MatrixS & operator+=(const MatrixS &src2) { for (int x=0;x<SIZE;x++) vecs[x]+=src2.vec[x]; return *this;}
	MatrixS & operator-=(const MatrixS &src2) { for (int x=0;x<SIZE;x++) vecs[x]-=src2.vec[x]; return *this;}
	MatrixS & operator+=(const float src2) { for (int x=0;x<SIZE;x++) vecs[x]+=src2; return *this;}
	MatrixS & operator-=(const float src2) { for (int x=0;x<SIZE;x++) vecs[x]-=src2;; return *this;}

	// muls urinary
	MatrixS & operator*=(const MatrixS &src2)
	{
		MatrixS Trans(src2); //todo heap allocate if matrix is really large (or maybe do something requiring the caller to pre transpose it?, or just use the gather loads..
		Trans.Transpose();

		for (int x=0; x<SIZE; x++)
		{
			VEC_T result;
			for (int y=0; y<SIZE; y++)
			{
				float val = (vecs[x] * Trans.vecs[y]).ElementSum();
				result.Set(y, val);
			}
			vecs[x]=result;
		}
		return *this;
	}
	MatrixS & operator/=(const MatrixS &src2)
	{
		MatrixS Trans(src2);
		Trans.Transpose();

		for (int x=0; x<SIZE; x++)
		{
			VEC_T result;
			for (int y=0; y<SIZE; y++)
			{
				float val = (vecs[x] / Trans.vecs[y]).ElementSum();
				result.Set(y, val);
			}
			vecs[x]=result;
		}
		return *this;
	}
	VEC_T & operator*=(const VEC_T &src2) { VEC_T ret; for (int x=0;x<SIZE;x++) ret.Set(x,(vecs[x]*src2).ElementSum()); return *this;}
	VEC_T & operator/=(const VEC_T &src2) { VEC_T ret; for (int x=0;x<SIZE;x++) ret.Set(x,(vecs[x]/src2).ElementSum()); return *this;} //probably should build inverse vec?

	MatrixS & operator*=(const float src2) { for (int x=0;x<SIZE;x++) vecs[x]*=src2; return *this;}
	MatrixS & operator/=(const float src2) { for (int x=0;x<SIZE;x++) vecs[x]/=src2;; return *this;}

	//other ops
	void Transpose() { for (int x=1;x<SIZE;x++) for (int y=0;y<x;y++)  { float tmp=vecs[x][y];	vecs[x].Set(y,vecs[y][x]);	vecs[y].Set(x,tmp); }  } //do something smarter?
	void det() {
		MatrixS ret(0);
		for (int i=0;i<SIZE;i++)
		{

		}
	}
	// inverse

  private:
	VEC_T vecs[SIZE];
};

// non square matrix based on big vecs above
// most of these loops should be expanded with openMP if the matrix sizes get sufficiently large
#ifdef __AVX__
template<int SIZE_ROW,int SIZE_COL, class VEC_TR = BigVector<8,SIZE_ROW,SSEx8>, class VEC_TC = BigVector<8,SIZE_COL,SSEx8>> class MatrixST
#else
template<int SIZE_ROW,int SIZE_COL, class VEC_TR = BigVector<4,SIZE_ROW,Vec4>, class VEC_TC = BigVector<4,SIZE_COL,Vec4>> class MatrixST
#endif
{
//	typedef BigVector<4,SIZE_ROW,Vec4> VEC_TX;
  public:
	MatrixST() {} //construct unititalized
	MatrixST(float val) { VEC_TR tmp; tmp.Fill(val); for (int y=0;y<SIZE_COL;y++) vecs[y]=tmp; }
	MatrixST(const MatrixST &orig) {for (int y=0; y<SIZE_COL; y++) vecs[y]=orig.vecs[y]; }

	// assign
	//BigVector & operator=(const BigVector &src_prm) { for (int x=0;x<VECS;x++) vec[x]=src_prm.vec[x]; }
	//BigVector & operator=(const float src[ELEM_CNT]) { for (int x=0;x<ELEM_CNT;x++) vec[x/VEC_CNT].Set(x-(x/VEC_CNT)*VEC_CNT,src[x]);};
	void Set(int col, int row, float Val) { vecs[row].Set(col, Val); }
	//template <typename... Args> void Set(int off, float Val, Args... args) { int n=off/VEC_CNT; vec[n].Set(off-n*VEC_CNT, Val); Set(++off,args...); };

	// reads
	VEC_TR &operator[](int index_prm) {return vecs[index_prm];};
	float Get(int col,int row) {return vecs[row][col];}
	//void save(float *dst_prm) { };
	std::string as_str(const char *format_prm="%05.2f,") { std::string ret; for (int x=0;x<SIZE_COL;x++) ret+=vecs[x].as_str(format_prm)+";\n"; return ret; }

	// adds urinary
	MatrixST & operator+=(const MatrixST &src2) { for (int y=0;y<SIZE_COL;y++) vecs[y]+=src2.vec[y]; return *this;}
	MatrixST & operator-=(const MatrixST &src2) { for (int y=0;y<SIZE_COL;y++) vecs[y]-=src2.vec[y]; return *this;}
	MatrixST & operator+=(const float src2) { for (int y=0;y<SIZE_COL;y++) vecs[y]+=src2; return *this;}
	MatrixST & operator-=(const float src2) { for (int y=0;y<SIZE_COL;y++) vecs[y]-=src2;; return *this;}

	// muls urinary (can't change the dimensions, so this only works when the result doesn't change dimensions)
	template<int S2_ROW,int S2_COL, class S2_VR,class S2_VC>
	   MatrixST<SIZE_ROW, S2_COL, VEC_TR, S2_VC> & operator*=(const MatrixST<S2_ROW, S2_COL, S2_VR, S2_VC> &src2)
	{
		//MatrixST<SIZE_COL,SIZE_ROW, VEC_TC, VEC_TR> Trans(src2); //todo heap allocate if matrix is really large (or maybe do something requiring the caller to pre transpose it?, or just use the gather loads..
		MatrixST<S2_COL, S2_ROW, S2_VC, S2_VR> Trans=src2.Transpose();

		for (int x=0; x<SIZE_ROW; x++)
		{
			VEC_TR result;
			for (int y=0; y<S2_COL; y++)
			{
				float val = (vecs[x] * Trans.vecs[y]).ElementSum();
				result.Set(y, val);
			}
			vecs[x]=result;
		}
		return *this;
	}

	// muls binary, because it returns a type it can be used to change the dimensions
	template<int S2_ROW,int S2_COL, class S2_VR,class S2_VC>
	   MatrixST<SIZE_COL, S2_ROW, VEC_TC, S2_VR> operator*(const MatrixST<S2_ROW, S2_COL, S2_VR, S2_VC> &src2)
	{
		//MatrixST<SIZE_COL,SIZE_ROW, VEC_TC, VEC_TR> Trans(src2); //todo heap allocate if matrix is really large (or maybe do something requiring the caller to pre transpose it?, or just use the gather loads..
		MatrixST<S2_COL, S2_ROW, S2_VC, S2_VR> Trans=src2.Transpose();
		MatrixST<SIZE_COL, S2_ROW, VEC_TC, S2_VR> ret(0);

		for (int x=0; x<SIZE_COL; x++)
		{
			VEC_TC result;
			for (int y=0; y<S2_ROW; y++)
			{
				float val = (vecs[x] * Trans.vecs[y]).ElementSum();
				result.Set(y, val);
			}
			ret[x]=result;
		}

		return ret;
	}

	// version of above with the src pre-transposed
	//	template<int S2_ROW,int S2_COL, class S2_VR,class S2_VC>
	// MatrixST<SIZE_COL, S2_ROW, VEC_TC, S2_VR> MultNoTrans(const MatrixST<S2_ROW, S2_COL, S2_VR, S2_VC> &src2)


	VEC_TR & operator*=(const VEC_TR &src2) { VEC_TR ret; for (int y=0;y<SIZE_ROW;y++) ret.Set(y,(vecs[y]*src2).ElementSum()); return *this;}
	VEC_TR & operator/=(const VEC_TR &src2) { VEC_TR ret; for (int y=0;y<SIZE_ROW;y++) ret.Set(y,(vecs[y]/src2).ElementSum()); return *this;} //probably should build inverse vec?
	MatrixST & operator*=(const float src2) { for (int y=0;y<SIZE_ROW;y++) vecs[y]*=src2; return *this;}
	MatrixST & operator/=(const float src2) { for (int y=0;y<SIZE_ROW;y++) vecs[y]/=src2;; return *this;}

	//other ops
	// must return transposed matrix because m X n becomes n X m
	MatrixST<SIZE_COL,SIZE_ROW, VEC_TC, VEC_TR> Transpose() const
	{
		MatrixST<SIZE_COL,SIZE_ROW, VEC_TC, VEC_TR> ret;

		for (int y=0;y<SIZE_ROW;y++)
		{
			for (int x=0;x<SIZE_COL;x++)
			{
				ret.Set(x, y, vecs[x][y]);
			}
		}
		return ret; //should be move?!
	} //do something smarter?
	//void ITranspose() { for (int x=1;x<SIZE;x++) for (int y=0;y<=x;y++)  { float tmp=vecs[x][y];	vecs[x].Set(y,1.0/vecs[y][x]);	vecs[y].Set(x,1.0/tmp); }  } //do something smarter?

	MatrixST LU(void)
	{
		MatrixST U(0),L(0);
		if (SIZE_ROW != SIZE_COL)
		{
			throw "can't deal with non square matrix";
		}
		for (int i=0;i<SIZE_COL;i++)
		{
			//U[i][i] = 1.0;
			U.Set(i,i,1.0);
		}
		for (int j=0;j<SIZE_COL;j++)
		{
			for (int y=j;y<SIZE_COL;y++)
			{
				float sum = 0;
				for (int k=0;k<j;k++)
				{
					sum += L[y][k] * U[k][j];
				}
				L.Set(j,y,vecs[y][j]-sum);
				//printf("L:\n%s\n", L.as_str().c_str());
			}

			for (int i=j;i<SIZE_COL;i++)
			{
				float sum = 0;
				for (int k=0;k<j;k++)
				{
					sum += L[j][k] * U[k][i];
				}
				if (L[j][j]== 0.0) L.Set(j,j,0.0000001);

				U.Set(i,j,(vecs[j][i]-sum)/L[j][j]);
			}
		}
		printf("L:\n%s\n", L.as_str().c_str());
		printf("U:\n%s\n", U.as_str().c_str());

		return L;
	}
	float Det(void)
	{
		MatrixST L=LU();
		float det = L[0][0];//*U[0][0];
		for (int i=1;i<SIZE_COL;i++)
		{
			det *= L[i][i];
		//	det *= U[i][i];
		}
		printf("Det:%f\n", det);
		return det;
	}
  private:
	VEC_TR vecs[SIZE_COL]; //number of rows
};

template<int A,int B,class C,class D,int SA,int SB,class SC,class SD,template<int ,int ,class ,class > class T1, template<int ,int ,class ,class > class T2>void muls(T1<A,B,C,D> &s1, T2<SA,SB,SC,SD> &s)
{
	MatrixST<A,SB,C,SD> tmp(0);
	printf("new big vec (%d,%d):(%d,%d):\n%s\n", A,B,SA,SB,tmp.as_str().c_str());
}

#ifdef __AVX__
typedef BigVector<8,8,SSEx8>  Vec8;
typedef BigVector<8,16,SSEx8> Vec16;
typedef BigVector<8,32,SSEx8> Vec32;
typedef BigVector<8,64,SSEx8> Vec64;
typedef BigVector<8,4096,SSEx8> Vec4k;
#else
typedef BigVector<4,8,Vec4>	 Vec8;
typedef BigVector<4,16,Vec4> Vec16;
typedef BigVector<4,32,Vec4> Vec32;
typedef BigVector<4,64,Vec4> Vec64;
typedef BigVector<4,4096,Vec4> Vec4k;
#endif

// recursive template calls? sure why not. This breaks the raw() import method (and printing)..
typedef BigVector<64,128,Vec64> Vec128;


#include <memory>

int main(int argv,char *argc[])
{
	Vec8 x8vec(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0);
	Vec8 x8vec2;
	x8vec2.Fill(1.0);
	Vec16 x16vec(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0);
	Vec128 x128vec(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0);

	printf("	big vec %s\n", x8vec.as_str().c_str());
	x8vec+=10;
	printf("+10 big vec %s\n", x8vec.as_str().c_str());
	x8vec/=2;
	printf("/2	big vec %s\n", x8vec.as_str().c_str());
	x8vec+=x8vec;
	printf("*2	big vec %s\n", x8vec.as_str().c_str());
	x8vec+=x8vec2;
	printf("+1	big vec %s\n", x8vec.as_str().c_str());


	printf("\nbig vec %s\n", x16vec.as_str().c_str());
	x16vec+=Vec16(x8vec.raw(), x8vec.raw_sz());
	printf("big vec %s\n", x16vec.as_str().c_str());
//printf("\nbig vec %s\n", x128vec.as_str().c_str());
	//std::unique_ptr<Vec4k> x4kvec=std::make_unique<Vec4096>();
	Vec4k *x4kvec=new Vec4k(0.0);
	*x4kvec+=6.5;
	printf("\nbig vec %s\n", x4kvec->as_str().c_str());
	delete x4kvec;

	MatrixS<8,Vec8> x8Mat(1);
	for (int x=0;x<8;x++) x8Mat.Set(x,x,float(x));
	for (int x=0;x<8;x++) x8Mat.Set(0,x,float(x));
	printf("\nbig mat:\n%s\n", x8Mat.as_str().c_str());
	MatrixS<8,Vec8> x8Mat2(x8Mat);
	x8Mat2.Transpose();
	printf("big mat:\n%s\n", x8Mat2.as_str().c_str());
	x8Mat2*=x8Mat;
	printf("big mat:\n%s\n", x8Mat2.as_str().c_str());
	/* not computed correctly yet because we want to calc inverse (and det()) first
	x8Mat2/=x8Mat;
	printf("big mat:\n%s\n", x8Mat2.as_str().c_str());*/

	// square version, validate against above.
	MatrixST<8,8,Vec8,Vec8> x8nsMat(1);
	for (int x=0;x<8;x++) x8nsMat.Set(x,x,float(x));
	for (int x=0;x<8;x++) x8nsMat.Set(0,x,float(x));
	printf("\nbig matv:\n%s\n", x8nsMat.as_str().c_str());
	MatrixST<8,8,Vec8,Vec8> x8nsMat2(x8nsMat.Transpose());
	printf("big matv:\n%s\n", x8nsMat2.as_str().c_str());
	x8nsMat2*=x8nsMat;
	printf("big matv:\n%s\n", x8nsMat2.as_str().c_str());

	// nonsquare version (with default prms)
	MatrixST<16,8> x8nsMat3(0);
	for (int x=0;x<8;x++) { x8nsMat3.Set(x,x,float(x)); x8nsMat3.Set(x+8,x,float(x));}
	for (int x=0;x<8;x++) x8nsMat3.Set(0,x,float(x));
	printf("big mat:\n%s\n", x8nsMat3.as_str().c_str());
	MatrixST<8,16> x8nsMat4(x8nsMat3.Transpose());
//	auto x8nsMat4(x8nsMat3.Transpose()); // this works?!!
	printf("big mat:\n%s\n", x8nsMat4.as_str().c_str());

	auto x8nsMat5 = x8nsMat4 * x8nsMat3;
	printf("big mat:\n%s\n", x8nsMat5.as_str().c_str());

	auto x8nsMat6 = x8nsMat3 * x8nsMat4;
	printf("big mat:\n%s\n", x8nsMat6.as_str().c_str());

	//x8nsMat6.LU();
	//x8nsMat2.LU();
	x8nsMat2.Det();
	MatrixST<2,2> x2Mat(1);
	printf("small mat:\n%s\n", x2Mat.as_str().c_str());
	MatrixST<8,8> x2Mat2(1);
	printf("small mat:\n%s\n", x2Mat2.as_str().c_str());
	x2Mat2.LU();
//	x8nsMat4 *= x8nsMat3; //could one get away with changing the type? lol!
//	printf("big mat:\n%s\n", x8nsMat4.as_str().c_str());

// this will eat all your stack! lol
//	MatrixST<4096,4096> x8nsMat7(1);
//	printf("big mat:\n%s\n", x8nsMat7.as_str().c_str());

//	muls(x8nsMat3,x8nsMat2);
}
