// SPDX-License-Identifier: Apache-2.0
// CPU-only vector/DMA model; does not validate NPU events or instruction lowering.

#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <type_traits>
#define __host__
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;
#ifdef _WIN32
#define HOST_EXPORT __declspec(dllexport)
#else
#define HOST_EXPORT
#endif
constexpr int PIPE_V=0;
namespace AscendC {
using event_t=int;
enum class HardEvent { V_S,S_V,V_MTE2,MTE2_V,V_MTE3,MTE3_V,S_MTE3,MTE3_S,MTE2_S,MTE2_MTE3 };
template<HardEvent> void SetFlag(int) {}
template<HardEvent> void WaitFlag(int) {}
template<int> void PipeBarrier() {}
enum class TPosition { VECCALC };
enum class RoundMode { CAST_ROUND };
enum class CMPMODE { EQ,NE,LT,LE,GT,GE };
enum class SELMODE { VSEL_TENSOR_SCALAR_MODE,VSEL_TENSOR_TENSOR_MODE };
template<class T> struct LocalTensor {
    uint8_t* bytes; size_t size;
    LocalTensor operator[](size_t n) const { assert(n*sizeof(T)<=size);return {bytes+n*sizeof(T),size-n*sizeof(T)}; }
    template<class U> LocalTensor<U> ReinterpretCast() const { return {bytes,size}; }
    T GetValue(size_t n) const { assert((n+1)*sizeof(T)<=size);T v;std::memcpy(&v,bytes+n*sizeof(T),
    sizeof(T));return v; }
    void SetValue(size_t n,T v) const { assert((n+1)*sizeof(T)<=size);std::memcpy(bytes+n*sizeof(T),&v,sizeof(T)); }
};
template<TPosition> struct TBuf {
    std::vector<uint8_t> bytes;
    template<class T> LocalTensor<T> Get() {return {bytes.data(),bytes.size()};}
};
struct TPipe {
    size_t allocated=0;
    template<TPosition P> void InitBuffer(TBuf<P>& b,size_t n) {allocated+=n;b.bytes.assign(n,0xa5);}
    int FetchEventID(HardEvent) {return 0;}
};
inline TPipe* GetTPipePtr() {static TPipe p;return &p;}
template<class T> struct GlobalTensor {
    T* data;size_t size;
    void SetGlobalBuffer(T* p,size_t n=std::numeric_limits<size_t>::max()/sizeof(T)) {data=p;size=n;}
    GlobalTensor operator[](size_t n) const {assert(n<=size);return {data+n,size-n};}
};
struct DataCopyExtParams {uint16_t blocks;uint32_t bytes,sourceStride,destStride,reserved;};
template<class T> struct DataCopyPadExtParams {bool enabled;uint8_t left,right;T value;};
template<class T> void DataCopy(LocalTensor<T> d,LocalTensor<T> s,uint32_t n) {
    assert(n*sizeof(T)<=d.size&&n*sizeof(T)<=s.size);std::memmove(d.bytes,s.bytes,n*sizeof(T));
}
template<class T> void DataCopy(LocalTensor<T> d,GlobalTensor<T> s,uint32_t n) {
    assert(n*sizeof(T)<=d.size&&n<=s.size);std::memcpy(d.bytes,s.data,n*sizeof(T));
}
template<class T> void DataCopy(GlobalTensor<T> d,LocalTensor<T> s,uint32_t n) {
    assert(n*sizeof(T)<=s.size&&n<=d.size);std::memcpy(d.data,s.bytes,n*sizeof(T));
}
template<class T> void DataCopyPad(LocalTensor<T> d,GlobalTensor<T> s,DataCopyExtParams p,
    DataCopyPadExtParams<T> pad) {
    // Match SDK semantics: even pad.enabled=false overwrites the final UB
    // block's dummy lanes with the first source element; it is not memcpy.
    const size_t left=pad.enabled?pad.left*sizeof(T):0;
    const size_t right=pad.enabled?pad.right*sizeof(T):0;
    const size_t block=(left+p.bytes+right+31)/32*32;
    for(uint32_t i=0;i<p.blocks;++i) {
        size_t src=i*(p.bytes+p.sourceStride),dst=i*(block+p.destStride*32);
        assert(src+p.bytes<=s.size*sizeof(T)&&dst+block<=d.size);
        T filler=pad.value;
        if(!pad.enabled)std::memcpy(&filler,reinterpret_cast<uint8_t*>(s.data)+src,sizeof(T));
        for(size_t lane=0;lane<block/sizeof(T);++lane)d.SetValue(dst/sizeof(T)+lane,filler);
        std::memcpy(d.bytes+dst+left,reinterpret_cast<uint8_t*>(s.data)+src,p.bytes);
    }
}
template<class T> void DataCopyPad(GlobalTensor<T> d,LocalTensor<T> s,DataCopyExtParams p) {
    // UB advances by the padded transfer length in 32-byte blocks. GM
    // receives only the requested bytes and its stride is also in bytes.
    const size_t block=(p.bytes+31)/32*32;
    for(uint32_t i=0;i<p.blocks;++i) {
        const size_t src=i*(block+size_t(p.sourceStride)*32);
        const size_t dst=i*(p.bytes+size_t(p.destStride));
        assert(src+p.bytes<=s.size&&dst+p.bytes<=d.size*sizeof(T));
        std::memcpy(reinterpret_cast<uint8_t*>(d.data)+dst,s.bytes+src,p.bytes);
    }
}
template<class T> void Duplicate(LocalTensor<T> d,T x,uint32_t n) {for(uint32_t i=0;i<n;++i)d.SetValue(i,x);}
template<class T,class U> void Cast(LocalTensor<T> d,LocalTensor<U> s,RoundMode,uint32_t n) {
    std::vector<T> values(n);for(uint32_t i=0;i<n;++i)values[i]=static_cast<T>(s.GetValue(i));
    for(uint32_t i=0;i<n;++i)d.SetValue(i,values[i]);
}

template<class T,class U> T WrapAdd(T a,U b) {
    if constexpr(std::is_integral_v<T>) {using V=std::make_unsigned_t<T>;return static_cast<T>(V(a)+V(b));}
    else return a+b;
}
template<class T,class U> T WrapSub(T a,U b) {
    if constexpr(std::is_integral_v<T>) {using V=std::make_unsigned_t<T>;return static_cast<T>(V(a)-V(b));}
    else return a-b;
}
template<class T,class U> T WrapMul(T a,U b) {
    if constexpr(std::is_integral_v<T>) {using V=std::make_unsigned_t<T>;return static_cast<T>(V(a)*V(b));}
    else return a*b;
}

#define UNARY(NAME,EXPR) template<class T,class U> void NAME(LocalTensor<T>d,LocalTensor<T>s,U x,uint32_t n){ \
    std::vector<T>v(n);for(uint32_t i=0;i<n;++i){T a=s.GetValue(i);v[i]=(EXPR);}for(uint32_t i=0;i<n; \
    ++i)d.SetValue(i,v[i]);}
UNARY(ShiftRight,a>>x) UNARY(ShiftLeft,a<<x) UNARY(Adds,WrapAdd(a,x)) UNARY(Muls,WrapMul(a,x)) UNARY(Mins,std::min(a,static_cast<T>(x))) UNARY(Maxs,std::max(a,static_cast<T>(x)))
#define BINARY(NAME,EXPR) template<class T> void NAME(LocalTensor<T>d,LocalTensor<T>a,LocalTensor<T>b, \
    uint32_t n){std::vector<T>v(n);for(uint32_t i=0;i<n;++i){T x=a.GetValue(i),y=b.GetValue(i);v[i]=(EXPR);} \
    for(uint32_t i=0;i<n;++i)d.SetValue(i,v[i]);}
BINARY(Add,WrapAdd(x,y)) BINARY(Sub,WrapSub(x,y)) BINARY(Min,std::min(x,y)) BINARY(And,x&y) BINARY(Or,x|y)
template<class T> void Not(LocalTensor<T>d,LocalTensor<T>s,uint32_t n){for(uint32_t i=0;i<n;++i)d.SetValue(i,
    ~s.GetValue(i));}
template<class T> inline bool CompareValue(T a,T b,CMPMODE mode){switch(mode){case CMPMODE::EQ:return a==b;
    case CMPMODE::NE:return a!=b;case CMPMODE::LT:return a<b;case CMPMODE::LE:return a<=b;
    case CMPMODE::GT:return a>b;case CMPMODE::GE:return a>=b;}return false;}
inline bool Bit(LocalTensor<uint8_t>m,uint32_t i){return m.GetValue(i/8)&(1<<(i%8));}
inline void PutBit(LocalTensor<uint8_t>m,uint32_t i,bool value){auto b=m.GetValue(i/8);m.SetValue(i/8,
    (b&~(1<<(i%8)))|(value<<(i%8)));}
template<class T> void CompareScalar(LocalTensor<uint8_t>m,LocalTensor<T>a,T b,CMPMODE mode,uint32_t n){
    assert(n*sizeof(T)%256==0);
    for(uint32_t i=0;i<n;++i)PutBit(m,i,CompareValue(a.GetValue(i),b,mode));}
template<class T> void Compare(LocalTensor<uint8_t>m,LocalTensor<T>a,LocalTensor<T>b,CMPMODE mode,uint32_t n){
    assert(n*sizeof(T)%256==0);
    for(uint32_t i=0;i<n;++i)PutBit(m,i,CompareValue(a.GetValue(i),b.GetValue(i),mode));}
template<class T> void Select(LocalTensor<T>d,LocalTensor<uint8_t>m,LocalTensor<T>a,T b,SELMODE,uint32_t n){
    std::vector<T>v(n);for(uint32_t i=0;i<n;++i)v[i]=Bit(m,i)?a.GetValue(i):b;for(uint32_t i=0;i<n;
    ++i)d.SetValue(i,v[i]);}
template<class T> void Select(LocalTensor<T>d,LocalTensor<uint8_t>m,LocalTensor<T>a,LocalTensor<T>b,SELMODE,
    uint32_t n){std::vector<T>v(n);for(uint32_t i=0;i<n;++i)v[i]=Bit(m,i)?a.GetValue(i):b.GetValue(i);
    for(uint32_t i=0;i<n;++i)d.SetValue(i,v[i]);}
template<class T> void Gather(LocalTensor<T>d,LocalTensor<T>s,LocalTensor<uint32_t>idx,uint32_t offset,
    uint32_t n){std::vector<T>v(n);for(uint32_t i=0;i<n;++i){auto j=idx.GetValue(i)+offset;
    assert(j%sizeof(T)==0);v[i]=s.GetValue(j/sizeof(T));}for(uint32_t i=0;i<n;++i)d.SetValue(i,v[i]);}
struct GatherMaskParams {uint8_t a,b,c,d;};
template<class T> void GatherMask(LocalTensor<T>d,LocalTensor<T>s,LocalTensor<uint32_t>m,bool,uint32_t n,
    GatherMaskParams,uint64_t& kept){auto mask=m.template ReinterpretCast<uint8_t>();std::vector<T>v;
    for(uint32_t i=0;i<n;++i)if(Bit(mask,i))v.push_back(s.GetValue(i));for(uint32_t i=0;i<v.size();
    ++i)d.SetValue(i,v[i]);kept=v.size();}
inline void ReduceMax(LocalTensor<float>d,LocalTensor<float>s,LocalTensor<float>,uint32_t n){
    float v=s.GetValue(0);for(uint32_t i=1;i<n;++i)v=std::max(v,s.GetValue(i));d.SetValue(0,v);}
struct SortEntry{float key;uint32_t index;};
inline void WriteEntries(LocalTensor<float>d,const std::vector<SortEntry>&v,uint32_t begin=0){
    for(uint32_t i=0;i<v.size();++i){d.SetValue(2*(begin+i),v[i].key);
    d.template ReinterpretCast<uint32_t>().SetValue(2*(begin+i)+1,v[i].index);}}
inline void SortEntries(std::vector<SortEntry>&v){std::sort(v.begin(),v.end(),[](auto a,auto b){
    return a.key>b.key||(a.key==b.key&&a.index>b.index);});}
inline void Sort32(LocalTensor<float>d,LocalTensor<float>keys,LocalTensor<uint32_t>pos,uint32_t groups){
    assert(groups > 0 && groups <= 255);
    for(uint32_t g=0;g<groups;++g){std::vector<SortEntry>v;for(uint32_t i=0;i<32;++i)v.push_back({
    keys.GetValue(g*32+i),pos.GetValue(g*32+i)});SortEntries(v);WriteEntries(d,v,g*32);}}
struct MrgSort4Info {uint32_t elementLengths[4];bool ifExhaustedSuspension;uint8_t validBit;uint32_t repeatTimes;};
template<class T> struct MrgSortSrcList {LocalTensor<T>src1,src2,src3,src4;};
template<class T> void MrgSort(LocalTensor<T>d,MrgSortSrcList<T>sources,MrgSort4Info p){LocalTensor<T>src[]={
    sources.src1,sources.src2,sources.src3,sources.src4};uint32_t width=0;for(int i=0;i<4;
    ++i)if(p.validBit&(1<<i))width+=p.elementLengths[i];for(uint32_t r=0;r<p.repeatTimes;++r){
    std::vector<SortEntry>v;for(int i=0;i<4;++i)if(p.validBit&(1<<i))for(uint32_t j=0;j<p.elementLengths[i];
    ++j){uint32_t k=2*(r*width+j);v.push_back({src[i].GetValue(k),
    src[i].template ReinterpretCast<uint32_t>().GetValue(k+1)});}SortEntries(v);WriteEntries(d,v,r*width);}}
inline void Extract(LocalTensor<float>keys,LocalTensor<uint32_t>idx,LocalTensor<float>src,uint32_t groups){
    assert(groups > 0 && groups <= 255);
    for(uint32_t i=0;i<groups*32;++i){keys.SetValue(i,src.GetValue(2*i));idx.SetValue(i,
    src.template ReinterpretCast<uint32_t>().GetValue(2*i+1));}}
} // AscendC
