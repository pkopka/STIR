//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementations for class SeparableMetzArrayFilter

  \author Matthew Jacobson
  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/SeparableMetzArrayFilter.h"
#include "stir/ArrayFilter1DUsingConvolutionSymmetricKernel.h"
#include "stir/VectorWithOffset.h"
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR


const float ZERO_TOL= 0.000001F; //MJ 12/05/98 Made consistent with other files
const double TPI=6.28318530717958647692;
const int  FORWARDFFT=1;
const int INVERSEFFT=-1;

// TODO get rid of this #defines
#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#define REALC(a) 2*(a)
#define IMGC(a) 2*(a)+1

// build Gaussian kernel according to the full width half maximum 
template <typename elemT>
static void build_gauss(VectorWithOffset<elemT>&kernel, 
			int res,float s2, float sampling_interval);

template <typename elemT>
static void discrete_fourier_transform(VectorWithOffset<elemT>&data, unsigned int nn, int isign);

template <typename elemT>
static void build_metz(VectorWithOffset<elemT>&kernel,
		       float N,float fwhm, float MmPerVoxel, int max_kernel_size);



template <int num_dimensions, typename elemT>
SeparableMetzArrayFilter<num_dimensions,elemT>::
SeparableMetzArrayFilter
  (const VectorWithOffset<float>& fwhms_v,
   const VectorWithOffset<float>& metz_powers_v,
   const BasicCoordinate<num_dimensions,float>& sampling_distances_v,
   const VectorWithOffset<int>& max_kernel_sizes_v)
 : fwhms(fwhms_v),
   metz_powers(metz_powers_v),
   sampling_distances(sampling_distances_v),
   max_kernel_sizes(max_kernel_sizes_v)
{
  assert(metz_powers.get_length() == num_dimensions);
  assert(fwhms.get_length() == num_dimensions);
  assert(metz_powers.get_min_index() == 1);
  assert(fwhms.get_min_index() == 1);
  assert(max_kernel_sizes.get_length() == num_dimensions);
  assert(max_kernel_sizes.get_min_index() == 1);
  
  for (int i=1; i<=num_dimensions; ++i)
  {
    VectorWithOffset<elemT> kernel;
    build_metz(kernel, metz_powers[i],fwhms[i],sampling_distances[i],max_kernel_sizes[i]);
    
    for (int j=0;j<kernel.get_length();j++)
      if(metz_powers[i]>0.0)  printf ("%d-dir Metz[%d]=%f\n",i,j,kernel[j]);   
      else printf ("%d-dir Gauss[%d]=%f\n",i,j,kernel[j]);
      
      
      all_1d_array_filters[i-1] = new ArrayFilter1DUsingConvolutionSymmetricKernel<elemT>(kernel);
      
  }
}


template <typename elemT>
void discrete_fourier_transform(VectorWithOffset<elemT>&data, unsigned int nn, int isign)
{
  unsigned int n,mmax,m,j,istep,i;
  double wtemp,wr,wpr,wpi,wi,theta;
  elemT tempr,tempi;
  n=nn << 1;
  j=1;
  for (i=1;i<n;i+=2) {
    if (j > i) {
      SWAP(data[j],data[i]);
      SWAP(data[j+1],data[i+1]);
    }
    m=n >> 1;
    while (m >= 2 && j > m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  mmax=2;
  while (n > mmax) {
    istep=mmax << 1;
    theta=isign*(TPI/mmax);
    wtemp=sin(0.5*theta);
    wpr = -2.0*wtemp*wtemp;
    wpi=sin(theta);
    wr=1.0;
    wi=0.0;
    for (m=1;m<mmax;m+=2) {
      for (i=m;i<=n;i+=istep) {
        j=i+mmax;
        tempr=wr*data[j]-wi*data[j+1];
        tempi=wr*data[j+1]+wi*data[j];
        data[j]=data[i]-tempr;
        data[j+1]=data[i+1]-tempi;
        data[i] += tempr;
        data[i+1] += tempi;
      }
      wr=(wtemp=wr)*wpr-wi*wpi+wr;
      wi=wi*wpr+wtemp*wpi+wi;
    }
    mmax=istep;
  }
}


template <typename elemT>
void build_gauss(VectorWithOffset<elemT>&kernel, int res,float s2,  float sampling_interval)
{
  
  
  elemT sum;
  int cutoff=0;
  int j,hres;
  
  
  
  hres = res/2;
  kernel[hres-1] = 1/sqrt(s2*TPI);
  sum =   kernel[hres-1];       
  kernel[res-1] = 0;
  for (j=1;(j<hres && !cutoff);j++){
    kernel[hres-j-1] = kernel[hres-1]*(double ) exp(-0.5*(j*sampling_interval)*(j*sampling_interval)/s2);
    kernel[hres+j-1] = kernel [hres-j-1];
    sum +=  2.0 * kernel[hres-j-1];
    if (kernel[hres-j-1]  <kernel[hres-1]*ZERO_TOL) cutoff=1;
            
  }  
  
  /* Normalize the filter to 1 */
  for (j=0;j<res;j++) kernel[j] /= sum; 
  
}







//MJ 19/04/99  Used KT's solution to the shifted index problem. Also build_metz now allocates the kernel.
template <typename elemT>
void build_metz(VectorWithOffset<elemT>& kernel,
		float N,float fwhm, float MmPerVox, int max_kernel_size)
{    
  
  int kernel_length = 0;
  
  if(fwhm>0.0F){
    
    
    // KT 30/05/2000 dropped unsigned
    int i;
    elemT xreal,ximg,zabs2;                                        
    
    //MJ 12/05/98 compute parameters relevant to DFT/IDFT
    
    elemT s2 = fwhm*fwhm/(8*log(2)); //variance in Mm
    
    const int n=7; //determines cut-off in both space and frequency domains
    const elemT sinc_length=10000.0;
    int samples_per_voxel=(int)(MmPerVox*2*sqrt(2*log(10)*n/s2)/TPI +1);
    const elemT sampling_interval=MmPerVox/samples_per_voxel;
    elemT stretch= (samples_per_voxel>1)?sinc_length:0.0;
    
    int Res=(int)(log((sqrt(8*n*log(10)*s2)+stretch)/sampling_interval)/log(2)+1);
    Res=(int) pow(2.0,(double) Res); //MJ 12/05/98 made adaptive 
    
    
    
    cerr<<endl<<"Variance: "<< s2<<endl; 
    cerr<<"Voxel dimension (in mm): "<< MmPerVox<<endl;  
    cerr<<"Samples per voxel: "<< samples_per_voxel<<endl;
    cerr<<"Sampling interval (in mm): "<< sampling_interval<<endl;
    cerr<<"FFT vector length: "<<Res<<endl;  
    
    
    /* allocate memory to metz arrays */
    VectorWithOffset<elemT> filter(Res);
    
    //MJ 05/03/2000 padded 1 more element to fftdata and pre-increment
    //The former technique was illegal.
    VectorWithOffset<elemT> fftdata(0,2*Res);
    for (i=0;i<Res ;i++ ) filter[i]=0.0;
    for (i=0;i<2*Res ; i++ ) fftdata[i]=0.0;     
    
    
    /* build gaussian */
    
    build_gauss(filter,Res,s2,sampling_interval);
    
    
    
    /* Build the fft array, odd coefficients are the imaginary part */
    
    for (i=0;i<=Res-(Res/2);i++) {
      fftdata[REALC(i)]=filter[Res/2-1+i];
      fftdata[IMGC(i)] = 0.0;
    }
    
    for (i=1;i<(Res/2);i++) {
      fftdata[REALC(Res-(Res/2)+i)]=filter[i-1];
      fftdata[IMGC(Res-(Res/2)+i)] = 0.0;
    }
    
    
    /* FFT to frequency space */
    fftdata.set_offset(1);
    discrete_fourier_transform(fftdata/*-1*/,Res,FORWARDFFT); 
    fftdata.set_offset(0);
    
    
    
    /* Build Metz */                       
    N++;
    
    
    int cutoff=(int) (sampling_interval*Res/(2*MmPerVox));
    //cerr<<endl<<"The cutoff was at: "<<cutoff<<endl;
    
    
    for (i=0;i<Res;i++) {
      
      
      xreal = fftdata[REALC(i)];
      ximg  = fftdata[IMGC(i)]; 
      zabs2= xreal*xreal+ximg*ximg;
	     filter[i]=0.0; // use this loop to clear the array for later
             
             
             //MJ 26/05/99 cut off the filter
             if(stretch>0.0){
               // cerr<<endl<<"truncating"<<endl;
               if(i>cutoff && Res-i>cutoff) zabs2=0;
               
             }
             
             if (zabs2>1) zabs2=(elemT) (1-ZERO_TOL);
             if (zabs2>0) {
               // if (zabs2>=1) cerr<<endl<<"zabs2 is "<<zabs2<<" and N is "<<N<<endl;
               fftdata[REALC(i)]=(1-pow((1-zabs2),N))*(xreal/zabs2);
               fftdata[IMGC(i)]=(1-pow((1-zabs2),N))*(-ximg/zabs2);
             }
             else {
               
               fftdata[REALC(i)]= 0.0;
               fftdata[IMGC(i)]= 0.0;
             }
             
    }
    /* return to the spatial space */               
    
    fftdata.set_offset(1);
    discrete_fourier_transform(fftdata/*-1*/,Res,INVERSEFFT); 
    fftdata.set_offset(0);
    
    
    
    
    
    
    /* collect the results, normalize*/
    
    for (i=0;i<=Res/2;i++) {
      if (i%samples_per_voxel==0){
        int j=i/samples_per_voxel;
        filter[j] = (fftdata[REALC(i)]*MmPerVox)/(Res*sampling_interval);
      }
      
    }
    
    
    
    //MJ 17/12/98 added step to undo zero padding (requested by RL)
    // KT 01/06/2001 added kernel_length stuff
    kernel_length=Res; 
    
    for (i=Res-1;i>=0;i--){
      if (fabs((double) filter[i])>=(0.0001)*filter[0]) break;
      else (kernel_length)--;
      
    }
    
    
#if 0
    // SM&KT 04/04/2001 removed this truncation of the kernel as we don't have the relevant parameter anymore
    if ((kernel_length)>length_of_row_to_filter/2){
      kernel_length=length_of_row_to_filter/2;
    }
#endif

    if (max_kernel_size>0 && (kernel_length)>max_kernel_size/2){
      kernel_length=max_kernel_size/2;
    }
    
    //VectorWithOffset<elemT> kernel(kernel_length);//=new elemT[(kernel_length)];
    kernel.grow(0,kernel_length-1);
    
    for (i=0;i<(kernel_length);i++) kernel[i]=filter[i];
    
    //return kernel;
  }
  
  else{
    //VectorWithOffset<elemT> kernel(1);//=new elemT[1];
    kernel.grow(0,0);
    //*kernel=1.0F;
    //kernel_length=1L;
    kernel[0] = 1.F;
    kernel_length=1;
    
    //return kernel;
  }

}



template SeparableMetzArrayFilter<3, float>;

END_NAMESPACE_STIR



