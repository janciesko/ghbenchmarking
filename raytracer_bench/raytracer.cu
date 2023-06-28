#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <fstream>
#define w  500
#define h  500
#define BPP 3

struct Vec {
  float x;
  float y;
  float z;

  __device__ Vec(float v = 0) {
    x = y = z = v;
  }

  __device__ Vec(float a, float b, float c = 0) {
    x = a;
    y = b;
    z = c;
  }

  __device__  Vec operator+(const Vec r) const  { return Vec(x + r.x , y + r.y , z + r.z); }
  __device__ Vec operator*(const Vec r) const { return   Vec(x * r.x , y * r.y , z * r.z); }
  __device__  float operator%(const Vec r) const {return     x * r.x + y * r.y + z * r.z;}
  __device__ Vec operator!() { return *this * rsqrtf(*this % *this); }
};

__shared__ curandState_t states[w];
__device__ float randomVal2() {
   return curand_uniform(&states[threadIdx.x]);
} 

__device__ int g_seed=1;
__device__ float randomVal() {
  g_seed = (214013*g_seed+2531011);
      return ((g_seed>>16)&0x7FFF) / (float)66635;
}


// Rectangle CSG equation. Returns minimum signed distance from
// space carved by
// lowerLeft vertex and opposite rectangle vertex upperRight.
__device__ float BoxTest(const Vec& position, Vec lowerLeft, Vec upperRight) {
  lowerLeft = position + lowerLeft * -1.0f;
  upperRight = upperRight + position * -1.0f;
  return -fminf(
      fminf(fminf(lowerLeft.x, upperRight.x), fminf(lowerLeft.y, upperRight.y)),
      fminf(lowerLeft.z, upperRight.z));
}

#define HIT_NONE 0
#define HIT_LETTER 1
#define HIT_WALL 2
#define HIT_SUN 3

// Sample the world using Signed Distance Fields.
__device__ float QueryDatabase(const Vec& position, int &hitType) {
  float distance = 1e9;//FLT_MAX;
  Vec f = position; // Flattened position (z=0)
  f.z = 0;
  
  
  char letters[15*4+1] =               // 15 two points lines
          "5O5_" "5W9W" "5_9_"         // P (without curve)
          "AOEO" "COC_" "A_E_"         // I
          "IOQ_" "I_QO"                // X
          "UOY_" "Y_]O" "WW[W"         // A
          "aOa_" "aWeW" "a_e_" "cWiO"; // R (without curve)

  for (int i = 0; i < sizeof(letters); i += 4) {
    Vec begin = Vec(letters[i] - 79, letters[i + 1] - 79) * .5;
    Vec e = Vec(letters[i + 2] - 79, letters[i + 3] - 79) * .5 + begin * -1;
    Vec o = f + (begin + e * fminf(
                                  -fminf(__fdividef((begin + f * -1) % e , (e % e)), 0),
                                   1)
                             ) * -1;
    distance = fminf(distance, o % o); // compare squared distance.
  }
  distance = sqrtf(distance); // Get real distance, not square distance.
  

  // Two curves (for P and R in PixaR) with hard-coded locations.
  Vec curves[] = {Vec(-11, 6), Vec(11, 6)};
  for (int i = 2; i--;) {
    Vec o = f + curves[i] * -1;
    distance = fminf(distance, o.x > 0 ? fabsf(sqrtf(o % o) - 2)
                                     : (o.y += o.y > 0 ? -2 : 2, sqrtf(o % o)));
  }
  distance = __powf(
    // distance * distance * distance  * distance * 
    // distance * distance *
    // distance  * distance  
    powf(distance, 8)
    + 
    // position.z * position.z * position.z * position.z *
    // position.z * position.z * 
    // position.z * position.z
    powf(position.z, 8)
    , 0.125f) - 0.5f;

  hitType = HIT_LETTER;
  

  float roomDist ;
  roomDist = fminf( // min(A,B) = Union with Constructive solid geometry
                  //-min carves an empty space
      -fminf(       // Lower room
          BoxTest(position, Vec(-30, -.5, -30), Vec(30, 18, 30)),
          // Upper room
          BoxTest(position, Vec(-25, 17, -25), Vec(25, 20, 25))
      ),
      BoxTest( // Ceiling "planks" spaced 8 units apart.
          Vec(fmodf(fabsf(position.x), 8), position.y, position.z),
          Vec(1.5, 18.5, -25), 
          Vec(6.5, 20, 25)
      )
  );
  if (roomDist < distance)
    distance = roomDist, hitType = HIT_WALL;

  float sun = 19.9 - position.y; // Everything above 19.9 is light source.
  if (sun < distance)
    distance = sun, hitType = HIT_SUN;

  return distance;
}

// Perform signed sphere marching
// Returns hitType 0, 1, 2, or 3 and update hit position/normal
__device__ int RayMarching(const Vec& origin, const Vec& direction, Vec& hitPos, Vec& hitNorm) {
  int hitType = HIT_NONE;
  int noHitCount = 0;

  // Signed distance marching
  float d; // distance from closest object in world.
  for (float total_d = 0; total_d < 100; total_d += d) {
    if ((d = QueryDatabase(hitPos = origin + direction * total_d, hitType)) < .01 || ++noHitCount > 99)
      return hitNorm =
                 !Vec(QueryDatabase(hitPos + Vec(.01, 0), noHitCount) - d,
                      QueryDatabase(hitPos + Vec(0, .01), noHitCount) - d,
                      QueryDatabase(hitPos + Vec(0, 0, .01), noHitCount) - d),
             hitType; // Weird return statement where a variable is also
                      // updated.
  }
  return 0;
}

__device__ Vec Trace(Vec origin, Vec direction) {
  Vec sampledPosition;
  Vec normal;
  Vec color = 0;
  Vec attenuation = 1;
  Vec lightDirection(!Vec(0.6f, 0.6f, 1.0f)); // Directional light

  for (int bounceCount = 3; bounceCount--;) {
    int hitType = RayMarching(origin, direction, sampledPosition, normal);
    if (hitType == HIT_NONE)
      break;                     // No hit. This is over, return color.
    if (hitType == HIT_LETTER) { // Specular bounce on a letter. No color acc.
      direction = direction + normal * (normal % direction * -2);
      origin = sampledPosition + direction * 0.1f;
      attenuation = attenuation * 0.2f; // Attenuation via distance traveled.
    }
    if (hitType == HIT_WALL) { // Wall hit uses color yellow?
      float incidence = normal % lightDirection;
      float p = 6.283185f * randomVal();
      float c = randomVal();
      float s = sqrtf(1 - c);
      float g = normal.z < 0 ? -1 : 1;
      float u = __fdividef(-1, (g + normal.z));
      float v = normal.x * normal.y * u;
      float cosp;
      float sinp;
      __sincosf(p, &sinp, &cosp); 
      direction =
          Vec(v, g + normal.y * normal.y * u, -normal.y) * (cosp * s) +
          Vec(1 + g * normal.x * normal.x * u, g * v, -g * normal.x) *
              (sinp * s) +
          normal * sqrtf(c);
      origin = sampledPosition + direction * 0.1f;
      attenuation = attenuation * 0.2f;
      if (incidence > 0.0f &&
          RayMarching(sampledPosition + normal * 0.1f, lightDirection,
                      sampledPosition, normal) == HIT_SUN)
        color = color + attenuation * Vec(500, 400, 100) * incidence;
    }
    if (hitType == HIT_SUN) { //
      color = color + attenuation * Vec(50, 80, 100);
      break; // Sun Color
    }
  }
  return color;
}


__global__ void GetColor(unsigned char *img, int samplesCount) {
  int x = blockIdx.x;
  int y = threadIdx.x;  

  curand_init( blockIdx.x * threadIdx.x, 0, 0 , &states[threadIdx.x]);

  const Vec position(-22.0f, 5.0f, 25.0f);
  const Vec goal = !(Vec(-3.0f, 4.0f, 0.0f) + position * -1.0f);
  const Vec left = !Vec(goal.z, 0, -goal.x) * (1.0f / w);

  // Cross-product to get the up vector
  const Vec up(goal.y *left.z - goal.z * left.y, goal.z *left.x - goal.x * left.z,
       goal.x *left.y - goal.y * left.x);

  Vec color;
  for (int p = samplesCount; p--;) {
    color = color + Trace(position, 
                          !(goal + left * (x - w / 2.0f + randomVal()) + up * (y - h / 2.0f + randomVal())));
  }

  // Reinhard tone mapping
  color = color * (1.0f / samplesCount) + 14.0f / 241.0f;
  Vec o = color + 1.0f;
  color = Vec(color.x / o.x, color.y / o.y, color.z / o.z) * 255.0f;
  img[(y * w + x) * BPP]     = color.x;
  img[(y * w + x) * BPP + 1] = color.y;
  img[(y * w + x) * BPP + 2] = color.z;
  
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char **argv) {
  std::ofstream outfile;
  unsigned char *dev_bitmap;
  cudaMalloc( (void**)&dev_bitmap, (w * h * BPP) );
  gpuErrchk( cudaPeekAtLastError() );

  int samplesCount = 1 << 10;
  if (argc > 1 ) {
    samplesCount = atoi(argv[1]);
  }  

  GetColor<<<w,h>>>(dev_bitmap, samplesCount); 
  gpuErrchk( cudaDeviceSynchronize() );

  char* bitmap = new char[w * h * BPP];
  cudaMemcpy( bitmap, dev_bitmap, (w * h * BPP), cudaMemcpyDeviceToHost );
  gpuErrchk( cudaPeekAtLastError() );
 
  outfile.open("out.ppm", std::ios_base::app);  
  outfile << "P6 " << w << " " << h << " 255 ";
  char* c = bitmap;
  for (int y = h; y--;) {
    for (int x = w; x--;) {
      c = &bitmap[y * w * BPP  + x * BPP];
      outfile << char(c[0]) << char(c[1]) << char(c[2]);
      c += BPP;
    }
  }

  delete bitmap;
  return EXIT_SUCCESS;
}

// Andrew Kensler


