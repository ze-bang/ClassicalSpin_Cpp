# 1 "CMakeCUDACompilerId.cu"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
# 1
#pragma GCC diagnostic push
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"
# 1
#pragma GCC diagnostic ignored "-Wunused-function"
# 1
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
# 1
#pragma GCC diagnostic pop
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"

# 1
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false
#if defined(__nv_is_extended_device_lambda_closure_type) && defined(__nv_is_extended_host_device_lambda_closure_type)
#endif

# 1
# 61 "/usr/include/cuda_runtime.h" 3
#pragma GCC diagnostic push
# 64
#pragma GCC diagnostic ignored "-Wunused-function"
# 68 "/usr/include/device_types.h" 3
#if 0
# 68
enum cudaRoundMode { 
# 70
cudaRoundNearest, 
# 71
cudaRoundZero, 
# 72
cudaRoundPosInf, 
# 73
cudaRoundMinInf
# 74
}; 
#endif
# 100 "/usr/include/vector_types.h" 3
#if 0
# 100
struct char1 { 
# 102
signed char x; 
# 103
}; 
#endif
# 105 "/usr/include/vector_types.h" 3
#if 0
# 105
struct uchar1 { 
# 107
unsigned char x; 
# 108
}; 
#endif
# 111 "/usr/include/vector_types.h" 3
#if 0
# 111
struct __attribute((aligned(2))) char2 { 
# 113
signed char x, y; 
# 114
}; 
#endif
# 116 "/usr/include/vector_types.h" 3
#if 0
# 116
struct __attribute((aligned(2))) uchar2 { 
# 118
unsigned char x, y; 
# 119
}; 
#endif
# 121 "/usr/include/vector_types.h" 3
#if 0
# 121
struct char3 { 
# 123
signed char x, y, z; 
# 124
}; 
#endif
# 126 "/usr/include/vector_types.h" 3
#if 0
# 126
struct uchar3 { 
# 128
unsigned char x, y, z; 
# 129
}; 
#endif
# 131 "/usr/include/vector_types.h" 3
#if 0
# 131
struct __attribute((aligned(4))) char4 { 
# 133
signed char x, y, z, w; 
# 134
}; 
#endif
# 136 "/usr/include/vector_types.h" 3
#if 0
# 136
struct __attribute((aligned(4))) uchar4 { 
# 138
unsigned char x, y, z, w; 
# 139
}; 
#endif
# 141 "/usr/include/vector_types.h" 3
#if 0
# 141
struct short1 { 
# 143
short x; 
# 144
}; 
#endif
# 146 "/usr/include/vector_types.h" 3
#if 0
# 146
struct ushort1 { 
# 148
unsigned short x; 
# 149
}; 
#endif
# 151 "/usr/include/vector_types.h" 3
#if 0
# 151
struct __attribute((aligned(4))) short2 { 
# 153
short x, y; 
# 154
}; 
#endif
# 156 "/usr/include/vector_types.h" 3
#if 0
# 156
struct __attribute((aligned(4))) ushort2 { 
# 158
unsigned short x, y; 
# 159
}; 
#endif
# 161 "/usr/include/vector_types.h" 3
#if 0
# 161
struct short3 { 
# 163
short x, y, z; 
# 164
}; 
#endif
# 166 "/usr/include/vector_types.h" 3
#if 0
# 166
struct ushort3 { 
# 168
unsigned short x, y, z; 
# 169
}; 
#endif
# 171 "/usr/include/vector_types.h" 3
#if 0
# 171
struct __attribute((aligned(8))) short4 { short x; short y; short z; short w; }; 
#endif
# 172 "/usr/include/vector_types.h" 3
#if 0
# 172
struct __attribute((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
# 174 "/usr/include/vector_types.h" 3
#if 0
# 174
struct int1 { 
# 176
int x; 
# 177
}; 
#endif
# 179 "/usr/include/vector_types.h" 3
#if 0
# 179
struct uint1 { 
# 181
unsigned x; 
# 182
}; 
#endif
# 184 "/usr/include/vector_types.h" 3
#if 0
# 184
struct __attribute((aligned(8))) int2 { int x; int y; }; 
#endif
# 185 "/usr/include/vector_types.h" 3
#if 0
# 185
struct __attribute((aligned(8))) uint2 { unsigned x; unsigned y; }; 
#endif
# 187 "/usr/include/vector_types.h" 3
#if 0
# 187
struct int3 { 
# 189
int x, y, z; 
# 190
}; 
#endif
# 192 "/usr/include/vector_types.h" 3
#if 0
# 192
struct uint3 { 
# 194
unsigned x, y, z; 
# 195
}; 
#endif
# 197 "/usr/include/vector_types.h" 3
#if 0
# 197
struct __attribute((aligned(16))) int4 { 
# 199
int x, y, z, w; 
# 200
}; 
#endif
# 202 "/usr/include/vector_types.h" 3
#if 0
# 202
struct __attribute((aligned(16))) uint4 { 
# 204
unsigned x, y, z, w; 
# 205
}; 
#endif
# 207 "/usr/include/vector_types.h" 3
#if 0
# 207
struct long1 { 
# 209
long x; 
# 210
}; 
#endif
# 212 "/usr/include/vector_types.h" 3
#if 0
# 212
struct ulong1 { 
# 214
unsigned long x; 
# 215
}; 
#endif
# 222 "/usr/include/vector_types.h" 3
#if 0
# 222
struct __attribute((aligned((2) * sizeof(long)))) long2 { 
# 224
long x, y; 
# 225
}; 
#endif
# 227 "/usr/include/vector_types.h" 3
#if 0
# 227
struct __attribute((aligned((2) * sizeof(unsigned long)))) ulong2 { 
# 229
unsigned long x, y; 
# 230
}; 
#endif
# 234 "/usr/include/vector_types.h" 3
#if 0
# 234
struct long3 { 
# 236
long x, y, z; 
# 237
}; 
#endif
# 239 "/usr/include/vector_types.h" 3
#if 0
# 239
struct ulong3 { 
# 241
unsigned long x, y, z; 
# 242
}; 
#endif
# 244 "/usr/include/vector_types.h" 3
#if 0
# 244
struct __attribute((aligned(16))) long4 { 
# 246
long x, y, z, w; 
# 247
}; 
#endif
# 249 "/usr/include/vector_types.h" 3
#if 0
# 249
struct __attribute((aligned(16))) ulong4 { 
# 251
unsigned long x, y, z, w; 
# 252
}; 
#endif
# 254 "/usr/include/vector_types.h" 3
#if 0
# 254
struct float1 { 
# 256
float x; 
# 257
}; 
#endif
# 276 "/usr/include/vector_types.h" 3
#if 0
# 276
struct __attribute((aligned(8))) float2 { float x; float y; }; 
#endif
# 281 "/usr/include/vector_types.h" 3
#if 0
# 281
struct float3 { 
# 283
float x, y, z; 
# 284
}; 
#endif
# 286 "/usr/include/vector_types.h" 3
#if 0
# 286
struct __attribute((aligned(16))) float4 { 
# 288
float x, y, z, w; 
# 289
}; 
#endif
# 291 "/usr/include/vector_types.h" 3
#if 0
# 291
struct longlong1 { 
# 293
long long x; 
# 294
}; 
#endif
# 296 "/usr/include/vector_types.h" 3
#if 0
# 296
struct ulonglong1 { 
# 298
unsigned long long x; 
# 299
}; 
#endif
# 301 "/usr/include/vector_types.h" 3
#if 0
# 301
struct __attribute((aligned(16))) longlong2 { 
# 303
long long x, y; 
# 304
}; 
#endif
# 306 "/usr/include/vector_types.h" 3
#if 0
# 306
struct __attribute((aligned(16))) ulonglong2 { 
# 308
unsigned long long x, y; 
# 309
}; 
#endif
# 311 "/usr/include/vector_types.h" 3
#if 0
# 311
struct longlong3 { 
# 313
long long x, y, z; 
# 314
}; 
#endif
# 316 "/usr/include/vector_types.h" 3
#if 0
# 316
struct ulonglong3 { 
# 318
unsigned long long x, y, z; 
# 319
}; 
#endif
# 321 "/usr/include/vector_types.h" 3
#if 0
# 321
struct __attribute((aligned(16))) longlong4 { 
# 323
long long x, y, z, w; 
# 324
}; 
#endif
# 326 "/usr/include/vector_types.h" 3
#if 0
# 326
struct __attribute((aligned(16))) ulonglong4 { 
# 328
unsigned long long x, y, z, w; 
# 329
}; 
#endif
# 331 "/usr/include/vector_types.h" 3
#if 0
# 331
struct double1 { 
# 333
double x; 
# 334
}; 
#endif
# 336 "/usr/include/vector_types.h" 3
#if 0
# 336
struct __attribute((aligned(16))) double2 { 
# 338
double x, y; 
# 339
}; 
#endif
# 341 "/usr/include/vector_types.h" 3
#if 0
# 341
struct double3 { 
# 343
double x, y, z; 
# 344
}; 
#endif
# 346 "/usr/include/vector_types.h" 3
#if 0
# 346
struct __attribute((aligned(16))) double4 { 
# 348
double x, y, z, w; 
# 349
}; 
#endif
# 363 "/usr/include/vector_types.h" 3
#if 0
typedef char1 
# 363
char1; 
#endif
# 364 "/usr/include/vector_types.h" 3
#if 0
typedef uchar1 
# 364
uchar1; 
#endif
# 365 "/usr/include/vector_types.h" 3
#if 0
typedef char2 
# 365
char2; 
#endif
# 366 "/usr/include/vector_types.h" 3
#if 0
typedef uchar2 
# 366
uchar2; 
#endif
# 367 "/usr/include/vector_types.h" 3
#if 0
typedef char3 
# 367
char3; 
#endif
# 368 "/usr/include/vector_types.h" 3
#if 0
typedef uchar3 
# 368
uchar3; 
#endif
# 369 "/usr/include/vector_types.h" 3
#if 0
typedef char4 
# 369
char4; 
#endif
# 370 "/usr/include/vector_types.h" 3
#if 0
typedef uchar4 
# 370
uchar4; 
#endif
# 371 "/usr/include/vector_types.h" 3
#if 0
typedef short1 
# 371
short1; 
#endif
# 372 "/usr/include/vector_types.h" 3
#if 0
typedef ushort1 
# 372
ushort1; 
#endif
# 373 "/usr/include/vector_types.h" 3
#if 0
typedef short2 
# 373
short2; 
#endif
# 374 "/usr/include/vector_types.h" 3
#if 0
typedef ushort2 
# 374
ushort2; 
#endif
# 375 "/usr/include/vector_types.h" 3
#if 0
typedef short3 
# 375
short3; 
#endif
# 376 "/usr/include/vector_types.h" 3
#if 0
typedef ushort3 
# 376
ushort3; 
#endif
# 377 "/usr/include/vector_types.h" 3
#if 0
typedef short4 
# 377
short4; 
#endif
# 378 "/usr/include/vector_types.h" 3
#if 0
typedef ushort4 
# 378
ushort4; 
#endif
# 379 "/usr/include/vector_types.h" 3
#if 0
typedef int1 
# 379
int1; 
#endif
# 380 "/usr/include/vector_types.h" 3
#if 0
typedef uint1 
# 380
uint1; 
#endif
# 381 "/usr/include/vector_types.h" 3
#if 0
typedef int2 
# 381
int2; 
#endif
# 382 "/usr/include/vector_types.h" 3
#if 0
typedef uint2 
# 382
uint2; 
#endif
# 383 "/usr/include/vector_types.h" 3
#if 0
typedef int3 
# 383
int3; 
#endif
# 384 "/usr/include/vector_types.h" 3
#if 0
typedef uint3 
# 384
uint3; 
#endif
# 385 "/usr/include/vector_types.h" 3
#if 0
typedef int4 
# 385
int4; 
#endif
# 386 "/usr/include/vector_types.h" 3
#if 0
typedef uint4 
# 386
uint4; 
#endif
# 387 "/usr/include/vector_types.h" 3
#if 0
typedef long1 
# 387
long1; 
#endif
# 388 "/usr/include/vector_types.h" 3
#if 0
typedef ulong1 
# 388
ulong1; 
#endif
# 389 "/usr/include/vector_types.h" 3
#if 0
typedef long2 
# 389
long2; 
#endif
# 390 "/usr/include/vector_types.h" 3
#if 0
typedef ulong2 
# 390
ulong2; 
#endif
# 391 "/usr/include/vector_types.h" 3
#if 0
typedef long3 
# 391
long3; 
#endif
# 392 "/usr/include/vector_types.h" 3
#if 0
typedef ulong3 
# 392
ulong3; 
#endif
# 393 "/usr/include/vector_types.h" 3
#if 0
typedef long4 
# 393
long4; 
#endif
# 394 "/usr/include/vector_types.h" 3
#if 0
typedef ulong4 
# 394
ulong4; 
#endif
# 395 "/usr/include/vector_types.h" 3
#if 0
typedef float1 
# 395
float1; 
#endif
# 396 "/usr/include/vector_types.h" 3
#if 0
typedef float2 
# 396
float2; 
#endif
# 397 "/usr/include/vector_types.h" 3
#if 0
typedef float3 
# 397
float3; 
#endif
# 398 "/usr/include/vector_types.h" 3
#if 0
typedef float4 
# 398
float4; 
#endif
# 399 "/usr/include/vector_types.h" 3
#if 0
typedef longlong1 
# 399
longlong1; 
#endif
# 400 "/usr/include/vector_types.h" 3
#if 0
typedef ulonglong1 
# 400
ulonglong1; 
#endif
# 401 "/usr/include/vector_types.h" 3
#if 0
typedef longlong2 
# 401
longlong2; 
#endif
# 402 "/usr/include/vector_types.h" 3
#if 0
typedef ulonglong2 
# 402
ulonglong2; 
#endif
# 403 "/usr/include/vector_types.h" 3
#if 0
typedef longlong3 
# 403
longlong3; 
#endif
# 404 "/usr/include/vector_types.h" 3
#if 0
typedef ulonglong3 
# 404
ulonglong3; 
#endif
# 405 "/usr/include/vector_types.h" 3
#if 0
typedef longlong4 
# 405
longlong4; 
#endif
# 406 "/usr/include/vector_types.h" 3
#if 0
typedef ulonglong4 
# 406
ulonglong4; 
#endif
# 407 "/usr/include/vector_types.h" 3
#if 0
typedef double1 
# 407
double1; 
#endif
# 408 "/usr/include/vector_types.h" 3
#if 0
typedef double2 
# 408
double2; 
#endif
# 409 "/usr/include/vector_types.h" 3
#if 0
typedef double3 
# 409
double3; 
#endif
# 410 "/usr/include/vector_types.h" 3
#if 0
typedef double4 
# 410
double4; 
#endif
# 418 "/usr/include/vector_types.h" 3
#if 0
# 418
struct dim3 { 
# 420
unsigned x, y, z; 
# 432
}; 
#endif
# 434 "/usr/include/vector_types.h" 3
#if 0
typedef dim3 
# 434
dim3; 
#endif
# 23 "/usr/include/x86_64-linux-gnu/bits/pthread_stack_min-dynamic.h" 3
extern "C" {
# 24
extern long __sysconf(int __name) noexcept(true); 
# 25
}
# 145 "/usr/lib/gcc/x86_64-linux-gnu/12/include/stddef.h" 3
typedef long ptrdiff_t; 
# 214 "/usr/lib/gcc/x86_64-linux-gnu/12/include/stddef.h" 3
typedef unsigned long size_t; 
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
# 435 "/usr/lib/gcc/x86_64-linux-gnu/12/include/stddef.h" 3
typedef 
# 424 "/usr/lib/gcc/x86_64-linux-gnu/12/include/stddef.h" 3
struct { 
# 425
long long __max_align_ll __attribute((__aligned__(__alignof__(long long)))); 
# 426
long double __max_align_ld __attribute((__aligned__(__alignof__(long double)))); 
# 435 "/usr/lib/gcc/x86_64-linux-gnu/12/include/stddef.h" 3
} max_align_t; 
# 442
typedef __decltype((nullptr)) nullptr_t; 
# 202 "/usr/include/driver_types.h" 3
#if 0
# 202
enum cudaError { 
# 209
cudaSuccess, 
# 215
cudaErrorInvalidValue, 
# 221
cudaErrorMemoryAllocation, 
# 227
cudaErrorInitializationError, 
# 234
cudaErrorCudartUnloading, 
# 241
cudaErrorProfilerDisabled, 
# 249
cudaErrorProfilerNotInitialized, 
# 256
cudaErrorProfilerAlreadyStarted, 
# 263
cudaErrorProfilerAlreadyStopped, 
# 272 "/usr/include/driver_types.h" 3
cudaErrorInvalidConfiguration, 
# 278
cudaErrorInvalidPitchValue = 12, 
# 284
cudaErrorInvalidSymbol, 
# 292
cudaErrorInvalidHostPointer = 16, 
# 300
cudaErrorInvalidDevicePointer, 
# 306
cudaErrorInvalidTexture, 
# 312
cudaErrorInvalidTextureBinding, 
# 319
cudaErrorInvalidChannelDescriptor, 
# 325
cudaErrorInvalidMemcpyDirection, 
# 335 "/usr/include/driver_types.h" 3
cudaErrorAddressOfConstant, 
# 344 "/usr/include/driver_types.h" 3
cudaErrorTextureFetchFailed, 
# 353 "/usr/include/driver_types.h" 3
cudaErrorTextureNotBound, 
# 362 "/usr/include/driver_types.h" 3
cudaErrorSynchronizationError, 
# 368
cudaErrorInvalidFilterSetting, 
# 374
cudaErrorInvalidNormSetting, 
# 382
cudaErrorMixedDeviceExecution, 
# 390
cudaErrorNotYetImplemented = 31, 
# 399 "/usr/include/driver_types.h" 3
cudaErrorMemoryValueTooLarge, 
# 406
cudaErrorStubLibrary = 34, 
# 413
cudaErrorInsufficientDriver, 
# 420
cudaErrorCallRequiresNewerDriver, 
# 426
cudaErrorInvalidSurface, 
# 432
cudaErrorDuplicateVariableName = 43, 
# 438
cudaErrorDuplicateTextureName, 
# 444
cudaErrorDuplicateSurfaceName, 
# 454 "/usr/include/driver_types.h" 3
cudaErrorDevicesUnavailable, 
# 467 "/usr/include/driver_types.h" 3
cudaErrorIncompatibleDriverContext = 49, 
# 473
cudaErrorMissingConfiguration = 52, 
# 482 "/usr/include/driver_types.h" 3
cudaErrorPriorLaunchFailure, 
# 489
cudaErrorLaunchMaxDepthExceeded = 65, 
# 497
cudaErrorLaunchFileScopedTex, 
# 505
cudaErrorLaunchFileScopedSurf, 
# 521 "/usr/include/driver_types.h" 3
cudaErrorSyncDepthExceeded, 
# 533 "/usr/include/driver_types.h" 3
cudaErrorLaunchPendingCountExceeded, 
# 539
cudaErrorInvalidDeviceFunction = 98, 
# 545
cudaErrorNoDevice = 100, 
# 552
cudaErrorInvalidDevice, 
# 557
cudaErrorDeviceNotLicensed, 
# 566 "/usr/include/driver_types.h" 3
cudaErrorSoftwareValidityNotEstablished, 
# 571
cudaErrorStartupFailure = 127, 
# 576
cudaErrorInvalidKernelImage = 200, 
# 586 "/usr/include/driver_types.h" 3
cudaErrorDeviceUninitialized, 
# 591
cudaErrorMapBufferObjectFailed = 205, 
# 596
cudaErrorUnmapBufferObjectFailed, 
# 602
cudaErrorArrayIsMapped, 
# 607
cudaErrorAlreadyMapped, 
# 615
cudaErrorNoKernelImageForDevice, 
# 620
cudaErrorAlreadyAcquired, 
# 625
cudaErrorNotMapped, 
# 631
cudaErrorNotMappedAsArray, 
# 637
cudaErrorNotMappedAsPointer, 
# 643
cudaErrorECCUncorrectable, 
# 649
cudaErrorUnsupportedLimit, 
# 655
cudaErrorDeviceAlreadyInUse, 
# 661
cudaErrorPeerAccessUnsupported, 
# 667
cudaErrorInvalidPtx, 
# 672
cudaErrorInvalidGraphicsContext, 
# 678
cudaErrorNvlinkUncorrectable, 
# 685
cudaErrorJitCompilerNotFound, 
# 692
cudaErrorUnsupportedPtxVersion, 
# 699
cudaErrorJitCompilationDisabled, 
# 704
cudaErrorUnsupportedExecAffinity, 
# 709
cudaErrorInvalidSource = 300, 
# 714
cudaErrorFileNotFound, 
# 719
cudaErrorSharedObjectSymbolNotFound, 
# 724
cudaErrorSharedObjectInitFailed, 
# 729
cudaErrorOperatingSystem, 
# 736
cudaErrorInvalidResourceHandle = 400, 
# 742
cudaErrorIllegalState, 
# 749
cudaErrorSymbolNotFound = 500, 
# 757
cudaErrorNotReady = 600, 
# 765
cudaErrorIllegalAddress = 700, 
# 774 "/usr/include/driver_types.h" 3
cudaErrorLaunchOutOfResources, 
# 785 "/usr/include/driver_types.h" 3
cudaErrorLaunchTimeout, 
# 791
cudaErrorLaunchIncompatibleTexturing, 
# 798
cudaErrorPeerAccessAlreadyEnabled, 
# 805
cudaErrorPeerAccessNotEnabled, 
# 818 "/usr/include/driver_types.h" 3
cudaErrorSetOnActiveProcess = 708, 
# 825
cudaErrorContextIsDestroyed, 
# 832
cudaErrorAssert, 
# 839
cudaErrorTooManyPeers, 
# 845
cudaErrorHostMemoryAlreadyRegistered, 
# 851
cudaErrorHostMemoryNotRegistered, 
# 860 "/usr/include/driver_types.h" 3
cudaErrorHardwareStackError, 
# 868
cudaErrorIllegalInstruction, 
# 877 "/usr/include/driver_types.h" 3
cudaErrorMisalignedAddress, 
# 888 "/usr/include/driver_types.h" 3
cudaErrorInvalidAddressSpace, 
# 896
cudaErrorInvalidPc, 
# 907 "/usr/include/driver_types.h" 3
cudaErrorLaunchFailure, 
# 916 "/usr/include/driver_types.h" 3
cudaErrorCooperativeLaunchTooLarge, 
# 921
cudaErrorNotPermitted = 800, 
# 927
cudaErrorNotSupported, 
# 936 "/usr/include/driver_types.h" 3
cudaErrorSystemNotReady, 
# 943
cudaErrorSystemDriverMismatch, 
# 952 "/usr/include/driver_types.h" 3
cudaErrorCompatNotSupportedOnDevice, 
# 957
cudaErrorMpsConnectionFailed, 
# 962
cudaErrorMpsRpcFailure, 
# 968
cudaErrorMpsServerNotReady, 
# 973
cudaErrorMpsMaxClientsReached, 
# 978
cudaErrorMpsMaxConnectionsReached, 
# 983
cudaErrorMpsClientTerminated, 
# 988
cudaErrorCdpNotSupported, 
# 993
cudaErrorCdpVersionMismatch, 
# 998
cudaErrorStreamCaptureUnsupported = 900, 
# 1004
cudaErrorStreamCaptureInvalidated, 
# 1010
cudaErrorStreamCaptureMerge, 
# 1015
cudaErrorStreamCaptureUnmatched, 
# 1021
cudaErrorStreamCaptureUnjoined, 
# 1028
cudaErrorStreamCaptureIsolation, 
# 1034
cudaErrorStreamCaptureImplicit, 
# 1040
cudaErrorCapturedEvent, 
# 1047
cudaErrorStreamCaptureWrongThread, 
# 1052
cudaErrorTimeout, 
# 1058
cudaErrorGraphExecUpdateFailure, 
# 1068 "/usr/include/driver_types.h" 3
cudaErrorExternalDevice, 
# 1074
cudaErrorInvalidClusterSize, 
# 1079
cudaErrorUnknown = 999, 
# 1087
cudaErrorApiFailureBase = 10000
# 1088
}; 
#endif
# 1093 "/usr/include/driver_types.h" 3
#if 0
# 1093
enum cudaChannelFormatKind { 
# 1095
cudaChannelFormatKindSigned, 
# 1096
cudaChannelFormatKindUnsigned, 
# 1097
cudaChannelFormatKindFloat, 
# 1098
cudaChannelFormatKindNone, 
# 1099
cudaChannelFormatKindNV12, 
# 1100
cudaChannelFormatKindUnsignedNormalized8X1, 
# 1101
cudaChannelFormatKindUnsignedNormalized8X2, 
# 1102
cudaChannelFormatKindUnsignedNormalized8X4, 
# 1103
cudaChannelFormatKindUnsignedNormalized16X1, 
# 1104
cudaChannelFormatKindUnsignedNormalized16X2, 
# 1105
cudaChannelFormatKindUnsignedNormalized16X4, 
# 1106
cudaChannelFormatKindSignedNormalized8X1, 
# 1107
cudaChannelFormatKindSignedNormalized8X2, 
# 1108
cudaChannelFormatKindSignedNormalized8X4, 
# 1109
cudaChannelFormatKindSignedNormalized16X1, 
# 1110
cudaChannelFormatKindSignedNormalized16X2, 
# 1111
cudaChannelFormatKindSignedNormalized16X4, 
# 1112
cudaChannelFormatKindUnsignedBlockCompressed1, 
# 1113
cudaChannelFormatKindUnsignedBlockCompressed1SRGB, 
# 1114
cudaChannelFormatKindUnsignedBlockCompressed2, 
# 1115
cudaChannelFormatKindUnsignedBlockCompressed2SRGB, 
# 1116
cudaChannelFormatKindUnsignedBlockCompressed3, 
# 1117
cudaChannelFormatKindUnsignedBlockCompressed3SRGB, 
# 1118
cudaChannelFormatKindUnsignedBlockCompressed4, 
# 1119
cudaChannelFormatKindSignedBlockCompressed4, 
# 1120
cudaChannelFormatKindUnsignedBlockCompressed5, 
# 1121
cudaChannelFormatKindSignedBlockCompressed5, 
# 1122
cudaChannelFormatKindUnsignedBlockCompressed6H, 
# 1123
cudaChannelFormatKindSignedBlockCompressed6H, 
# 1124
cudaChannelFormatKindUnsignedBlockCompressed7, 
# 1125
cudaChannelFormatKindUnsignedBlockCompressed7SRGB
# 1126
}; 
#endif
# 1131 "/usr/include/driver_types.h" 3
#if 0
# 1131
struct cudaChannelFormatDesc { 
# 1133
int x; 
# 1134
int y; 
# 1135
int z; 
# 1136
int w; 
# 1137
cudaChannelFormatKind f; 
# 1138
}; 
#endif
# 1143 "/usr/include/driver_types.h" 3
typedef struct cudaArray *cudaArray_t; 
# 1148
typedef const cudaArray *cudaArray_const_t; 
# 1150
struct cudaArray; 
# 1155
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
# 1160
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
# 1162
struct cudaMipmappedArray; 
# 1172 "/usr/include/driver_types.h" 3
#if 0
# 1172
struct cudaArraySparseProperties { 
# 1173
struct { 
# 1174
unsigned width; 
# 1175
unsigned height; 
# 1176
unsigned depth; 
# 1177
} tileExtent; 
# 1178
unsigned miptailFirstLevel; 
# 1179
unsigned long long miptailSize; 
# 1180
unsigned flags; 
# 1181
unsigned reserved[4]; 
# 1182
}; 
#endif
# 1187 "/usr/include/driver_types.h" 3
#if 0
# 1187
struct cudaArrayMemoryRequirements { 
# 1188
size_t size; 
# 1189
size_t alignment; 
# 1190
unsigned reserved[4]; 
# 1191
}; 
#endif
# 1196 "/usr/include/driver_types.h" 3
#if 0
# 1196
enum cudaMemoryType { 
# 1198
cudaMemoryTypeUnregistered, 
# 1199
cudaMemoryTypeHost, 
# 1200
cudaMemoryTypeDevice, 
# 1201
cudaMemoryTypeManaged
# 1202
}; 
#endif
# 1207 "/usr/include/driver_types.h" 3
#if 0
# 1207
enum cudaMemcpyKind { 
# 1209
cudaMemcpyHostToHost, 
# 1210
cudaMemcpyHostToDevice, 
# 1211
cudaMemcpyDeviceToHost, 
# 1212
cudaMemcpyDeviceToDevice, 
# 1213
cudaMemcpyDefault
# 1214
}; 
#endif
# 1221 "/usr/include/driver_types.h" 3
#if 0
# 1221
struct cudaPitchedPtr { 
# 1223
void *ptr; 
# 1224
size_t pitch; 
# 1225
size_t xsize; 
# 1226
size_t ysize; 
# 1227
}; 
#endif
# 1234 "/usr/include/driver_types.h" 3
#if 0
# 1234
struct cudaExtent { 
# 1236
size_t width; 
# 1237
size_t height; 
# 1238
size_t depth; 
# 1239
}; 
#endif
# 1246 "/usr/include/driver_types.h" 3
#if 0
# 1246
struct cudaPos { 
# 1248
size_t x; 
# 1249
size_t y; 
# 1250
size_t z; 
# 1251
}; 
#endif
# 1256 "/usr/include/driver_types.h" 3
#if 0
# 1256
struct cudaMemcpy3DParms { 
# 1258
cudaArray_t srcArray; 
# 1259
cudaPos srcPos; 
# 1260
cudaPitchedPtr srcPtr; 
# 1262
cudaArray_t dstArray; 
# 1263
cudaPos dstPos; 
# 1264
cudaPitchedPtr dstPtr; 
# 1266
cudaExtent extent; 
# 1267
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1268
}; 
#endif
# 1273 "/usr/include/driver_types.h" 3
#if 0
# 1273
struct cudaMemcpy3DPeerParms { 
# 1275
cudaArray_t srcArray; 
# 1276
cudaPos srcPos; 
# 1277
cudaPitchedPtr srcPtr; 
# 1278
int srcDevice; 
# 1280
cudaArray_t dstArray; 
# 1281
cudaPos dstPos; 
# 1282
cudaPitchedPtr dstPtr; 
# 1283
int dstDevice; 
# 1285
cudaExtent extent; 
# 1286
}; 
#endif
# 1291 "/usr/include/driver_types.h" 3
#if 0
# 1291
struct cudaMemsetParams { 
# 1292
void *dst; 
# 1293
size_t pitch; 
# 1294
unsigned value; 
# 1295
unsigned elementSize; 
# 1296
size_t width; 
# 1297
size_t height; 
# 1298
}; 
#endif
# 1303 "/usr/include/driver_types.h" 3
#if 0
# 1303
enum cudaAccessProperty { 
# 1304
cudaAccessPropertyNormal, 
# 1305
cudaAccessPropertyStreaming, 
# 1306
cudaAccessPropertyPersisting
# 1307
}; 
#endif
# 1320 "/usr/include/driver_types.h" 3
#if 0
# 1320
struct cudaAccessPolicyWindow { 
# 1321
void *base_ptr; 
# 1322
size_t num_bytes; 
# 1323
float hitRatio; 
# 1324
cudaAccessProperty hitProp; 
# 1325
cudaAccessProperty missProp; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1326
}; 
#endif
# 1338 "/usr/include/driver_types.h" 3
typedef void (*cudaHostFn_t)(void * userData); 
# 1343
#if 0
# 1343
struct cudaHostNodeParams { 
# 1344
cudaHostFn_t fn; 
# 1345
void *userData; 
# 1346
}; 
#endif
# 1351 "/usr/include/driver_types.h" 3
#if 0
# 1351
enum cudaStreamCaptureStatus { 
# 1352
cudaStreamCaptureStatusNone, 
# 1353
cudaStreamCaptureStatusActive, 
# 1354
cudaStreamCaptureStatusInvalidated
# 1356
}; 
#endif
# 1362 "/usr/include/driver_types.h" 3
#if 0
# 1362
enum cudaStreamCaptureMode { 
# 1363
cudaStreamCaptureModeGlobal, 
# 1364
cudaStreamCaptureModeThreadLocal, 
# 1365
cudaStreamCaptureModeRelaxed
# 1366
}; 
#endif
# 1368 "/usr/include/driver_types.h" 3
#if 0
# 1368
enum cudaSynchronizationPolicy { 
# 1369
cudaSyncPolicyAuto = 1, 
# 1370
cudaSyncPolicySpin, 
# 1371
cudaSyncPolicyYield, 
# 1372
cudaSyncPolicyBlockingSync
# 1373
}; 
#endif
# 1378 "/usr/include/driver_types.h" 3
#if 0
# 1378
enum cudaClusterSchedulingPolicy { 
# 1379
cudaClusterSchedulingPolicyDefault, 
# 1380
cudaClusterSchedulingPolicySpread, 
# 1381
cudaClusterSchedulingPolicyLoadBalancing
# 1382
}; 
#endif
# 1387 "/usr/include/driver_types.h" 3
#if 0
# 1387
enum cudaStreamUpdateCaptureDependenciesFlags { 
# 1388
cudaStreamAddCaptureDependencies, 
# 1389
cudaStreamSetCaptureDependencies
# 1390
}; 
#endif
# 1395 "/usr/include/driver_types.h" 3
#if 0
# 1395
enum cudaUserObjectFlags { 
# 1396
cudaUserObjectNoDestructorSync = 1
# 1397
}; 
#endif
# 1402 "/usr/include/driver_types.h" 3
#if 0
# 1402
enum cudaUserObjectRetainFlags { 
# 1403
cudaGraphUserObjectMove = 1
# 1404
}; 
#endif
# 1409 "/usr/include/driver_types.h" 3
struct cudaGraphicsResource; 
# 1414
#if 0
# 1414
enum cudaGraphicsRegisterFlags { 
# 1416
cudaGraphicsRegisterFlagsNone, 
# 1417
cudaGraphicsRegisterFlagsReadOnly, 
# 1418
cudaGraphicsRegisterFlagsWriteDiscard, 
# 1419
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
# 1420
cudaGraphicsRegisterFlagsTextureGather = 8
# 1421
}; 
#endif
# 1426 "/usr/include/driver_types.h" 3
#if 0
# 1426
enum cudaGraphicsMapFlags { 
# 1428
cudaGraphicsMapFlagsNone, 
# 1429
cudaGraphicsMapFlagsReadOnly, 
# 1430
cudaGraphicsMapFlagsWriteDiscard
# 1431
}; 
#endif
# 1436 "/usr/include/driver_types.h" 3
#if 0
# 1436
enum cudaGraphicsCubeFace { 
# 1438
cudaGraphicsCubeFacePositiveX, 
# 1439
cudaGraphicsCubeFaceNegativeX, 
# 1440
cudaGraphicsCubeFacePositiveY, 
# 1441
cudaGraphicsCubeFaceNegativeY, 
# 1442
cudaGraphicsCubeFacePositiveZ, 
# 1443
cudaGraphicsCubeFaceNegativeZ
# 1444
}; 
#endif
# 1449 "/usr/include/driver_types.h" 3
#if 0
# 1449
enum cudaResourceType { 
# 1451
cudaResourceTypeArray, 
# 1452
cudaResourceTypeMipmappedArray, 
# 1453
cudaResourceTypeLinear, 
# 1454
cudaResourceTypePitch2D
# 1455
}; 
#endif
# 1460 "/usr/include/driver_types.h" 3
#if 0
# 1460
enum cudaResourceViewFormat { 
# 1462
cudaResViewFormatNone, 
# 1463
cudaResViewFormatUnsignedChar1, 
# 1464
cudaResViewFormatUnsignedChar2, 
# 1465
cudaResViewFormatUnsignedChar4, 
# 1466
cudaResViewFormatSignedChar1, 
# 1467
cudaResViewFormatSignedChar2, 
# 1468
cudaResViewFormatSignedChar4, 
# 1469
cudaResViewFormatUnsignedShort1, 
# 1470
cudaResViewFormatUnsignedShort2, 
# 1471
cudaResViewFormatUnsignedShort4, 
# 1472
cudaResViewFormatSignedShort1, 
# 1473
cudaResViewFormatSignedShort2, 
# 1474
cudaResViewFormatSignedShort4, 
# 1475
cudaResViewFormatUnsignedInt1, 
# 1476
cudaResViewFormatUnsignedInt2, 
# 1477
cudaResViewFormatUnsignedInt4, 
# 1478
cudaResViewFormatSignedInt1, 
# 1479
cudaResViewFormatSignedInt2, 
# 1480
cudaResViewFormatSignedInt4, 
# 1481
cudaResViewFormatHalf1, 
# 1482
cudaResViewFormatHalf2, 
# 1483
cudaResViewFormatHalf4, 
# 1484
cudaResViewFormatFloat1, 
# 1485
cudaResViewFormatFloat2, 
# 1486
cudaResViewFormatFloat4, 
# 1487
cudaResViewFormatUnsignedBlockCompressed1, 
# 1488
cudaResViewFormatUnsignedBlockCompressed2, 
# 1489
cudaResViewFormatUnsignedBlockCompressed3, 
# 1490
cudaResViewFormatUnsignedBlockCompressed4, 
# 1491
cudaResViewFormatSignedBlockCompressed4, 
# 1492
cudaResViewFormatUnsignedBlockCompressed5, 
# 1493
cudaResViewFormatSignedBlockCompressed5, 
# 1494
cudaResViewFormatUnsignedBlockCompressed6H, 
# 1495
cudaResViewFormatSignedBlockCompressed6H, 
# 1496
cudaResViewFormatUnsignedBlockCompressed7
# 1497
}; 
#endif
# 1502 "/usr/include/driver_types.h" 3
#if 0
# 1502
struct cudaResourceDesc { 
# 1503
cudaResourceType resType; 
# 1505
union { 
# 1506
struct { 
# 1507
cudaArray_t array; 
# 1508
} array; 
# 1509
struct { 
# 1510
cudaMipmappedArray_t mipmap; 
# 1511
} mipmap; 
# 1512
struct { 
# 1513
void *devPtr; 
# 1514
cudaChannelFormatDesc desc; 
# 1515
size_t sizeInBytes; 
# 1516
} linear; 
# 1517
struct { 
# 1518
void *devPtr; 
# 1519
cudaChannelFormatDesc desc; 
# 1520
size_t width; 
# 1521
size_t height; 
# 1522
size_t pitchInBytes; 
# 1523
} pitch2D; 
# 1524
} res; 
# 1525
}; 
#endif
# 1530 "/usr/include/driver_types.h" 3
#if 0
# 1530
struct cudaResourceViewDesc { 
# 1532
cudaResourceViewFormat format; 
# 1533
size_t width; 
# 1534
size_t height; 
# 1535
size_t depth; 
# 1536
unsigned firstMipmapLevel; 
# 1537
unsigned lastMipmapLevel; 
# 1538
unsigned firstLayer; 
# 1539
unsigned lastLayer; 
# 1540
}; 
#endif
# 1545 "/usr/include/driver_types.h" 3
#if 0
# 1545
struct cudaPointerAttributes { 
# 1551
cudaMemoryType type; 
# 1562 "/usr/include/driver_types.h" 3
int device; 
# 1568
void *devicePointer; 
# 1577 "/usr/include/driver_types.h" 3
void *hostPointer; 
# 1578
}; 
#endif
# 1583 "/usr/include/driver_types.h" 3
#if 0
# 1583
struct cudaFuncAttributes { 
# 1590
size_t sharedSizeBytes; 
# 1596
size_t constSizeBytes; 
# 1601
size_t localSizeBytes; 
# 1608
int maxThreadsPerBlock; 
# 1613
int numRegs; 
# 1620
int ptxVersion; 
# 1627
int binaryVersion; 
# 1633
int cacheModeCA; 
# 1640
int maxDynamicSharedSizeBytes; 
# 1649 "/usr/include/driver_types.h" 3
int preferredShmemCarveout; 
# 1655
int clusterDimMustBeSet; 
# 1666 "/usr/include/driver_types.h" 3
int requiredClusterWidth; 
# 1667
int requiredClusterHeight; 
# 1668
int requiredClusterDepth; 
# 1674
int clusterSchedulingPolicyPreference; 
# 1696 "/usr/include/driver_types.h" 3
int nonPortableClusterSizeAllowed; 
# 1701
int reserved[16]; 
# 1702
}; 
#endif
# 1707 "/usr/include/driver_types.h" 3
#if 0
# 1707
enum cudaFuncAttribute { 
# 1709
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
# 1710
cudaFuncAttributePreferredSharedMemoryCarveout, 
# 1711
cudaFuncAttributeClusterDimMustBeSet, 
# 1712
cudaFuncAttributeRequiredClusterWidth, 
# 1713
cudaFuncAttributeRequiredClusterHeight, 
# 1714
cudaFuncAttributeRequiredClusterDepth, 
# 1715
cudaFuncAttributeNonPortableClusterSizeAllowed, 
# 1716
cudaFuncAttributeClusterSchedulingPolicyPreference, 
# 1717
cudaFuncAttributeMax
# 1718
}; 
#endif
# 1723 "/usr/include/driver_types.h" 3
#if 0
# 1723
enum cudaFuncCache { 
# 1725
cudaFuncCachePreferNone, 
# 1726
cudaFuncCachePreferShared, 
# 1727
cudaFuncCachePreferL1, 
# 1728
cudaFuncCachePreferEqual
# 1729
}; 
#endif
# 1735 "/usr/include/driver_types.h" 3
#if 0
# 1735
enum cudaSharedMemConfig { 
# 1737
cudaSharedMemBankSizeDefault, 
# 1738
cudaSharedMemBankSizeFourByte, 
# 1739
cudaSharedMemBankSizeEightByte
# 1740
}; 
#endif
# 1745 "/usr/include/driver_types.h" 3
#if 0
# 1745
enum cudaSharedCarveout { 
# 1746
cudaSharedmemCarveoutDefault = (-1), 
# 1747
cudaSharedmemCarveoutMaxShared = 100, 
# 1748
cudaSharedmemCarveoutMaxL1 = 0
# 1749
}; 
#endif
# 1754 "/usr/include/driver_types.h" 3
#if 0
# 1754
enum cudaComputeMode { 
# 1756
cudaComputeModeDefault, 
# 1757
cudaComputeModeExclusive, 
# 1758
cudaComputeModeProhibited, 
# 1759
cudaComputeModeExclusiveProcess
# 1760
}; 
#endif
# 1765 "/usr/include/driver_types.h" 3
#if 0
# 1765
enum cudaLimit { 
# 1767
cudaLimitStackSize, 
# 1768
cudaLimitPrintfFifoSize, 
# 1769
cudaLimitMallocHeapSize, 
# 1770
cudaLimitDevRuntimeSyncDepth, 
# 1771
cudaLimitDevRuntimePendingLaunchCount, 
# 1772
cudaLimitMaxL2FetchGranularity, 
# 1773
cudaLimitPersistingL2CacheSize
# 1774
}; 
#endif
# 1779 "/usr/include/driver_types.h" 3
#if 0
# 1779
enum cudaMemoryAdvise { 
# 1781
cudaMemAdviseSetReadMostly = 1, 
# 1782
cudaMemAdviseUnsetReadMostly, 
# 1783
cudaMemAdviseSetPreferredLocation, 
# 1784
cudaMemAdviseUnsetPreferredLocation, 
# 1785
cudaMemAdviseSetAccessedBy, 
# 1786
cudaMemAdviseUnsetAccessedBy
# 1787
}; 
#endif
# 1792 "/usr/include/driver_types.h" 3
#if 0
# 1792
enum cudaMemRangeAttribute { 
# 1794
cudaMemRangeAttributeReadMostly = 1, 
# 1795
cudaMemRangeAttributePreferredLocation, 
# 1796
cudaMemRangeAttributeAccessedBy, 
# 1797
cudaMemRangeAttributeLastPrefetchLocation
# 1798
}; 
#endif
# 1803 "/usr/include/driver_types.h" 3
#if 0
# 1803
enum cudaFlushGPUDirectRDMAWritesOptions { 
# 1804
cudaFlushGPUDirectRDMAWritesOptionHost = (1 << 0), 
# 1805
cudaFlushGPUDirectRDMAWritesOptionMemOps
# 1806
}; 
#endif
# 1811 "/usr/include/driver_types.h" 3
#if 0
# 1811
enum cudaGPUDirectRDMAWritesOrdering { 
# 1812
cudaGPUDirectRDMAWritesOrderingNone, 
# 1813
cudaGPUDirectRDMAWritesOrderingOwner = 100, 
# 1814
cudaGPUDirectRDMAWritesOrderingAllDevices = 200
# 1815
}; 
#endif
# 1820 "/usr/include/driver_types.h" 3
#if 0
# 1820
enum cudaFlushGPUDirectRDMAWritesScope { 
# 1821
cudaFlushGPUDirectRDMAWritesToOwner = 100, 
# 1822
cudaFlushGPUDirectRDMAWritesToAllDevices = 200
# 1823
}; 
#endif
# 1828 "/usr/include/driver_types.h" 3
#if 0
# 1828
enum cudaFlushGPUDirectRDMAWritesTarget { 
# 1829
cudaFlushGPUDirectRDMAWritesTargetCurrentDevice
# 1830
}; 
#endif
# 1836 "/usr/include/driver_types.h" 3
#if 0
# 1836
enum cudaDeviceAttr { 
# 1838
cudaDevAttrMaxThreadsPerBlock = 1, 
# 1839
cudaDevAttrMaxBlockDimX, 
# 1840
cudaDevAttrMaxBlockDimY, 
# 1841
cudaDevAttrMaxBlockDimZ, 
# 1842
cudaDevAttrMaxGridDimX, 
# 1843
cudaDevAttrMaxGridDimY, 
# 1844
cudaDevAttrMaxGridDimZ, 
# 1845
cudaDevAttrMaxSharedMemoryPerBlock, 
# 1846
cudaDevAttrTotalConstantMemory, 
# 1847
cudaDevAttrWarpSize, 
# 1848
cudaDevAttrMaxPitch, 
# 1849
cudaDevAttrMaxRegistersPerBlock, 
# 1850
cudaDevAttrClockRate, 
# 1851
cudaDevAttrTextureAlignment, 
# 1852
cudaDevAttrGpuOverlap, 
# 1853
cudaDevAttrMultiProcessorCount, 
# 1854
cudaDevAttrKernelExecTimeout, 
# 1855
cudaDevAttrIntegrated, 
# 1856
cudaDevAttrCanMapHostMemory, 
# 1857
cudaDevAttrComputeMode, 
# 1858
cudaDevAttrMaxTexture1DWidth, 
# 1859
cudaDevAttrMaxTexture2DWidth, 
# 1860
cudaDevAttrMaxTexture2DHeight, 
# 1861
cudaDevAttrMaxTexture3DWidth, 
# 1862
cudaDevAttrMaxTexture3DHeight, 
# 1863
cudaDevAttrMaxTexture3DDepth, 
# 1864
cudaDevAttrMaxTexture2DLayeredWidth, 
# 1865
cudaDevAttrMaxTexture2DLayeredHeight, 
# 1866
cudaDevAttrMaxTexture2DLayeredLayers, 
# 1867
cudaDevAttrSurfaceAlignment, 
# 1868
cudaDevAttrConcurrentKernels, 
# 1869
cudaDevAttrEccEnabled, 
# 1870
cudaDevAttrPciBusId, 
# 1871
cudaDevAttrPciDeviceId, 
# 1872
cudaDevAttrTccDriver, 
# 1873
cudaDevAttrMemoryClockRate, 
# 1874
cudaDevAttrGlobalMemoryBusWidth, 
# 1875
cudaDevAttrL2CacheSize, 
# 1876
cudaDevAttrMaxThreadsPerMultiProcessor, 
# 1877
cudaDevAttrAsyncEngineCount, 
# 1878
cudaDevAttrUnifiedAddressing, 
# 1879
cudaDevAttrMaxTexture1DLayeredWidth, 
# 1880
cudaDevAttrMaxTexture1DLayeredLayers, 
# 1881
cudaDevAttrMaxTexture2DGatherWidth = 45, 
# 1882
cudaDevAttrMaxTexture2DGatherHeight, 
# 1883
cudaDevAttrMaxTexture3DWidthAlt, 
# 1884
cudaDevAttrMaxTexture3DHeightAlt, 
# 1885
cudaDevAttrMaxTexture3DDepthAlt, 
# 1886
cudaDevAttrPciDomainId, 
# 1887
cudaDevAttrTexturePitchAlignment, 
# 1888
cudaDevAttrMaxTextureCubemapWidth, 
# 1889
cudaDevAttrMaxTextureCubemapLayeredWidth, 
# 1890
cudaDevAttrMaxTextureCubemapLayeredLayers, 
# 1891
cudaDevAttrMaxSurface1DWidth, 
# 1892
cudaDevAttrMaxSurface2DWidth, 
# 1893
cudaDevAttrMaxSurface2DHeight, 
# 1894
cudaDevAttrMaxSurface3DWidth, 
# 1895
cudaDevAttrMaxSurface3DHeight, 
# 1896
cudaDevAttrMaxSurface3DDepth, 
# 1897
cudaDevAttrMaxSurface1DLayeredWidth, 
# 1898
cudaDevAttrMaxSurface1DLayeredLayers, 
# 1899
cudaDevAttrMaxSurface2DLayeredWidth, 
# 1900
cudaDevAttrMaxSurface2DLayeredHeight, 
# 1901
cudaDevAttrMaxSurface2DLayeredLayers, 
# 1902
cudaDevAttrMaxSurfaceCubemapWidth, 
# 1903
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
# 1904
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
# 1905
cudaDevAttrMaxTexture1DLinearWidth, 
# 1906
cudaDevAttrMaxTexture2DLinearWidth, 
# 1907
cudaDevAttrMaxTexture2DLinearHeight, 
# 1908
cudaDevAttrMaxTexture2DLinearPitch, 
# 1909
cudaDevAttrMaxTexture2DMipmappedWidth, 
# 1910
cudaDevAttrMaxTexture2DMipmappedHeight, 
# 1911
cudaDevAttrComputeCapabilityMajor, 
# 1912
cudaDevAttrComputeCapabilityMinor, 
# 1913
cudaDevAttrMaxTexture1DMipmappedWidth, 
# 1914
cudaDevAttrStreamPrioritiesSupported, 
# 1915
cudaDevAttrGlobalL1CacheSupported, 
# 1916
cudaDevAttrLocalL1CacheSupported, 
# 1917
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
# 1918
cudaDevAttrMaxRegistersPerMultiprocessor, 
# 1919
cudaDevAttrManagedMemory, 
# 1920
cudaDevAttrIsMultiGpuBoard, 
# 1921
cudaDevAttrMultiGpuBoardGroupID, 
# 1922
cudaDevAttrHostNativeAtomicSupported, 
# 1923
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
# 1924
cudaDevAttrPageableMemoryAccess, 
# 1925
cudaDevAttrConcurrentManagedAccess, 
# 1926
cudaDevAttrComputePreemptionSupported, 
# 1927
cudaDevAttrCanUseHostPointerForRegisteredMem, 
# 1928
cudaDevAttrReserved92, 
# 1929
cudaDevAttrReserved93, 
# 1930
cudaDevAttrReserved94, 
# 1931
cudaDevAttrCooperativeLaunch, 
# 1932
cudaDevAttrCooperativeMultiDeviceLaunch, 
# 1933
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
# 1934
cudaDevAttrCanFlushRemoteWrites, 
# 1935
cudaDevAttrHostRegisterSupported, 
# 1936
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
# 1937
cudaDevAttrDirectManagedMemAccessFromHost, 
# 1938
cudaDevAttrMaxBlocksPerMultiprocessor = 106, 
# 1939
cudaDevAttrMaxPersistingL2CacheSize = 108, 
# 1940
cudaDevAttrMaxAccessPolicyWindowSize, 
# 1941
cudaDevAttrReservedSharedMemoryPerBlock = 111, 
# 1942
cudaDevAttrSparseCudaArraySupported, 
# 1943
cudaDevAttrHostRegisterReadOnlySupported, 
# 1944
cudaDevAttrTimelineSemaphoreInteropSupported, 
# 1945
cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114, 
# 1946
cudaDevAttrMemoryPoolsSupported, 
# 1947
cudaDevAttrGPUDirectRDMASupported, 
# 1948
cudaDevAttrGPUDirectRDMAFlushWritesOptions, 
# 1949
cudaDevAttrGPUDirectRDMAWritesOrdering, 
# 1950
cudaDevAttrMemoryPoolSupportedHandleTypes, 
# 1951
cudaDevAttrClusterLaunch, 
# 1952
cudaDevAttrDeferredMappingCudaArraySupported, 
# 1953
cudaDevAttrReserved122, 
# 1954
cudaDevAttrReserved123, 
# 1955
cudaDevAttrReserved124, 
# 1956
cudaDevAttrIpcEventSupport, 
# 1957
cudaDevAttrMemSyncDomainCount, 
# 1958
cudaDevAttrMax
# 1959
}; 
#endif
# 1964 "/usr/include/driver_types.h" 3
#if 0
# 1964
enum cudaMemPoolAttr { 
# 1974 "/usr/include/driver_types.h" 3
cudaMemPoolReuseFollowEventDependencies = 1, 
# 1981
cudaMemPoolReuseAllowOpportunistic, 
# 1989
cudaMemPoolReuseAllowInternalDependencies, 
# 2000 "/usr/include/driver_types.h" 3
cudaMemPoolAttrReleaseThreshold, 
# 2006
cudaMemPoolAttrReservedMemCurrent, 
# 2013
cudaMemPoolAttrReservedMemHigh, 
# 2019
cudaMemPoolAttrUsedMemCurrent, 
# 2026
cudaMemPoolAttrUsedMemHigh
# 2027
}; 
#endif
# 2032 "/usr/include/driver_types.h" 3
#if 0
# 2032
enum cudaMemLocationType { 
# 2033
cudaMemLocationTypeInvalid, 
# 2034
cudaMemLocationTypeDevice
# 2035
}; 
#endif
# 2042 "/usr/include/driver_types.h" 3
#if 0
# 2042
struct cudaMemLocation { 
# 2043
cudaMemLocationType type; 
# 2044
int id; 
# 2045
}; 
#endif
# 2050 "/usr/include/driver_types.h" 3
#if 0
# 2050
enum cudaMemAccessFlags { 
# 2051
cudaMemAccessFlagsProtNone, 
# 2052
cudaMemAccessFlagsProtRead, 
# 2053
cudaMemAccessFlagsProtReadWrite = 3
# 2054
}; 
#endif
# 2059 "/usr/include/driver_types.h" 3
#if 0
# 2059
struct cudaMemAccessDesc { 
# 2060
cudaMemLocation location; 
# 2061
cudaMemAccessFlags flags; 
# 2062
}; 
#endif
# 2067 "/usr/include/driver_types.h" 3
#if 0
# 2067
enum cudaMemAllocationType { 
# 2068
cudaMemAllocationTypeInvalid, 
# 2072
cudaMemAllocationTypePinned, 
# 2073
cudaMemAllocationTypeMax = 2147483647
# 2074
}; 
#endif
# 2079 "/usr/include/driver_types.h" 3
#if 0
# 2079
enum cudaMemAllocationHandleType { 
# 2080
cudaMemHandleTypeNone, 
# 2081
cudaMemHandleTypePosixFileDescriptor, 
# 2082
cudaMemHandleTypeWin32, 
# 2083
cudaMemHandleTypeWin32Kmt = 4
# 2084
}; 
#endif
# 2089 "/usr/include/driver_types.h" 3
#if 0
# 2089
struct cudaMemPoolProps { 
# 2090
cudaMemAllocationType allocType; 
# 2091
cudaMemAllocationHandleType handleTypes; 
# 2092
cudaMemLocation location; 
# 2099
void *win32SecurityAttributes; 
# 2100
unsigned char reserved[64]; 
# 2101
}; 
#endif
# 2106 "/usr/include/driver_types.h" 3
#if 0
# 2106
struct cudaMemPoolPtrExportData { 
# 2107
unsigned char reserved[64]; 
# 2108
}; 
#endif
# 2113 "/usr/include/driver_types.h" 3
#if 0
# 2113
struct cudaMemAllocNodeParams { 
# 2118
cudaMemPoolProps poolProps; 
# 2119
const cudaMemAccessDesc *accessDescs; 
# 2120
size_t accessDescCount; 
# 2121
size_t bytesize; 
# 2122
void *dptr; 
# 2123
}; 
#endif
# 2128 "/usr/include/driver_types.h" 3
#if 0
# 2128
enum cudaGraphMemAttributeType { 
# 2133
cudaGraphMemAttrUsedMemCurrent, 
# 2140
cudaGraphMemAttrUsedMemHigh, 
# 2147
cudaGraphMemAttrReservedMemCurrent, 
# 2154
cudaGraphMemAttrReservedMemHigh
# 2155
}; 
#endif
# 2161 "/usr/include/driver_types.h" 3
#if 0
# 2161
enum cudaDeviceP2PAttr { 
# 2162
cudaDevP2PAttrPerformanceRank = 1, 
# 2163
cudaDevP2PAttrAccessSupported, 
# 2164
cudaDevP2PAttrNativeAtomicSupported, 
# 2165
cudaDevP2PAttrCudaArrayAccessSupported
# 2166
}; 
#endif
# 2173 "/usr/include/driver_types.h" 3
#if 0
# 2173
struct CUuuid_st { 
# 2174
char bytes[16]; 
# 2175
}; 
#endif
# 2176 "/usr/include/driver_types.h" 3
#if 0
typedef CUuuid_st 
# 2176
CUuuid; 
#endif
# 2178 "/usr/include/driver_types.h" 3
#if 0
typedef CUuuid_st 
# 2178
cudaUUID_t; 
#endif
# 2183 "/usr/include/driver_types.h" 3
#if 0
# 2183
struct cudaDeviceProp { 
# 2185
char name[256]; 
# 2186
cudaUUID_t uuid; 
# 2187
char luid[8]; 
# 2188
unsigned luidDeviceNodeMask; 
# 2189
size_t totalGlobalMem; 
# 2190
size_t sharedMemPerBlock; 
# 2191
int regsPerBlock; 
# 2192
int warpSize; 
# 2193
size_t memPitch; 
# 2194
int maxThreadsPerBlock; 
# 2195
int maxThreadsDim[3]; 
# 2196
int maxGridSize[3]; 
# 2197
int clockRate; 
# 2198
size_t totalConstMem; 
# 2199
int major; 
# 2200
int minor; 
# 2201
size_t textureAlignment; 
# 2202
size_t texturePitchAlignment; 
# 2203
int deviceOverlap; 
# 2204
int multiProcessorCount; 
# 2205
int kernelExecTimeoutEnabled; 
# 2206
int integrated; 
# 2207
int canMapHostMemory; 
# 2208
int computeMode; 
# 2209
int maxTexture1D; 
# 2210
int maxTexture1DMipmap; 
# 2211
int maxTexture1DLinear; 
# 2212
int maxTexture2D[2]; 
# 2213
int maxTexture2DMipmap[2]; 
# 2214
int maxTexture2DLinear[3]; 
# 2215
int maxTexture2DGather[2]; 
# 2216
int maxTexture3D[3]; 
# 2217
int maxTexture3DAlt[3]; 
# 2218
int maxTextureCubemap; 
# 2219
int maxTexture1DLayered[2]; 
# 2220
int maxTexture2DLayered[3]; 
# 2221
int maxTextureCubemapLayered[2]; 
# 2222
int maxSurface1D; 
# 2223
int maxSurface2D[2]; 
# 2224
int maxSurface3D[3]; 
# 2225
int maxSurface1DLayered[2]; 
# 2226
int maxSurface2DLayered[3]; 
# 2227
int maxSurfaceCubemap; 
# 2228
int maxSurfaceCubemapLayered[2]; 
# 2229
size_t surfaceAlignment; 
# 2230
int concurrentKernels; 
# 2231
int ECCEnabled; 
# 2232
int pciBusID; 
# 2233
int pciDeviceID; 
# 2234
int pciDomainID; 
# 2235
int tccDriver; 
# 2236
int asyncEngineCount; 
# 2237
int unifiedAddressing; 
# 2238
int memoryClockRate; 
# 2239
int memoryBusWidth; 
# 2240
int l2CacheSize; 
# 2241
int persistingL2CacheMaxSize; 
# 2242
int maxThreadsPerMultiProcessor; 
# 2243
int streamPrioritiesSupported; 
# 2244
int globalL1CacheSupported; 
# 2245
int localL1CacheSupported; 
# 2246
size_t sharedMemPerMultiprocessor; 
# 2247
int regsPerMultiprocessor; 
# 2248
int managedMemory; 
# 2249
int isMultiGpuBoard; 
# 2250
int multiGpuBoardGroupID; 
# 2251
int hostNativeAtomicSupported; 
# 2252
int singleToDoublePrecisionPerfRatio; 
# 2253
int pageableMemoryAccess; 
# 2254
int concurrentManagedAccess; 
# 2255
int computePreemptionSupported; 
# 2256
int canUseHostPointerForRegisteredMem; 
# 2257
int cooperativeLaunch; 
# 2258
int cooperativeMultiDeviceLaunch; 
# 2259
size_t sharedMemPerBlockOptin; 
# 2260
int pageableMemoryAccessUsesHostPageTables; 
# 2261
int directManagedMemAccessFromHost; 
# 2262
int maxBlocksPerMultiProcessor; 
# 2263
int accessPolicyMaxWindowSize; 
# 2264
size_t reservedSharedMemPerBlock; 
# 2265
int hostRegisterSupported; 
# 2266
int sparseCudaArraySupported; 
# 2267
int hostRegisterReadOnlySupported; 
# 2268
int timelineSemaphoreInteropSupported; 
# 2269
int memoryPoolsSupported; 
# 2270
int gpuDirectRDMASupported; 
# 2271
unsigned gpuDirectRDMAFlushWritesOptions; 
# 2272
int gpuDirectRDMAWritesOrdering; 
# 2273
unsigned memoryPoolSupportedHandleTypes; 
# 2274
int deferredMappingCudaArraySupported; 
# 2275
int ipcEventSupported; 
# 2276
int clusterLaunch; 
# 2277
int unifiedFunctionPointers; 
# 2278
int reserved[63]; 
# 2279
}; 
#endif
# 2292 "/usr/include/driver_types.h" 3
#if 0
typedef 
# 2289
struct cudaIpcEventHandle_st { 
# 2291
char reserved[64]; 
# 2292
} cudaIpcEventHandle_t; 
#endif
# 2300 "/usr/include/driver_types.h" 3
#if 0
typedef 
# 2297
struct cudaIpcMemHandle_st { 
# 2299
char reserved[64]; 
# 2300
} cudaIpcMemHandle_t; 
#endif
# 2305 "/usr/include/driver_types.h" 3
#if 0
# 2305
enum cudaExternalMemoryHandleType { 
# 2309
cudaExternalMemoryHandleTypeOpaqueFd = 1, 
# 2313
cudaExternalMemoryHandleTypeOpaqueWin32, 
# 2317
cudaExternalMemoryHandleTypeOpaqueWin32Kmt, 
# 2321
cudaExternalMemoryHandleTypeD3D12Heap, 
# 2325
cudaExternalMemoryHandleTypeD3D12Resource, 
# 2329
cudaExternalMemoryHandleTypeD3D11Resource, 
# 2333
cudaExternalMemoryHandleTypeD3D11ResourceKmt, 
# 2337
cudaExternalMemoryHandleTypeNvSciBuf
# 2338
}; 
#endif
# 2380 "/usr/include/driver_types.h" 3
#if 0
# 2380
struct cudaExternalMemoryHandleDesc { 
# 2384
cudaExternalMemoryHandleType type; 
# 2385
union { 
# 2391
int fd; 
# 2407 "/usr/include/driver_types.h" 3
struct { 
# 2411
void *handle; 
# 2416
const void *name; 
# 2417
} win32; 
# 2422
const void *nvSciBufObject; 
# 2423
} handle; 
# 2427
unsigned long long size; 
# 2431
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 2432
}; 
#endif
# 2437 "/usr/include/driver_types.h" 3
#if 0
# 2437
struct cudaExternalMemoryBufferDesc { 
# 2441
unsigned long long offset; 
# 2445
unsigned long long size; 
# 2449
unsigned flags; 
# 2450
}; 
#endif
# 2455 "/usr/include/driver_types.h" 3
#if 0
# 2455
struct cudaExternalMemoryMipmappedArrayDesc { 
# 2460
unsigned long long offset; 
# 2464
cudaChannelFormatDesc formatDesc; 
# 2468
cudaExtent extent; 
# 2473
unsigned flags; 
# 2477
unsigned numLevels; 
# 2478
}; 
#endif
# 2483 "/usr/include/driver_types.h" 3
#if 0
# 2483
enum cudaExternalSemaphoreHandleType { 
# 2487
cudaExternalSemaphoreHandleTypeOpaqueFd = 1, 
# 2491
cudaExternalSemaphoreHandleTypeOpaqueWin32, 
# 2495
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, 
# 2499
cudaExternalSemaphoreHandleTypeD3D12Fence, 
# 2503
cudaExternalSemaphoreHandleTypeD3D11Fence, 
# 2507
cudaExternalSemaphoreHandleTypeNvSciSync, 
# 2511
cudaExternalSemaphoreHandleTypeKeyedMutex, 
# 2515
cudaExternalSemaphoreHandleTypeKeyedMutexKmt, 
# 2519
cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd, 
# 2523
cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
# 2524
}; 
#endif
# 2529 "/usr/include/driver_types.h" 3
#if 0
# 2529
struct cudaExternalSemaphoreHandleDesc { 
# 2533
cudaExternalSemaphoreHandleType type; 
# 2534
union { 
# 2541
int fd; 
# 2557 "/usr/include/driver_types.h" 3
struct { 
# 2561
void *handle; 
# 2566
const void *name; 
# 2567
} win32; 
# 2571
const void *nvSciSyncObj; 
# 2572
} handle; 
# 2576
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 2577
}; 
#endif
# 2582 "/usr/include/driver_types.h" 3
#if 0
# 2582
struct cudaExternalSemaphoreSignalParams_v1 { 
# 2583
struct { 
# 2587
struct { 
# 2591
unsigned long long value; 
# 2592
} fence; 
# 2593
union { 
# 2598
void *fence; 
# 2599
unsigned long long reserved; 
# 2600
} nvSciSync; 
# 2604
struct { 
# 2608
unsigned long long key; 
# 2609
} keyedMutex; 
# 2610
} params; 
# 2621 "/usr/include/driver_types.h" 3
unsigned flags; 
# 2622
}; 
#endif
# 2627 "/usr/include/driver_types.h" 3
#if 0
# 2627
struct cudaExternalSemaphoreWaitParams_v1 { 
# 2628
struct { 
# 2632
struct { 
# 2636
unsigned long long value; 
# 2637
} fence; 
# 2638
union { 
# 2643
void *fence; 
# 2644
unsigned long long reserved; 
# 2645
} nvSciSync; 
# 2649
struct { 
# 2653
unsigned long long key; 
# 2657
unsigned timeoutMs; 
# 2658
} keyedMutex; 
# 2659
} params; 
# 2670 "/usr/include/driver_types.h" 3
unsigned flags; 
# 2671
}; 
#endif
# 2676 "/usr/include/driver_types.h" 3
#if 0
# 2676
struct cudaExternalSemaphoreSignalParams { 
# 2677
struct { 
# 2681
struct { 
# 2685
unsigned long long value; 
# 2686
} fence; 
# 2687
union { 
# 2692
void *fence; 
# 2693
unsigned long long reserved; 
# 2694
} nvSciSync; 
# 2698
struct { 
# 2702
unsigned long long key; 
# 2703
} keyedMutex; 
# 2704
unsigned reserved[12]; 
# 2705
} params; 
# 2716 "/usr/include/driver_types.h" 3
unsigned flags; 
# 2717
unsigned reserved[16]; 
# 2718
}; 
#endif
# 2723 "/usr/include/driver_types.h" 3
#if 0
# 2723
struct cudaExternalSemaphoreWaitParams { 
# 2724
struct { 
# 2728
struct { 
# 2732
unsigned long long value; 
# 2733
} fence; 
# 2734
union { 
# 2739
void *fence; 
# 2740
unsigned long long reserved; 
# 2741
} nvSciSync; 
# 2745
struct { 
# 2749
unsigned long long key; 
# 2753
unsigned timeoutMs; 
# 2754
} keyedMutex; 
# 2755
unsigned reserved[10]; 
# 2756
} params; 
# 2767 "/usr/include/driver_types.h" 3
unsigned flags; 
# 2768
unsigned reserved[16]; 
# 2769
}; 
#endif
# 2780 "/usr/include/driver_types.h" 3
#if 0
typedef cudaError 
# 2780
cudaError_t; 
#endif
# 2785 "/usr/include/driver_types.h" 3
#if 0
typedef struct CUstream_st *
# 2785
cudaStream_t; 
#endif
# 2790 "/usr/include/driver_types.h" 3
#if 0
typedef struct CUevent_st *
# 2790
cudaEvent_t; 
#endif
# 2795 "/usr/include/driver_types.h" 3
#if 0
typedef cudaGraphicsResource *
# 2795
cudaGraphicsResource_t; 
#endif
# 2800 "/usr/include/driver_types.h" 3
#if 0
typedef struct CUexternalMemory_st *
# 2800
cudaExternalMemory_t; 
#endif
# 2805 "/usr/include/driver_types.h" 3
#if 0
typedef struct CUexternalSemaphore_st *
# 2805
cudaExternalSemaphore_t; 
#endif
# 2810 "/usr/include/driver_types.h" 3
#if 0
typedef struct CUgraph_st *
# 2810
cudaGraph_t; 
#endif
# 2815 "/usr/include/driver_types.h" 3
#if 0
typedef struct CUgraphNode_st *
# 2815
cudaGraphNode_t; 
#endif
# 2820 "/usr/include/driver_types.h" 3
#if 0
typedef struct CUuserObject_st *
# 2820
cudaUserObject_t; 
#endif
# 2825 "/usr/include/driver_types.h" 3
#if 0
typedef struct CUfunc_st *
# 2825
cudaFunction_t; 
#endif
# 2830 "/usr/include/driver_types.h" 3
#if 0
typedef struct CUmemPoolHandle_st *
# 2830
cudaMemPool_t; 
#endif
# 2835 "/usr/include/driver_types.h" 3
#if 0
# 2835
enum cudaCGScope { 
# 2836
cudaCGScopeInvalid, 
# 2837
cudaCGScopeGrid, 
# 2838
cudaCGScopeMultiGrid
# 2839
}; 
#endif
# 2844 "/usr/include/driver_types.h" 3
#if 0
# 2844
struct cudaLaunchParams { 
# 2846
void *func; 
# 2847
dim3 gridDim; 
# 2848
dim3 blockDim; 
# 2849
void **args; 
# 2850
size_t sharedMem; 
# 2851
cudaStream_t stream; 
# 2852
}; 
#endif
# 2857 "/usr/include/driver_types.h" 3
#if 0
# 2857
struct cudaKernelNodeParams { 
# 2858
void *func; 
# 2859
dim3 gridDim; 
# 2860
dim3 blockDim; 
# 2861
unsigned sharedMemBytes; 
# 2862
void **kernelParams; 
# 2863
void **extra; 
# 2864
}; 
#endif
# 2869 "/usr/include/driver_types.h" 3
#if 0
# 2869
struct cudaExternalSemaphoreSignalNodeParams { 
# 2870
cudaExternalSemaphore_t *extSemArray; 
# 2871
const cudaExternalSemaphoreSignalParams *paramsArray; 
# 2872
unsigned numExtSems; 
# 2873
}; 
#endif
# 2878 "/usr/include/driver_types.h" 3
#if 0
# 2878
struct cudaExternalSemaphoreWaitNodeParams { 
# 2879
cudaExternalSemaphore_t *extSemArray; 
# 2880
const cudaExternalSemaphoreWaitParams *paramsArray; 
# 2881
unsigned numExtSems; 
# 2882
}; 
#endif
# 2887 "/usr/include/driver_types.h" 3
#if 0
# 2887
enum cudaGraphNodeType { 
# 2888
cudaGraphNodeTypeKernel, 
# 2889
cudaGraphNodeTypeMemcpy, 
# 2890
cudaGraphNodeTypeMemset, 
# 2891
cudaGraphNodeTypeHost, 
# 2892
cudaGraphNodeTypeGraph, 
# 2893
cudaGraphNodeTypeEmpty, 
# 2894
cudaGraphNodeTypeWaitEvent, 
# 2895
cudaGraphNodeTypeEventRecord, 
# 2896
cudaGraphNodeTypeExtSemaphoreSignal, 
# 2897
cudaGraphNodeTypeExtSemaphoreWait, 
# 2898
cudaGraphNodeTypeMemAlloc, 
# 2899
cudaGraphNodeTypeMemFree, 
# 2900
cudaGraphNodeTypeCount
# 2901
}; 
#endif
# 2906 "/usr/include/driver_types.h" 3
typedef struct CUgraphExec_st *cudaGraphExec_t; 
# 2911
#if 0
# 2911
enum cudaGraphExecUpdateResult { 
# 2912
cudaGraphExecUpdateSuccess, 
# 2913
cudaGraphExecUpdateError, 
# 2914
cudaGraphExecUpdateErrorTopologyChanged, 
# 2915
cudaGraphExecUpdateErrorNodeTypeChanged, 
# 2916
cudaGraphExecUpdateErrorFunctionChanged, 
# 2917
cudaGraphExecUpdateErrorParametersChanged, 
# 2918
cudaGraphExecUpdateErrorNotSupported, 
# 2919
cudaGraphExecUpdateErrorUnsupportedFunctionChange, 
# 2920
cudaGraphExecUpdateErrorAttributesChanged
# 2921
}; 
#endif
# 2932 "/usr/include/driver_types.h" 3
#if 0
typedef 
# 2926
enum cudaGraphInstantiateResult { 
# 2927
cudaGraphInstantiateSuccess, 
# 2928
cudaGraphInstantiateError, 
# 2929
cudaGraphInstantiateInvalidStructure, 
# 2930
cudaGraphInstantiateNodeOperationNotSupported, 
# 2931
cudaGraphInstantiateMultipleDevicesNotSupported
# 2932
} cudaGraphInstantiateResult; 
#endif
# 2943 "/usr/include/driver_types.h" 3
#if 0
typedef 
# 2937
struct cudaGraphInstantiateParams_st { 
# 2939
unsigned long long flags; 
# 2940
cudaStream_t uploadStream; 
# 2941
cudaGraphNode_t errNode_out; 
# 2942
cudaGraphInstantiateResult result_out; 
# 2943
} cudaGraphInstantiateParams; 
#endif
# 2965 "/usr/include/driver_types.h" 3
#if 0
typedef 
# 2948
struct cudaGraphExecUpdateResultInfo_st { 
# 2952
cudaGraphExecUpdateResult result; 
# 2959
cudaGraphNode_t errorNode; 
# 2964
cudaGraphNode_t errorFromNode; 
# 2965
} cudaGraphExecUpdateResultInfo; 
#endif
# 2971 "/usr/include/driver_types.h" 3
#if 0
# 2971
enum cudaGetDriverEntryPointFlags { 
# 2972
cudaEnableDefault, 
# 2973
cudaEnableLegacyStream, 
# 2974
cudaEnablePerThreadDefaultStream
# 2975
}; 
#endif
# 2980 "/usr/include/driver_types.h" 3
#if 0
# 2980
enum cudaDriverEntryPointQueryResult { 
# 2981
cudaDriverEntryPointSuccess, 
# 2982
cudaDriverEntryPointSymbolNotFound, 
# 2983
cudaDriverEntryPointVersionNotSufficent
# 2984
}; 
#endif
# 2989 "/usr/include/driver_types.h" 3
#if 0
# 2989
enum cudaGraphDebugDotFlags { 
# 2990
cudaGraphDebugDotFlagsVerbose = (1 << 0), 
# 2991
cudaGraphDebugDotFlagsKernelNodeParams = (1 << 2), 
# 2992
cudaGraphDebugDotFlagsMemcpyNodeParams = (1 << 3), 
# 2993
cudaGraphDebugDotFlagsMemsetNodeParams = (1 << 4), 
# 2994
cudaGraphDebugDotFlagsHostNodeParams = (1 << 5), 
# 2995
cudaGraphDebugDotFlagsEventNodeParams = (1 << 6), 
# 2996
cudaGraphDebugDotFlagsExtSemasSignalNodeParams = (1 << 7), 
# 2997
cudaGraphDebugDotFlagsExtSemasWaitNodeParams = (1 << 8), 
# 2998
cudaGraphDebugDotFlagsKernelNodeAttributes = (1 << 9), 
# 2999
cudaGraphDebugDotFlagsHandles = (1 << 10)
# 3000
}; 
#endif
# 3005 "/usr/include/driver_types.h" 3
#if 0
# 3005
enum cudaGraphInstantiateFlags { 
# 3006
cudaGraphInstantiateFlagAutoFreeOnLaunch = 1, 
# 3007
cudaGraphInstantiateFlagUpload, 
# 3008
cudaGraphInstantiateFlagDeviceLaunch = 4, 
# 3009
cudaGraphInstantiateFlagUseNodePriority = 8
# 3011
}; 
#endif
# 3016 "/usr/include/driver_types.h" 3
#if 0
typedef 
# 3013
enum cudaLaunchMemSyncDomain { 
# 3014
cudaLaunchMemSyncDomainDefault, 
# 3015
cudaLaunchMemSyncDomainRemote
# 3016
} cudaLaunchMemSyncDomain; 
#endif
# 3021 "/usr/include/driver_types.h" 3
#if 0
typedef 
# 3018
struct cudaLaunchMemSyncDomainMap_st { 
# 3019
unsigned char default_; 
# 3020
unsigned char remote; 
# 3021
} cudaLaunchMemSyncDomainMap; 
#endif
# 3067 "/usr/include/driver_types.h" 3
#if 0
typedef 
# 3026 "/usr/include/driver_types.h" 3
enum cudaLaunchAttributeID { 
# 3027
cudaLaunchAttributeIgnore, 
# 3028
cudaLaunchAttributeAccessPolicyWindow, 
# 3029
cudaLaunchAttributeCooperative, 
# 3030
cudaLaunchAttributeSynchronizationPolicy, 
# 3031
cudaLaunchAttributeClusterDimension, 
# 3032
cudaLaunchAttributeClusterSchedulingPolicyPreference, 
# 3033
cudaLaunchAttributeProgrammaticStreamSerialization, 
# 3044 "/usr/include/driver_types.h" 3
cudaLaunchAttributeProgrammaticEvent, 
# 3064 "/usr/include/driver_types.h" 3
cudaLaunchAttributePriority, 
# 3065
cudaLaunchAttributeMemSyncDomainMap, 
# 3066
cudaLaunchAttributeMemSyncDomain
# 3067
} cudaLaunchAttributeID; 
#endif
# 3092 "/usr/include/driver_types.h" 3
#if 0
typedef 
# 3072
union cudaLaunchAttributeValue { 
# 3073
char pad[64]; 
# 3074
cudaAccessPolicyWindow accessPolicyWindow; 
# 3075
int cooperative; 
# 3076
cudaSynchronizationPolicy syncPolicy; 
# 3077
struct { 
# 3078
unsigned x; 
# 3079
unsigned y; 
# 3080
unsigned z; 
# 3081
} clusterDim; 
# 3082
cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference; 
# 3083
int programmaticStreamSerializationAllowed; 
# 3084
struct { 
# 3085
cudaEvent_t event; 
# 3086
int flags; 
# 3087
int triggerAtBlockStart; 
# 3088
} programmaticEvent; 
# 3089
int priority; 
# 3090
cudaLaunchMemSyncDomainMap memSyncDomainMap; 
# 3091
cudaLaunchMemSyncDomain memSyncDomain; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 3092
} cudaLaunchAttributeValue; 
#endif
# 3101 "/usr/include/driver_types.h" 3
#if 0
typedef 
# 3097
struct cudaLaunchAttribute_st { 
# 3098
cudaLaunchAttributeID id; 
# 3099
char pad[(8) - sizeof(cudaLaunchAttributeID)]; 
# 3100
cudaLaunchAttributeValue val; 
# 3101
} cudaLaunchAttribute; 
#endif
# 3113 "/usr/include/driver_types.h" 3
#if 0
typedef 
# 3106
struct cudaLaunchConfig_st { 
# 3107
dim3 gridDim; 
# 3108
dim3 blockDim; 
# 3109
size_t dynamicSmemBytes; 
# 3110
cudaStream_t stream; 
# 3111
cudaLaunchAttribute *attrs; 
# 3112
unsigned numAttrs; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 3113
} cudaLaunchConfig_t; 
#endif
# 84 "/usr/include/surface_types.h" 3
#if 0
# 84
enum cudaSurfaceBoundaryMode { 
# 86
cudaBoundaryModeZero, 
# 87
cudaBoundaryModeClamp, 
# 88
cudaBoundaryModeTrap
# 89
}; 
#endif
# 94 "/usr/include/surface_types.h" 3
#if 0
# 94
enum cudaSurfaceFormatMode { 
# 96
cudaFormatModeForced, 
# 97
cudaFormatModeAuto
# 98
}; 
#endif
# 103 "/usr/include/surface_types.h" 3
#if 0
typedef unsigned long long 
# 103
cudaSurfaceObject_t; 
#endif
# 84 "/usr/include/texture_types.h" 3
#if 0
# 84
enum cudaTextureAddressMode { 
# 86
cudaAddressModeWrap, 
# 87
cudaAddressModeClamp, 
# 88
cudaAddressModeMirror, 
# 89
cudaAddressModeBorder
# 90
}; 
#endif
# 95 "/usr/include/texture_types.h" 3
#if 0
# 95
enum cudaTextureFilterMode { 
# 97
cudaFilterModePoint, 
# 98
cudaFilterModeLinear
# 99
}; 
#endif
# 104 "/usr/include/texture_types.h" 3
#if 0
# 104
enum cudaTextureReadMode { 
# 106
cudaReadModeElementType, 
# 107
cudaReadModeNormalizedFloat
# 108
}; 
#endif
# 113 "/usr/include/texture_types.h" 3
#if 0
# 113
struct cudaTextureDesc { 
# 118
cudaTextureAddressMode addressMode[3]; 
# 122
cudaTextureFilterMode filterMode; 
# 126
cudaTextureReadMode readMode; 
# 130
int sRGB; 
# 134
float borderColor[4]; 
# 138
int normalizedCoords; 
# 142
unsigned maxAnisotropy; 
# 146
cudaTextureFilterMode mipmapFilterMode; 
# 150
float mipmapLevelBias; 
# 154
float minMipmapLevelClamp; 
# 158
float maxMipmapLevelClamp; 
# 162
int disableTrilinearOptimization; 
# 166
int seamlessCubemap; 
# 167
}; 
#endif
# 172 "/usr/include/texture_types.h" 3
#if 0
typedef unsigned long long 
# 172
cudaTextureObject_t; 
#endif
# 87 "/usr/include/library_types.h" 3
typedef 
# 55
enum cudaDataType_t { 
# 57
CUDA_R_16F = 2, 
# 58
CUDA_C_16F = 6, 
# 59
CUDA_R_16BF = 14, 
# 60
CUDA_C_16BF, 
# 61
CUDA_R_32F = 0, 
# 62
CUDA_C_32F = 4, 
# 63
CUDA_R_64F = 1, 
# 64
CUDA_C_64F = 5, 
# 65
CUDA_R_4I = 16, 
# 66
CUDA_C_4I, 
# 67
CUDA_R_4U, 
# 68
CUDA_C_4U, 
# 69
CUDA_R_8I = 3, 
# 70
CUDA_C_8I = 7, 
# 71
CUDA_R_8U, 
# 72
CUDA_C_8U, 
# 73
CUDA_R_16I = 20, 
# 74
CUDA_C_16I, 
# 75
CUDA_R_16U, 
# 76
CUDA_C_16U, 
# 77
CUDA_R_32I = 10, 
# 78
CUDA_C_32I, 
# 79
CUDA_R_32U, 
# 80
CUDA_C_32U, 
# 81
CUDA_R_64I = 24, 
# 82
CUDA_C_64I, 
# 83
CUDA_R_64U, 
# 84
CUDA_C_64U, 
# 85
CUDA_R_8F_E4M3, 
# 86
CUDA_R_8F_E5M2
# 87
} cudaDataType; 
# 95
typedef 
# 90
enum libraryPropertyType_t { 
# 92
MAJOR_VERSION, 
# 93
MINOR_VERSION, 
# 94
PATCH_LEVEL
# 95
} libraryPropertyType; 
# 296 "/usr/include/x86_64-linux-gnu/c++/12/bits/c++config.h" 3
namespace std { 
# 298
typedef unsigned long size_t; 
# 299
typedef long ptrdiff_t; 
# 302
typedef __decltype((nullptr)) nullptr_t; 
# 305
#pragma GCC visibility push ( default )
# 308
__attribute((__noreturn__, __always_inline__)) inline void 
# 309
__terminate() noexcept 
# 310
{ 
# 311
void terminate() noexcept __attribute((__noreturn__)); 
# 312
terminate(); 
# 313
} 
#pragma GCC visibility pop
}
# 329 "/usr/include/x86_64-linux-gnu/c++/12/bits/c++config.h" 3
namespace std { 
# 331
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 332
}
# 333
namespace __gnu_cxx { 
# 335
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 336
}
# 515 "/usr/include/x86_64-linux-gnu/c++/12/bits/c++config.h" 3
namespace std { 
# 517
#pragma GCC visibility push ( default )
# 523
constexpr bool __is_constant_evaluated() noexcept 
# 524
{ 
# 530
return __builtin_is_constant_evaluated(); 
# 534
} 
#pragma GCC visibility pop
}
# 34 "/usr/include/stdlib.h" 3
extern "C" {
# 74 "/usr/include/x86_64-linux-gnu/bits/floatn.h" 3
typedef float __complex__ __cfloat128 __attribute((__mode__(__TC__))); 
# 86 "/usr/include/x86_64-linux-gnu/bits/floatn.h" 3
typedef __float128 _Float128; 
# 214 "/usr/include/x86_64-linux-gnu/bits/floatn-common.h" 3
typedef float _Float32; 
# 251 "/usr/include/x86_64-linux-gnu/bits/floatn-common.h" 3
typedef double _Float64; 
# 268 "/usr/include/x86_64-linux-gnu/bits/floatn-common.h" 3
typedef double _Float32x; 
# 285 "/usr/include/x86_64-linux-gnu/bits/floatn-common.h" 3
typedef long double _Float64x; 
# 63 "/usr/include/stdlib.h" 3
typedef 
# 60
struct { 
# 61
int quot; 
# 62
int rem; 
# 63
} div_t; 
# 71
typedef 
# 68
struct { 
# 69
long quot; 
# 70
long rem; 
# 71
} ldiv_t; 
# 81
__extension__ typedef 
# 78
struct { 
# 79
long long quot; 
# 80
long long rem; 
# 81
} lldiv_t; 
# 98 "/usr/include/stdlib.h" 3
extern size_t __ctype_get_mb_cur_max() noexcept(true); 
# 102
extern double atof(const char * __nptr) noexcept(true)
# 103
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 105
extern int atoi(const char * __nptr) noexcept(true)
# 106
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 108
extern long atol(const char * __nptr) noexcept(true)
# 109
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 113
__extension__ extern long long atoll(const char * __nptr) noexcept(true)
# 114
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 118
extern double strtod(const char *__restrict__ __nptr, char **__restrict__ __endptr) noexcept(true)
# 120
 __attribute((__nonnull__(1))); 
# 124
extern float strtof(const char *__restrict__ __nptr, char **__restrict__ __endptr) noexcept(true)
# 125
 __attribute((__nonnull__(1))); 
# 127
extern long double strtold(const char *__restrict__ __nptr, char **__restrict__ __endptr) noexcept(true)
# 129
 __attribute((__nonnull__(1))); 
# 141 "/usr/include/stdlib.h" 3
extern _Float32 strtof32(const char *__restrict__ __nptr, char **__restrict__ __endptr) noexcept(true)
# 143
 __attribute((__nonnull__(1))); 
# 147
extern _Float64 strtof64(const char *__restrict__ __nptr, char **__restrict__ __endptr) noexcept(true)
# 149
 __attribute((__nonnull__(1))); 
# 153
extern _Float128 strtof128(const char *__restrict__ __nptr, char **__restrict__ __endptr) noexcept(true)
# 155
 __attribute((__nonnull__(1))); 
# 159
extern _Float32x strtof32x(const char *__restrict__ __nptr, char **__restrict__ __endptr) noexcept(true)
# 161
 __attribute((__nonnull__(1))); 
# 165
extern _Float64x strtof64x(const char *__restrict__ __nptr, char **__restrict__ __endptr) noexcept(true)
# 167
 __attribute((__nonnull__(1))); 
# 177 "/usr/include/stdlib.h" 3
extern long strtol(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtol")
# 179
 __attribute((__nonnull__(1))); 
# 181
extern unsigned long strtoul(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtoul")
# 183
 __attribute((__nonnull__(1))); 
# 188
__extension__ extern long long strtoq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtoll")
# 190
 __attribute((__nonnull__(1))); 
# 193
__extension__ extern unsigned long long strtouq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtoull")
# 195
 __attribute((__nonnull__(1))); 
# 201
__extension__ extern long long strtoll(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtoll")
# 203
 __attribute((__nonnull__(1))); 
# 206
__extension__ extern unsigned long long strtoull(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtoull")
# 208
 __attribute((__nonnull__(1))); 
# 215
extern long strtol(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtol")
# 218
 __attribute((__nonnull__(1))); 
# 219
extern unsigned long strtoul(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtoul")
# 223
 __attribute((__nonnull__(1))); 
# 226
__extension__ extern long long strtoq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtoll")
# 229
 __attribute((__nonnull__(1))); 
# 231
__extension__ extern unsigned long long strtouq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtoull")
# 235
 __attribute((__nonnull__(1))); 
# 238
__extension__ extern long long strtoll(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtoll")
# 241
 __attribute((__nonnull__(1))); 
# 243
__extension__ extern unsigned long long strtoull(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) noexcept(true) __asm__("__isoc23_strtoull")
# 247
 __attribute((__nonnull__(1))); 
# 278 "/usr/include/stdlib.h" 3
extern int strfromd(char * __dest, size_t __size, const char * __format, double __f) noexcept(true)
# 280
 __attribute((__nonnull__(3))); 
# 282
extern int strfromf(char * __dest, size_t __size, const char * __format, float __f) noexcept(true)
# 284
 __attribute((__nonnull__(3))); 
# 286
extern int strfroml(char * __dest, size_t __size, const char * __format, long double __f) noexcept(true)
# 288
 __attribute((__nonnull__(3))); 
# 298 "/usr/include/stdlib.h" 3
extern int strfromf32(char * __dest, size_t __size, const char * __format, _Float32 __f) noexcept(true)
# 300
 __attribute((__nonnull__(3))); 
# 304
extern int strfromf64(char * __dest, size_t __size, const char * __format, _Float64 __f) noexcept(true)
# 306
 __attribute((__nonnull__(3))); 
# 310
extern int strfromf128(char * __dest, size_t __size, const char * __format, _Float128 __f) noexcept(true)
# 312
 __attribute((__nonnull__(3))); 
# 316
extern int strfromf32x(char * __dest, size_t __size, const char * __format, _Float32x __f) noexcept(true)
# 318
 __attribute((__nonnull__(3))); 
# 322
extern int strfromf64x(char * __dest, size_t __size, const char * __format, _Float64x __f) noexcept(true)
# 324
 __attribute((__nonnull__(3))); 
# 27 "/usr/include/x86_64-linux-gnu/bits/types/__locale_t.h" 3
struct __locale_struct { 
# 30
struct __locale_data *__locales[13]; 
# 33
const unsigned short *__ctype_b; 
# 34
const int *__ctype_tolower; 
# 35
const int *__ctype_toupper; 
# 38
const char *__names[13]; 
# 39
}; 
# 41
typedef __locale_struct *__locale_t; 
# 24 "/usr/include/x86_64-linux-gnu/bits/types/locale_t.h" 3
typedef __locale_t locale_t; 
# 340 "/usr/include/stdlib.h" 3
extern long strtol_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) noexcept(true) __asm__("__isoc23_strtol_l")
# 342
 __attribute((__nonnull__(1, 4))); 
# 344
extern unsigned long strtoul_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) noexcept(true) __asm__("__isoc23_strtoul_l")
# 347
 __attribute((__nonnull__(1, 4))); 
# 350
__extension__ extern long long strtoll_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) noexcept(true) __asm__("__isoc23_strtoll_l")
# 353
 __attribute((__nonnull__(1, 4))); 
# 356
__extension__ extern unsigned long long strtoull_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) noexcept(true) __asm__("__isoc23_strtoull_l")
# 359
 __attribute((__nonnull__(1, 4))); 
# 365
extern long strtol_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) noexcept(true) __asm__("__isoc23_strtol_l")
# 369
 __attribute((__nonnull__(1, 4))); 
# 370
extern unsigned long strtoul_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) noexcept(true) __asm__("__isoc23_strtoul_l")
# 375
 __attribute((__nonnull__(1, 4))); 
# 377
__extension__ extern long long strtoll_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) noexcept(true) __asm__("__isoc23_strtoll_l")
# 382
 __attribute((__nonnull__(1, 4))); 
# 384
__extension__ extern unsigned long long strtoull_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) noexcept(true) __asm__("__isoc23_strtoull_l")
# 389
 __attribute((__nonnull__(1, 4))); 
# 415 "/usr/include/stdlib.h" 3
extern double strtod_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) noexcept(true)
# 417
 __attribute((__nonnull__(1, 3))); 
# 419
extern float strtof_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) noexcept(true)
# 421
 __attribute((__nonnull__(1, 3))); 
# 423
extern long double strtold_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) noexcept(true)
# 426
 __attribute((__nonnull__(1, 3))); 
# 436 "/usr/include/stdlib.h" 3
extern _Float32 strtof32_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) noexcept(true)
# 439
 __attribute((__nonnull__(1, 3))); 
# 443
extern _Float64 strtof64_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) noexcept(true)
# 446
 __attribute((__nonnull__(1, 3))); 
# 450
extern _Float128 strtof128_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) noexcept(true)
# 453
 __attribute((__nonnull__(1, 3))); 
# 457
extern _Float32x strtof32x_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) noexcept(true)
# 460
 __attribute((__nonnull__(1, 3))); 
# 464
extern _Float64x strtof64x_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) noexcept(true)
# 467
 __attribute((__nonnull__(1, 3))); 
# 505 "/usr/include/stdlib.h" 3
extern char *l64a(long __n) noexcept(true); 
# 508
extern long a64l(const char * __s) noexcept(true)
# 509
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 27 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" {
# 31 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
typedef unsigned char __u_char; 
# 32
typedef unsigned short __u_short; 
# 33
typedef unsigned __u_int; 
# 34
typedef unsigned long __u_long; 
# 37
typedef signed char __int8_t; 
# 38
typedef unsigned char __uint8_t; 
# 39
typedef signed short __int16_t; 
# 40
typedef unsigned short __uint16_t; 
# 41
typedef signed int __int32_t; 
# 42
typedef unsigned __uint32_t; 
# 44
typedef signed long __int64_t; 
# 45
typedef unsigned long __uint64_t; 
# 52
typedef __int8_t __int_least8_t; 
# 53
typedef __uint8_t __uint_least8_t; 
# 54
typedef __int16_t __int_least16_t; 
# 55
typedef __uint16_t __uint_least16_t; 
# 56
typedef __int32_t __int_least32_t; 
# 57
typedef __uint32_t __uint_least32_t; 
# 58
typedef __int64_t __int_least64_t; 
# 59
typedef __uint64_t __uint_least64_t; 
# 63
typedef long __quad_t; 
# 64
typedef unsigned long __u_quad_t; 
# 72
typedef long __intmax_t; 
# 73
typedef unsigned long __uintmax_t; 
# 145 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
typedef unsigned long __dev_t; 
# 146
typedef unsigned __uid_t; 
# 147
typedef unsigned __gid_t; 
# 148
typedef unsigned long __ino_t; 
# 149
typedef unsigned long __ino64_t; 
# 150
typedef unsigned __mode_t; 
# 151
typedef unsigned long __nlink_t; 
# 152
typedef long __off_t; 
# 153
typedef long __off64_t; 
# 154
typedef int __pid_t; 
# 155
typedef struct { int __val[2]; } __fsid_t; 
# 156
typedef long __clock_t; 
# 157
typedef unsigned long __rlim_t; 
# 158
typedef unsigned long __rlim64_t; 
# 159
typedef unsigned __id_t; 
# 160
typedef long __time_t; 
# 161
typedef unsigned __useconds_t; 
# 162
typedef long __suseconds_t; 
# 163
typedef long __suseconds64_t; 
# 165
typedef int __daddr_t; 
# 166
typedef int __key_t; 
# 169
typedef int __clockid_t; 
# 172
typedef void *__timer_t; 
# 175
typedef long __blksize_t; 
# 180
typedef long __blkcnt_t; 
# 181
typedef long __blkcnt64_t; 
# 184
typedef unsigned long __fsblkcnt_t; 
# 185
typedef unsigned long __fsblkcnt64_t; 
# 188
typedef unsigned long __fsfilcnt_t; 
# 189
typedef unsigned long __fsfilcnt64_t; 
# 192
typedef long __fsword_t; 
# 194
typedef long __ssize_t; 
# 197
typedef long __syscall_slong_t; 
# 199
typedef unsigned long __syscall_ulong_t; 
# 203
typedef __off64_t __loff_t; 
# 204
typedef char *__caddr_t; 
# 207
typedef long __intptr_t; 
# 210
typedef unsigned __socklen_t; 
# 215
typedef int __sig_atomic_t; 
# 33 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __u_char u_char; 
# 34
typedef __u_short u_short; 
# 35
typedef __u_int u_int; 
# 36
typedef __u_long u_long; 
# 37
typedef __quad_t quad_t; 
# 38
typedef __u_quad_t u_quad_t; 
# 39
typedef __fsid_t fsid_t; 
# 42
typedef __loff_t loff_t; 
# 47
typedef __ino_t ino_t; 
# 54
typedef __ino64_t ino64_t; 
# 59
typedef __dev_t dev_t; 
# 64
typedef __gid_t gid_t; 
# 69
typedef __mode_t mode_t; 
# 74
typedef __nlink_t nlink_t; 
# 79
typedef __uid_t uid_t; 
# 85
typedef __off_t off_t; 
# 92
typedef __off64_t off64_t; 
# 97
typedef __pid_t pid_t; 
# 103
typedef __id_t id_t; 
# 108
typedef __ssize_t ssize_t; 
# 114
typedef __daddr_t daddr_t; 
# 115
typedef __caddr_t caddr_t; 
# 121
typedef __key_t key_t; 
# 7 "/usr/include/x86_64-linux-gnu/bits/types/clock_t.h" 3
typedef __clock_t clock_t; 
# 7 "/usr/include/x86_64-linux-gnu/bits/types/clockid_t.h" 3
typedef __clockid_t clockid_t; 
# 10 "/usr/include/x86_64-linux-gnu/bits/types/time_t.h" 3
typedef __time_t time_t; 
# 7 "/usr/include/x86_64-linux-gnu/bits/types/timer_t.h" 3
typedef __timer_t timer_t; 
# 134 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __useconds_t useconds_t; 
# 138
typedef __suseconds_t suseconds_t; 
# 148 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef unsigned long ulong; 
# 149
typedef unsigned short ushort; 
# 150
typedef unsigned uint; 
# 24 "/usr/include/x86_64-linux-gnu/bits/stdint-intn.h" 3
typedef __int8_t int8_t; 
# 25
typedef __int16_t int16_t; 
# 26
typedef __int32_t int32_t; 
# 27
typedef __int64_t int64_t; 
# 158 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __uint8_t u_int8_t; 
# 159
typedef __uint16_t u_int16_t; 
# 160
typedef __uint32_t u_int32_t; 
# 161
typedef __uint64_t u_int64_t; 
# 164
typedef long register_t __attribute((__mode__(__word__))); 
# 34 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 3
static inline __uint16_t __bswap_16(__uint16_t __bsx) 
# 35
{ 
# 37
return __builtin_bswap16(__bsx); 
# 41
} 
# 49
static inline __uint32_t __bswap_32(__uint32_t __bsx) 
# 50
{ 
# 52
return __builtin_bswap32(__bsx); 
# 56
} 
# 70 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 3
__extension__ static inline __uint64_t __bswap_64(__uint64_t __bsx) 
# 71
{ 
# 73
return __builtin_bswap64(__bsx); 
# 77
} 
# 33 "/usr/include/x86_64-linux-gnu/bits/uintn-identity.h" 3
static inline __uint16_t __uint16_identity(__uint16_t __x) 
# 34
{ 
# 35
return __x; 
# 36
} 
# 39
static inline __uint32_t __uint32_identity(__uint32_t __x) 
# 40
{ 
# 41
return __x; 
# 42
} 
# 45
static inline __uint64_t __uint64_identity(__uint64_t __x) 
# 46
{ 
# 47
return __x; 
# 48
} 
# 8 "/usr/include/x86_64-linux-gnu/bits/types/__sigset_t.h" 3
typedef 
# 6
struct { 
# 7
unsigned long __val[(1024) / ((8) * sizeof(unsigned long))]; 
# 8
} __sigset_t; 
# 7 "/usr/include/x86_64-linux-gnu/bits/types/sigset_t.h" 3
typedef __sigset_t sigset_t; 
# 8 "/usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h" 3
struct timeval { 
# 14
__time_t tv_sec; 
# 15
__suseconds_t tv_usec; 
# 17
}; 
# 11 "/usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h" 3
struct timespec { 
# 16
__time_t tv_sec; 
# 21
__syscall_slong_t tv_nsec; 
# 31 "/usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h" 3
}; 
# 49 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
typedef long __fd_mask; 
# 70 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
typedef 
# 60
struct { 
# 64
__fd_mask fds_bits[1024 / (8 * ((int)sizeof(__fd_mask)))]; 
# 70
} fd_set; 
# 77
typedef __fd_mask fd_mask; 
# 91 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern "C" {
# 102 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern int select(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, timeval *__restrict__ __timeout); 
# 127 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern int pselect(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, const timespec *__restrict__ __timeout, const __sigset_t *__restrict__ __sigmask); 
# 153 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
}
# 185 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __blksize_t blksize_t; 
# 192
typedef __blkcnt_t blkcnt_t; 
# 196
typedef __fsblkcnt_t fsblkcnt_t; 
# 200
typedef __fsfilcnt_t fsfilcnt_t; 
# 219 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __blkcnt64_t blkcnt64_t; 
# 220
typedef __fsblkcnt64_t fsblkcnt64_t; 
# 221
typedef __fsfilcnt64_t fsfilcnt64_t; 
# 33 "/usr/include/x86_64-linux-gnu/bits/atomic_wide_counter.h" 3
typedef 
# 26
union { 
# 27
__extension__ unsigned long long __value64; 
# 29
struct { 
# 30
unsigned __low; 
# 31
unsigned __high; 
# 32
} __value32; 
# 33
} __atomic_wide_counter; 
# 55 "/usr/include/x86_64-linux-gnu/bits/thread-shared-types.h" 3
typedef 
# 51
struct __pthread_internal_list { 
# 53
__pthread_internal_list *__prev; 
# 54
__pthread_internal_list *__next; 
# 55
} __pthread_list_t; 
# 60
typedef 
# 57
struct __pthread_internal_slist { 
# 59
__pthread_internal_slist *__next; 
# 60
} __pthread_slist_t; 
# 22 "/usr/include/x86_64-linux-gnu/bits/struct_mutex.h" 3
struct __pthread_mutex_s { 
# 24
int __lock; 
# 25
unsigned __count; 
# 26
int __owner; 
# 28
unsigned __nusers; 
# 32
int __kind; 
# 34
short __spins; 
# 35
short __elision; 
# 36
__pthread_list_t __list; 
# 53 "/usr/include/x86_64-linux-gnu/bits/struct_mutex.h" 3
}; 
# 23 "/usr/include/x86_64-linux-gnu/bits/struct_rwlock.h" 3
struct __pthread_rwlock_arch_t { 
# 25
unsigned __readers; 
# 26
unsigned __writers; 
# 27
unsigned __wrphase_futex; 
# 28
unsigned __writers_futex; 
# 29
unsigned __pad3; 
# 30
unsigned __pad4; 
# 32
int __cur_writer; 
# 33
int __shared; 
# 34
signed char __rwelision; 
# 39
unsigned char __pad1[7]; 
# 42
unsigned long __pad2; 
# 45
unsigned __flags; 
# 55 "/usr/include/x86_64-linux-gnu/bits/struct_rwlock.h" 3
}; 
# 94 "/usr/include/x86_64-linux-gnu/bits/thread-shared-types.h" 3
struct __pthread_cond_s { 
# 96
__atomic_wide_counter __wseq; 
# 97
__atomic_wide_counter __g1_start; 
# 98
unsigned __g_refs[2]; 
# 99
unsigned __g_size[2]; 
# 100
unsigned __g1_orig_size; 
# 101
unsigned __wrefs; 
# 102
unsigned __g_signals[2]; 
# 103
}; 
# 105
typedef unsigned __tss_t; 
# 106
typedef unsigned long __thrd_t; 
# 111
typedef 
# 109
struct { 
# 110
int __data; 
# 111
} __once_flag; 
# 27 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
typedef unsigned long pthread_t; 
# 36
typedef 
# 33
union { 
# 34
char __size[4]; 
# 35
int __align; 
# 36
} pthread_mutexattr_t; 
# 45
typedef 
# 42
union { 
# 43
char __size[4]; 
# 44
int __align; 
# 45
} pthread_condattr_t; 
# 49
typedef unsigned pthread_key_t; 
# 53
typedef int pthread_once_t; 
# 56
union pthread_attr_t { 
# 58
char __size[56]; 
# 59
long __align; 
# 60
}; 
# 62
typedef pthread_attr_t pthread_attr_t; 
# 72
typedef 
# 68
union { 
# 69
__pthread_mutex_s __data; 
# 70
char __size[40]; 
# 71
long __align; 
# 72
} pthread_mutex_t; 
# 80
typedef 
# 76
union { 
# 77
__pthread_cond_s __data; 
# 78
char __size[48]; 
# 79
__extension__ long long __align; 
# 80
} pthread_cond_t; 
# 91
typedef 
# 87
union { 
# 88
__pthread_rwlock_arch_t __data; 
# 89
char __size[56]; 
# 90
long __align; 
# 91
} pthread_rwlock_t; 
# 97
typedef 
# 94
union { 
# 95
char __size[8]; 
# 96
long __align; 
# 97
} pthread_rwlockattr_t; 
# 103
typedef volatile int pthread_spinlock_t; 
# 112
typedef 
# 109
union { 
# 110
char __size[32]; 
# 111
long __align; 
# 112
} pthread_barrier_t; 
# 118
typedef 
# 115
union { 
# 116
char __size[4]; 
# 117
int __align; 
# 118
} pthread_barrierattr_t; 
# 230 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
}
# 521 "/usr/include/stdlib.h" 3
extern long random() noexcept(true); 
# 524
extern void srandom(unsigned __seed) noexcept(true); 
# 530
extern char *initstate(unsigned __seed, char * __statebuf, size_t __statelen) noexcept(true)
# 531
 __attribute((__nonnull__(2))); 
# 535
extern char *setstate(char * __statebuf) noexcept(true) __attribute((__nonnull__(1))); 
# 543
struct random_data { 
# 545
int32_t *fptr; 
# 546
int32_t *rptr; 
# 547
int32_t *state; 
# 548
int rand_type; 
# 549
int rand_deg; 
# 550
int rand_sep; 
# 551
int32_t *end_ptr; 
# 552
}; 
# 554
extern int random_r(random_data *__restrict__ __buf, int32_t *__restrict__ __result) noexcept(true)
# 555
 __attribute((__nonnull__(1, 2))); 
# 557
extern int srandom_r(unsigned __seed, random_data * __buf) noexcept(true)
# 558
 __attribute((__nonnull__(2))); 
# 560
extern int initstate_r(unsigned __seed, char *__restrict__ __statebuf, size_t __statelen, random_data *__restrict__ __buf) noexcept(true)
# 563
 __attribute((__nonnull__(2, 4))); 
# 565
extern int setstate_r(char *__restrict__ __statebuf, random_data *__restrict__ __buf) noexcept(true)
# 567
 __attribute((__nonnull__(1, 2))); 
# 573
extern int rand() noexcept(true); 
# 575
extern void srand(unsigned __seed) noexcept(true); 
# 579
extern int rand_r(unsigned * __seed) noexcept(true); 
# 587
extern double drand48() noexcept(true); 
# 588
extern double erand48(unsigned short  __xsubi[3]) noexcept(true) __attribute((__nonnull__(1))); 
# 591
extern long lrand48() noexcept(true); 
# 592
extern long nrand48(unsigned short  __xsubi[3]) noexcept(true)
# 593
 __attribute((__nonnull__(1))); 
# 596
extern long mrand48() noexcept(true); 
# 597
extern long jrand48(unsigned short  __xsubi[3]) noexcept(true)
# 598
 __attribute((__nonnull__(1))); 
# 601
extern void srand48(long __seedval) noexcept(true); 
# 602
extern unsigned short *seed48(unsigned short  __seed16v[3]) noexcept(true)
# 603
 __attribute((__nonnull__(1))); 
# 604
extern void lcong48(unsigned short  __param[7]) noexcept(true) __attribute((__nonnull__(1))); 
# 610
struct drand48_data { 
# 612
unsigned short __x[3]; 
# 613
unsigned short __old_x[3]; 
# 614
unsigned short __c; 
# 615
unsigned short __init; 
# 616
__extension__ unsigned long long __a; 
# 618
}; 
# 621
extern int drand48_r(drand48_data *__restrict__ __buffer, double *__restrict__ __result) noexcept(true)
# 622
 __attribute((__nonnull__(1, 2))); 
# 623
extern int erand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, double *__restrict__ __result) noexcept(true)
# 625
 __attribute((__nonnull__(1, 2))); 
# 628
extern int lrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) noexcept(true)
# 630
 __attribute((__nonnull__(1, 2))); 
# 631
extern int nrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) noexcept(true)
# 634
 __attribute((__nonnull__(1, 2))); 
# 637
extern int mrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) noexcept(true)
# 639
 __attribute((__nonnull__(1, 2))); 
# 640
extern int jrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) noexcept(true)
# 643
 __attribute((__nonnull__(1, 2))); 
# 646
extern int srand48_r(long __seedval, drand48_data * __buffer) noexcept(true)
# 647
 __attribute((__nonnull__(2))); 
# 649
extern int seed48_r(unsigned short  __seed16v[3], drand48_data * __buffer) noexcept(true)
# 650
 __attribute((__nonnull__(1, 2))); 
# 652
extern int lcong48_r(unsigned short  __param[7], drand48_data * __buffer) noexcept(true)
# 654
 __attribute((__nonnull__(1, 2))); 
# 657
extern __uint32_t arc4random() noexcept(true); 
# 661
extern void arc4random_buf(void * __buf, size_t __size) noexcept(true)
# 662
 __attribute((__nonnull__(1))); 
# 666
extern __uint32_t arc4random_uniform(__uint32_t __upper_bound) noexcept(true); 
# 672
extern void *malloc(size_t __size) noexcept(true) __attribute((__malloc__))
# 673
 __attribute((__alloc_size__(1))); 
# 675
extern void *calloc(size_t __nmemb, size_t __size) noexcept(true)
# 676
 __attribute((__malloc__)) __attribute((__alloc_size__(1, 2))); 
# 683
extern void *realloc(void * __ptr, size_t __size) noexcept(true)
# 684
 __attribute((__warn_unused_result__)) __attribute((__alloc_size__(2))); 
# 687
extern void free(void * __ptr) noexcept(true); 
# 695
extern void *reallocarray(void * __ptr, size_t __nmemb, size_t __size) noexcept(true)
# 696
 __attribute((__warn_unused_result__))
# 697
 __attribute((__alloc_size__(2, 3)))
# 698
 __attribute((__malloc__(__builtin_free, 1))); 
# 701
extern void *reallocarray(void * __ptr, size_t __nmemb, size_t __size) noexcept(true)
# 702
 __attribute((__malloc__(reallocarray, 1))); 
# 26 "/usr/include/alloca.h" 3
extern "C" {
# 32
extern void *alloca(size_t __size) noexcept(true); 
# 38
}
# 712 "/usr/include/stdlib.h" 3
extern void *valloc(size_t __size) noexcept(true) __attribute((__malloc__))
# 713
 __attribute((__alloc_size__(1))); 
# 718
extern int posix_memalign(void ** __memptr, size_t __alignment, size_t __size) noexcept(true)
# 719
 __attribute((__nonnull__(1))); 
# 724
extern void *aligned_alloc(size_t __alignment, size_t __size) noexcept(true)
# 725
 __attribute((__malloc__)) __attribute((__alloc_align__(1 )))
# 726
 __attribute((__alloc_size__(2))); 
# 730
extern void abort() noexcept(true) __attribute((__noreturn__)); 
# 734
extern int atexit(void (* __func)(void)) noexcept(true) __attribute((__nonnull__(1))); 
# 739
extern "C++" int at_quick_exit(void (* __func)(void)) noexcept(true) __asm__("at_quick_exit")
# 740
 __attribute((__nonnull__(1))); 
# 749 "/usr/include/stdlib.h" 3
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg) noexcept(true)
# 750
 __attribute((__nonnull__(1))); 
# 756
extern void exit(int __status) noexcept(true) __attribute((__noreturn__)); 
# 762
extern void quick_exit(int __status) noexcept(true) __attribute((__noreturn__)); 
# 768
extern void _Exit(int __status) noexcept(true) __attribute((__noreturn__)); 
# 773
extern char *getenv(const char * __name) noexcept(true) __attribute((__nonnull__(1))); 
# 778
extern char *secure_getenv(const char * __name) noexcept(true)
# 779
 __attribute((__nonnull__(1))); 
# 786
extern int putenv(char * __string) noexcept(true) __attribute((__nonnull__(1))); 
# 792
extern int setenv(const char * __name, const char * __value, int __replace) noexcept(true)
# 793
 __attribute((__nonnull__(2))); 
# 796
extern int unsetenv(const char * __name) noexcept(true) __attribute((__nonnull__(1))); 
# 803
extern int clearenv() noexcept(true); 
# 814 "/usr/include/stdlib.h" 3
extern char *mktemp(char * __template) noexcept(true) __attribute((__nonnull__(1))); 
# 827 "/usr/include/stdlib.h" 3
extern int mkstemp(char * __template) __attribute((__nonnull__(1))); 
# 837 "/usr/include/stdlib.h" 3
extern int mkstemp64(char * __template) __attribute((__nonnull__(1))); 
# 849 "/usr/include/stdlib.h" 3
extern int mkstemps(char * __template, int __suffixlen) __attribute((__nonnull__(1))); 
# 859 "/usr/include/stdlib.h" 3
extern int mkstemps64(char * __template, int __suffixlen)
# 860
 __attribute((__nonnull__(1))); 
# 870 "/usr/include/stdlib.h" 3
extern char *mkdtemp(char * __template) noexcept(true) __attribute((__nonnull__(1))); 
# 881 "/usr/include/stdlib.h" 3
extern int mkostemp(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 891 "/usr/include/stdlib.h" 3
extern int mkostemp64(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 901 "/usr/include/stdlib.h" 3
extern int mkostemps(char * __template, int __suffixlen, int __flags)
# 902
 __attribute((__nonnull__(1))); 
# 913 "/usr/include/stdlib.h" 3
extern int mkostemps64(char * __template, int __suffixlen, int __flags)
# 914
 __attribute((__nonnull__(1))); 
# 923 "/usr/include/stdlib.h" 3
extern int system(const char * __command); 
# 929
extern char *canonicalize_file_name(const char * __name) noexcept(true)
# 930
 __attribute((__nonnull__(1))) __attribute((__malloc__))
# 931
 __attribute((__malloc__(__builtin_free, 1))); 
# 940 "/usr/include/stdlib.h" 3
extern char *realpath(const char *__restrict__ __name, char *__restrict__ __resolved) noexcept(true); 
# 948
typedef int (*__compar_fn_t)(const void *, const void *); 
# 951
typedef __compar_fn_t comparison_fn_t; 
# 955
typedef int (*__compar_d_fn_t)(const void *, const void *, void *); 
# 960
extern void *bsearch(const void * __key, const void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 962
 __attribute((__nonnull__(1, 2, 5))); 
# 970
extern void qsort(void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 971
 __attribute((__nonnull__(1, 4))); 
# 973
extern void qsort_r(void * __base, size_t __nmemb, size_t __size, __compar_d_fn_t __compar, void * __arg)
# 975
 __attribute((__nonnull__(1, 4))); 
# 980
extern int abs(int __x) noexcept(true) __attribute((const)); 
# 981
extern long labs(long __x) noexcept(true) __attribute((const)); 
# 984
__extension__ extern long long llabs(long long __x) noexcept(true)
# 985
 __attribute((const)); 
# 992
extern div_t div(int __numer, int __denom) noexcept(true)
# 993
 __attribute((const)); 
# 994
extern ldiv_t ldiv(long __numer, long __denom) noexcept(true)
# 995
 __attribute((const)); 
# 998
__extension__ extern lldiv_t lldiv(long long __numer, long long __denom) noexcept(true)
# 1000
 __attribute((const)); 
# 1012 "/usr/include/stdlib.h" 3
extern char *ecvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) noexcept(true)
# 1013
 __attribute((__nonnull__(3, 4))); 
# 1018
extern char *fcvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) noexcept(true)
# 1019
 __attribute((__nonnull__(3, 4))); 
# 1024
extern char *gcvt(double __value, int __ndigit, char * __buf) noexcept(true)
# 1025
 __attribute((__nonnull__(3))); 
# 1030
extern char *qecvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) noexcept(true)
# 1032
 __attribute((__nonnull__(3, 4))); 
# 1033
extern char *qfcvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) noexcept(true)
# 1035
 __attribute((__nonnull__(3, 4))); 
# 1036
extern char *qgcvt(long double __value, int __ndigit, char * __buf) noexcept(true)
# 1037
 __attribute((__nonnull__(3))); 
# 1042
extern int ecvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) noexcept(true)
# 1044
 __attribute((__nonnull__(3, 4, 5))); 
# 1045
extern int fcvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) noexcept(true)
# 1047
 __attribute((__nonnull__(3, 4, 5))); 
# 1049
extern int qecvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) noexcept(true)
# 1052
 __attribute((__nonnull__(3, 4, 5))); 
# 1053
extern int qfcvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) noexcept(true)
# 1056
 __attribute((__nonnull__(3, 4, 5))); 
# 1062
extern int mblen(const char * __s, size_t __n) noexcept(true); 
# 1065
extern int mbtowc(wchar_t *__restrict__ __pwc, const char *__restrict__ __s, size_t __n) noexcept(true); 
# 1069
extern int wctomb(char * __s, wchar_t __wchar) noexcept(true); 
# 1073
extern size_t mbstowcs(wchar_t *__restrict__ __pwcs, const char *__restrict__ __s, size_t __n) noexcept(true)
# 1075
 __attribute((__access__(__read_only__ , 2 ))); 
# 1077
extern size_t wcstombs(char *__restrict__ __s, const wchar_t *__restrict__ __pwcs, size_t __n) noexcept(true)
# 1080
 __attribute((__access__(__write_only__ , 1 , 3 )))
# 1081
 __attribute((__access__(__read_only__ , 2 ))); 
# 1088
extern int rpmatch(const char * __response) noexcept(true) __attribute((__nonnull__(1))); 
# 1099 "/usr/include/stdlib.h" 3
extern int getsubopt(char **__restrict__ __optionp, char *const *__restrict__ __tokens, char **__restrict__ __valuep) noexcept(true)
# 1102
 __attribute((__nonnull__(1, 2, 3))); 
# 1110
extern int posix_openpt(int __oflag); 
# 1118
extern int grantpt(int __fd) noexcept(true); 
# 1122
extern int unlockpt(int __fd) noexcept(true); 
# 1127
extern char *ptsname(int __fd) noexcept(true); 
# 1134
extern int ptsname_r(int __fd, char * __buf, size_t __buflen) noexcept(true)
# 1135
 __attribute((__nonnull__(2))) __attribute((__access__(__write_only__ , 2 , 3 ))); 
# 1138
extern int getpt(); 
# 1145
extern int getloadavg(double  __loadavg[], int __nelem) noexcept(true)
# 1146
 __attribute((__nonnull__(1))); 
# 1167 "/usr/include/stdlib.h" 3
}
# 46 "/usr/include/c++/12/bits/std_abs.h" 3
extern "C++" {
# 48
namespace std __attribute((__visibility__("default"))) { 
# 52
using ::abs;
# 56
inline long abs(long __i) { return __builtin_labs(__i); } 
# 61
inline long long abs(long long __x) { return __builtin_llabs(__x); } 
# 71 "/usr/include/c++/12/bits/std_abs.h" 3
constexpr double abs(double __x) 
# 72
{ return __builtin_fabs(__x); } 
# 75
constexpr float abs(float __x) 
# 76
{ return __builtin_fabsf(__x); } 
# 79
constexpr long double abs(long double __x) 
# 80
{ return __builtin_fabsl(__x); } 
# 85
__extension__ constexpr __int128 abs(__int128 __x) { return (__x >= (0)) ? __x : (-__x); } 
# 117 "/usr/include/c++/12/bits/std_abs.h" 3
}
# 118
}
# 121 "/usr/include/c++/12/cstdlib" 3
extern "C++" {
# 123
namespace std __attribute((__visibility__("default"))) { 
# 127
using ::div_t;
# 128
using ::ldiv_t;
# 130
using ::abort;
# 132
using ::aligned_alloc;
# 134
using ::atexit;
# 137
using ::at_quick_exit;
# 140
using ::atof;
# 141
using ::atoi;
# 142
using ::atol;
# 143
using ::bsearch;
# 144
using ::calloc;
# 145
using ::div;
# 146
using ::exit;
# 147
using ::free;
# 148
using ::getenv;
# 149
using ::labs;
# 150
using ::ldiv;
# 151
using ::malloc;
# 153
using ::mblen;
# 154
using ::mbstowcs;
# 155
using ::mbtowc;
# 157
using ::qsort;
# 160
using ::quick_exit;
# 163
using ::rand;
# 164
using ::realloc;
# 165
using ::srand;
# 166
using ::strtod;
# 167
using ::strtol;
# 168
using ::strtoul;
# 169
using ::system;
# 171
using ::wcstombs;
# 172
using ::wctomb;
# 177
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); } 
# 182
}
# 195 "/usr/include/c++/12/cstdlib" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 200
using ::lldiv_t;
# 206
using ::_Exit;
# 210
using ::llabs;
# 213
inline lldiv_t div(long long __n, long long __d) 
# 214
{ lldiv_t __q; (__q.quot) = (__n / __d); (__q.rem) = (__n % __d); return __q; } 
# 216
using ::lldiv;
# 227 "/usr/include/c++/12/cstdlib" 3
using ::atoll;
# 228
using ::strtoll;
# 229
using ::strtoull;
# 231
using ::strtof;
# 232
using ::strtold;
# 235
}
# 237
namespace std { 
# 240
using __gnu_cxx::lldiv_t;
# 242
using __gnu_cxx::_Exit;
# 244
using __gnu_cxx::llabs;
# 245
using __gnu_cxx::div;
# 246
using __gnu_cxx::lldiv;
# 248
using __gnu_cxx::atoll;
# 249
using __gnu_cxx::strtof;
# 250
using __gnu_cxx::strtoll;
# 251
using __gnu_cxx::strtoull;
# 252
using __gnu_cxx::strtold;
# 253
}
# 257
}
# 38 "/usr/include/c++/12/stdlib.h" 3
using std::abort;
# 39
using std::atexit;
# 40
using std::exit;
# 43
using std::at_quick_exit;
# 46
using std::quick_exit;
# 54
using std::abs;
# 55
using std::atof;
# 56
using std::atoi;
# 57
using std::atol;
# 58
using std::bsearch;
# 59
using std::calloc;
# 60
using std::div;
# 61
using std::free;
# 62
using std::getenv;
# 63
using std::labs;
# 64
using std::ldiv;
# 65
using std::malloc;
# 67
using std::mblen;
# 68
using std::mbstowcs;
# 69
using std::mbtowc;
# 71
using std::qsort;
# 72
using std::rand;
# 73
using std::realloc;
# 74
using std::srand;
# 75
using std::strtod;
# 76
using std::strtol;
# 77
using std::strtoul;
# 78
using std::system;
# 80
using std::wcstombs;
# 81
using std::wctomb;
# 179 "/usr/include/cuda_device_runtime_api.h" 3
extern "C" {
# 186
__attribute__((unused)) extern cudaError_t __cudaDeviceSynchronizeDeprecationAvoidance(); 
# 235 "/usr/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) extern cudaError_t __cudaCDP2DeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 236
__attribute__((unused)) extern cudaError_t __cudaCDP2DeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 237
__attribute__((unused)) extern cudaError_t __cudaCDP2DeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 238
__attribute__((unused)) extern cudaError_t __cudaCDP2DeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 239
__attribute__((unused)) extern cudaError_t __cudaCDP2GetLastError(); 
# 240
__attribute__((unused)) extern cudaError_t __cudaCDP2PeekAtLastError(); 
# 241
__attribute__((unused)) extern const char *__cudaCDP2GetErrorString(cudaError_t error); 
# 242
__attribute__((unused)) extern const char *__cudaCDP2GetErrorName(cudaError_t error); 
# 243
__attribute__((unused)) extern cudaError_t __cudaCDP2GetDeviceCount(int * count); 
# 244
__attribute__((unused)) extern cudaError_t __cudaCDP2GetDevice(int * device); 
# 245
__attribute__((unused)) extern cudaError_t __cudaCDP2StreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 246
__attribute__((unused)) extern cudaError_t __cudaCDP2StreamDestroy(cudaStream_t stream); 
# 247
__attribute__((unused)) extern cudaError_t __cudaCDP2StreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 248
__attribute__((unused)) extern cudaError_t __cudaCDP2StreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 249
__attribute__((unused)) extern cudaError_t __cudaCDP2EventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 250
__attribute__((unused)) extern cudaError_t __cudaCDP2EventRecord(cudaEvent_t event, cudaStream_t stream); 
# 251
__attribute__((unused)) extern cudaError_t __cudaCDP2EventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
# 252
__attribute__((unused)) extern cudaError_t __cudaCDP2EventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
# 253
__attribute__((unused)) extern cudaError_t __cudaCDP2EventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
# 254
__attribute__((unused)) extern cudaError_t __cudaCDP2EventDestroy(cudaEvent_t event); 
# 255
__attribute__((unused)) extern cudaError_t __cudaCDP2FuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 256
__attribute__((unused)) extern cudaError_t __cudaCDP2Free(void * devPtr); 
# 257
__attribute__((unused)) extern cudaError_t __cudaCDP2Malloc(void ** devPtr, size_t size); 
# 258
__attribute__((unused)) extern cudaError_t __cudaCDP2MemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 259
__attribute__((unused)) extern cudaError_t __cudaCDP2MemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 260
__attribute__((unused)) extern cudaError_t __cudaCDP2Memcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 261
__attribute__((unused)) extern cudaError_t __cudaCDP2Memcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 262
__attribute__((unused)) extern cudaError_t __cudaCDP2Memcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 263
__attribute__((unused)) extern cudaError_t __cudaCDP2Memcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 264
__attribute__((unused)) extern cudaError_t __cudaCDP2MemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 265
__attribute__((unused)) extern cudaError_t __cudaCDP2MemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 266
__attribute__((unused)) extern cudaError_t __cudaCDP2Memset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 267
__attribute__((unused)) extern cudaError_t __cudaCDP2Memset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 268
__attribute__((unused)) extern cudaError_t __cudaCDP2Memset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 269
__attribute__((unused)) extern cudaError_t __cudaCDP2Memset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 270
__attribute__((unused)) extern cudaError_t __cudaCDP2RuntimeGetVersion(int * runtimeVersion); 
# 271
__attribute__((unused)) extern void *__cudaCDP2GetParameterBuffer(size_t alignment, size_t size); 
# 272
__attribute__((unused)) extern void *__cudaCDP2GetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
# 273
__attribute__((unused)) extern cudaError_t __cudaCDP2LaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 274
__attribute__((unused)) extern cudaError_t __cudaCDP2LaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
# 275
__attribute__((unused)) extern cudaError_t __cudaCDP2LaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 276
__attribute__((unused)) extern cudaError_t __cudaCDP2LaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
# 277
__attribute__((unused)) extern cudaError_t __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
# 278
__attribute__((unused)) extern cudaError_t __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 281
extern cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 300 "/usr/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) static inline cudaGraphExec_t cudaGetCurrentGraphExec() 
# 301
{int volatile ___ = 1;
# 305
::exit(___);}
#if 0
# 301
{ 
# 302
unsigned long long current_graph_exec; 
# 303
__asm__("mov.u64 %0, %%current_graph_exec;" : "=l" (current_graph_exec) :); 
# 304
return (cudaGraphExec_t)current_graph_exec; 
# 305
} 
#endif
# 323 "/usr/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) static inline void cudaTriggerProgrammaticLaunchCompletion() 
# 324
{int volatile ___ = 1;
# 326
::exit(___);}
#if 0
# 324
{ 
# 325
__asm__ volatile("griddepcontrol.launch_dependents;" : :); 
# 326
} 
#endif
# 339 "/usr/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) static inline void cudaGridDependencySynchronize() 
# 340
{int volatile ___ = 1;
# 342
::exit(___);}
#if 0
# 340
{ 
# 341
__asm__ volatile("griddepcontrol.wait;" : : : "memory"); 
# 342
} 
#endif
# 346 "/usr/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) extern unsigned long long cudaCGGetIntrinsicHandle(cudaCGScope scope); 
# 347
__attribute__((unused)) extern cudaError_t cudaCGSynchronize(unsigned long long handle, unsigned flags); 
# 348
__attribute__((unused)) extern cudaError_t cudaCGSynchronizeGrid(unsigned long long handle, unsigned flags); 
# 349
__attribute__((unused)) extern cudaError_t cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned long long handle); 
# 350
__attribute__((unused)) extern cudaError_t cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned long long handle); 
# 572 "/usr/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) static inline void *cudaGetParameterBuffer(size_t alignment, size_t size) 
# 573
{int volatile ___ = 1;(void)alignment;(void)size;
# 575
::exit(___);}
#if 0
# 573
{ 
# 574
return __cudaCDP2GetParameterBuffer(alignment, size); 
# 575
} 
#endif
# 608 "/usr/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) static inline void *cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize) 
# 609
{int volatile ___ = 1;(void)func;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;
# 611
::exit(___);}
#if 0
# 609
{ 
# 610
return __cudaCDP2GetParameterBufferV2(func, gridDimension, blockDimension, sharedMemSize); 
# 611
} 
#endif
# 618 "/usr/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) static inline cudaError_t cudaLaunchDevice_ptsz(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream) 
# 619
{int volatile ___ = 1;(void)func;(void)parameterBuffer;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;(void)stream;
# 621
::exit(___);}
#if 0
# 619
{ 
# 620
return __cudaCDP2LaunchDevice_ptsz(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream); 
# 621
} 
#endif
# 623 "/usr/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) static inline cudaError_t cudaLaunchDeviceV2_ptsz(void *parameterBuffer, cudaStream_t stream) 
# 624
{int volatile ___ = 1;(void)parameterBuffer;(void)stream;
# 626
::exit(___);}
#if 0
# 624
{ 
# 625
return __cudaCDP2LaunchDeviceV2_ptsz(parameterBuffer, stream); 
# 626
} 
#endif
# 658 "/usr/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) static inline cudaError_t cudaLaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream) 
# 659
{int volatile ___ = 1;(void)func;(void)parameterBuffer;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;(void)stream;
# 661
::exit(___);}
#if 0
# 659
{ 
# 660
return __cudaCDP2LaunchDevice(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream); 
# 661
} 
#endif
# 663 "/usr/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) static inline cudaError_t cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream) 
# 664
{int volatile ___ = 1;(void)parameterBuffer;(void)stream;
# 666
::exit(___);}
#if 0
# 664
{ 
# 665
return __cudaCDP2LaunchDeviceV2(parameterBuffer, stream); 
# 666
} 
#endif
# 720 "/usr/include/cuda_device_runtime_api.h" 3
}
# 722
template< class T> static inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
# 723
template< class T> static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
# 724
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
# 725
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 271 "/usr/include/cuda_runtime_api.h" 3
extern "C" {
# 311 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceReset(); 
# 333 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSynchronize(); 
# 419 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value); 
# 455 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 478 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const cudaChannelFormatDesc * fmtDesc, int device); 
# 512 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 549 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
# 593 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
# 624 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 668 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
# 695 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
# 725 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
# 775 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
# 818 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
# 862 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
# 928 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
# 966 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaIpcCloseMemHandle(void * devPtr); 
# 998 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope); 
# 1041 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadExit(); 
# 1067 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadSynchronize(); 
# 1116 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value); 
# 1149 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
# 1185 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 1232 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
# 1297 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetLastError(); 
# 1348 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaPeekAtLastError(); 
# 1364 "/usr/include/cuda_runtime_api.h" 3
extern const char *cudaGetErrorName(cudaError_t error); 
# 1380 "/usr/include/cuda_runtime_api.h" 3
extern const char *cudaGetErrorString(cudaError_t error); 
# 1409 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetDeviceCount(int * count); 
# 1714 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp * prop, int device); 
# 1916 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 1934 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int device); 
# 1958 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool); 
# 1978 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetMemPool(cudaMemPool_t * memPool, int device); 
# 2026 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int device, int flags); 
# 2066 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
# 2088 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
# 2117 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaInitDevice(int device, unsigned deviceFlags, unsigned flags); 
# 2163 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaSetDevice(int device); 
# 2185 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetDevice(int * device); 
# 2216 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaSetValidDevices(int * device_arr, int len); 
# 2282 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaSetDeviceFlags(unsigned flags); 
# 2327 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetDeviceFlags(unsigned * flags); 
# 2367 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamCreate(cudaStream_t * pStream); 
# 2399 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 2445 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
# 2472 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
# 2497 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
# 2532 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long * streamId); 
# 2547 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaCtxResetPersistingL2Cache(); 
# 2567 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src); 
# 2588 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
# 2612 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
# 2646 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 2677 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags = 0); 
# 2685
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
# 2752 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
# 2776 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream); 
# 2801 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamQuery(cudaStream_t stream); 
# 2885 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
# 2924 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode); 
# 2975 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode * mode); 
# 3003 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * pGraph); 
# 3041 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus); 
# 3089 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned long long * id_out = 0, cudaGraph_t * graph_out = 0, const cudaGraphNode_t ** dependencies_out = 0, size_t * numDependencies_out = 0); 
# 3121 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t * dependencies, size_t numDependencies, unsigned flags = 0); 
# 3158 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventCreate(cudaEvent_t * event); 
# 3195 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 3235 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
# 3282 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned flags = 0); 
# 3314 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventQuery(cudaEvent_t event); 
# 3344 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventSynchronize(cudaEvent_t event); 
# 3373 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 3417 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
# 3598 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const cudaExternalMemoryHandleDesc * memHandleDesc); 
# 3653 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc * bufferDesc); 
# 3713 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc * mipmapDesc); 
# 3737 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem); 
# 3891 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const cudaExternalSemaphoreHandleDesc * semHandleDesc); 
# 3974 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreSignalParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 4050 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreWaitParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 4073 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem); 
# 4140 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 4202 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t * config, const void * func, void ** args); 
# 4259 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 4360 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
# 4405 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
# 4460 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
# 4493 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 4530 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
# 4554 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForDevice(double * d); 
# 4578 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForHost(double * d); 
# 4644 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void * userData); 
# 4701 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
# 4730 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int numBlocks, int blockSize); 
# 4775 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
# 4810 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaOccupancyMaxPotentialClusterSize(int * clusterSize, const void * func, const cudaLaunchConfig_t * launchConfig); 
# 4849 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaOccupancyMaxActiveClusters(int * numClusters, const void * func, const cudaLaunchConfig_t * launchConfig); 
# 4969 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
# 5002 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 5035 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocHost(void ** ptr, size_t size); 
# 5078 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
# 5130 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
# 5168 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFree(void * devPtr); 
# 5191 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFreeHost(void * ptr); 
# 5214 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFreeArray(cudaArray_t array); 
# 5237 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
# 5303 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
# 5396 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned flags); 
# 5419 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaHostUnregister(void * ptr); 
# 5464 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
# 5486 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaHostGetFlags(unsigned * pFlags, void * pHost); 
# 5525 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
# 5670 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
# 5815 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
# 5848 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
# 5953 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms * p); 
# 5985 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
# 6103 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
# 6130 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
# 6164 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemGetInfo(size_t * free, size_t * total); 
# 6190 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
# 6219 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaArrayGetPlane(cudaArray_t * pPlaneArray, cudaArray_t hArray, unsigned planeIdx); 
# 6242 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaArray_t array, int device); 
# 6266 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaMipmappedArray_t mipmap, int device); 
# 6294 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaArray_t array); 
# 6324 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t mipmap); 
# 6369 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 6404 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
# 6453 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 6503 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 6553 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
# 6600 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 6643 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
# 6686 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
# 6743 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6778 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
# 6841 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6899 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6956 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7007 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7058 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7087 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemset(void * devPtr, int value, size_t count); 
# 7121 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
# 7167 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
# 7203 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
# 7244 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
# 7297 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
# 7325 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
# 7352 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol); 
# 7422 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
# 7538 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
# 7597 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
# 7636 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
# 7696 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
# 7738 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
# 7781 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 7832 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7882 "/usr/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7951 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocAsync(void ** devPtr, size_t size, cudaStream_t hStream); 
# 7977 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFreeAsync(void * devPtr, cudaStream_t hStream); 
# 8002 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep); 
# 8046 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
# 8094 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
# 8109 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc * descList, size_t count); 
# 8122 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags * flags, cudaMemPool_t memPool, cudaMemLocation * location); 
# 8142 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolCreate(cudaMemPool_t * memPool, const cudaMemPoolProps * poolProps); 
# 8164 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool); 
# 8200 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocFromPoolAsync(void ** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream); 
# 8225 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolExportToShareableHandle(void * shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned flags); 
# 8252 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t * memPool, void * shareableHandle, cudaMemAllocationHandleType handleType, unsigned flags); 
# 8275 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData * exportData, void * ptr); 
# 8304 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolImportPointer(void ** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData * exportData); 
# 8457 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
# 8498 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
# 8540 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
# 8562 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceDisablePeerAccess(int peerDevice); 
# 8626 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
# 8661 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
# 8700 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 8735 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 8767 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
# 8805 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
# 8834 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
# 8869 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
# 8899 "/usr/include/cuda_runtime_api.h" 3
extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
# 9123 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
# 9143 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject); 
# 9163 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
# 9183 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
# 9204 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
# 9249 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
# 9269 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
# 9288 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
# 9322 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDriverGetVersion(int * driverVersion); 
# 9351 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 9398 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned flags); 
# 9495 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams); 
# 9528 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams); 
# 9553 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 9573 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst); 
# 9596 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
# 9620 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
# 9670 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams); 
# 9729 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 9798 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 9866 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 9898 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams); 
# 9924 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 9963 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10009 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10055 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 10102 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams); 
# 10125 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams); 
# 10148 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 10189 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams); 
# 10212 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams); 
# 10235 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 10275 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph); 
# 10302 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph); 
# 10339 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies); 
# 10382 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
# 10409 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
# 10436 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
# 10482 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
# 10509 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
# 10536 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
# 10585 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 10618 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams * params_out); 
# 10645 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 10694 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 10727 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams * params_out); 
# 10754 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 10831 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaMemAllocNodeParams * nodeParams); 
# 10858 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams * params_out); 
# 10918 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dptr); 
# 10942 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void * dptr_out); 
# 10970 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGraphMemTrim(int device); 
# 11007 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
# 11041 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
# 11069 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph); 
# 11097 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph); 
# 11128 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * pType); 
# 11159 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes); 
# 11190 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes); 
# 11224 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges); 
# 11255 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies); 
# 11287 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes); 
# 11318 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 11349 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 11379 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node); 
# 11441 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned long long flags = 0); 
# 11512 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned long long flags = 0); 
# 11617 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams * instantiateParams); 
# 11642 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long * flags); 
# 11693 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 11743 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 11798 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 11861 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 11922 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 11976 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 12015 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 12061 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph); 
# 12105 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
# 12149 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
# 12196 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 12243 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 12283 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned isEnabled); 
# 12317 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned * isEnabled); 
# 12402 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo * resultInfo); 
# 12427 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 12458 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 12481 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec); 
# 12502 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphDestroy(cudaGraph_t graph); 
# 12521 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char * path, unsigned flags); 
# 12557 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaUserObjectCreate(cudaUserObject_t * object_out, void * ptr, cudaHostFn_t destroy, unsigned initialRefcount, unsigned flags); 
# 12581 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned count = 1); 
# 12609 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned count = 1); 
# 12637 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1, unsigned flags = 0); 
# 12662 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1); 
# 12740 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult * driverStatus = 0); 
# 12748
extern cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
# 12927 "/usr/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetFuncBySymbol(cudaFunction_t * functionPtr, const void * symbolPtr); 
# 13088 "/usr/include/cuda_runtime_api.h" 3
}
# 117 "/usr/include/channel_descriptor.h" 3
template< class T> inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 118
{ 
# 119
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 120
} 
# 122
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
# 123
{ 
# 124
int e = (((int)sizeof(unsigned short)) * 8); 
# 126
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 127
} 
# 129
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
# 130
{ 
# 131
int e = (((int)sizeof(unsigned short)) * 8); 
# 133
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 134
} 
# 136
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
# 137
{ 
# 138
int e = (((int)sizeof(unsigned short)) * 8); 
# 140
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 141
} 
# 143
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
# 144
{ 
# 145
int e = (((int)sizeof(unsigned short)) * 8); 
# 147
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 148
} 
# 150
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
# 151
{ 
# 152
int e = (((int)sizeof(char)) * 8); 
# 157
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 159
} 
# 161
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
# 162
{ 
# 163
int e = (((int)sizeof(signed char)) * 8); 
# 165
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 166
} 
# 168
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
# 169
{ 
# 170
int e = (((int)sizeof(unsigned char)) * 8); 
# 172
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 173
} 
# 175
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
# 176
{ 
# 177
int e = (((int)sizeof(signed char)) * 8); 
# 179
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 180
} 
# 182
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
# 183
{ 
# 184
int e = (((int)sizeof(unsigned char)) * 8); 
# 186
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 187
} 
# 189
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
# 190
{ 
# 191
int e = (((int)sizeof(signed char)) * 8); 
# 193
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 194
} 
# 196
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
# 197
{ 
# 198
int e = (((int)sizeof(unsigned char)) * 8); 
# 200
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 201
} 
# 203
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
# 204
{ 
# 205
int e = (((int)sizeof(signed char)) * 8); 
# 207
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 208
} 
# 210
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
# 211
{ 
# 212
int e = (((int)sizeof(unsigned char)) * 8); 
# 214
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 215
} 
# 217
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
# 218
{ 
# 219
int e = (((int)sizeof(short)) * 8); 
# 221
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 222
} 
# 224
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
# 225
{ 
# 226
int e = (((int)sizeof(unsigned short)) * 8); 
# 228
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 229
} 
# 231
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
# 232
{ 
# 233
int e = (((int)sizeof(short)) * 8); 
# 235
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 236
} 
# 238
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
# 239
{ 
# 240
int e = (((int)sizeof(unsigned short)) * 8); 
# 242
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 243
} 
# 245
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
# 246
{ 
# 247
int e = (((int)sizeof(short)) * 8); 
# 249
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 250
} 
# 252
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
# 253
{ 
# 254
int e = (((int)sizeof(unsigned short)) * 8); 
# 256
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 257
} 
# 259
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
# 260
{ 
# 261
int e = (((int)sizeof(short)) * 8); 
# 263
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 264
} 
# 266
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
# 267
{ 
# 268
int e = (((int)sizeof(unsigned short)) * 8); 
# 270
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 271
} 
# 273
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
# 274
{ 
# 275
int e = (((int)sizeof(int)) * 8); 
# 277
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 278
} 
# 280
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
# 281
{ 
# 282
int e = (((int)sizeof(unsigned)) * 8); 
# 284
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 285
} 
# 287
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
# 288
{ 
# 289
int e = (((int)sizeof(int)) * 8); 
# 291
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 292
} 
# 294
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
# 295
{ 
# 296
int e = (((int)sizeof(unsigned)) * 8); 
# 298
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 299
} 
# 301
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
# 302
{ 
# 303
int e = (((int)sizeof(int)) * 8); 
# 305
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 306
} 
# 308
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
# 309
{ 
# 310
int e = (((int)sizeof(unsigned)) * 8); 
# 312
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 313
} 
# 315
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
# 316
{ 
# 317
int e = (((int)sizeof(int)) * 8); 
# 319
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 320
} 
# 322
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
# 323
{ 
# 324
int e = (((int)sizeof(unsigned)) * 8); 
# 326
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 327
} 
# 389 "/usr/include/channel_descriptor.h" 3
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
# 390
{ 
# 391
int e = (((int)sizeof(float)) * 8); 
# 393
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 394
} 
# 396
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
# 397
{ 
# 398
int e = (((int)sizeof(float)) * 8); 
# 400
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 401
} 
# 403
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
# 404
{ 
# 405
int e = (((int)sizeof(float)) * 8); 
# 407
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 408
} 
# 410
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
# 411
{ 
# 412
int e = (((int)sizeof(float)) * 8); 
# 414
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 415
} 
# 417
static inline cudaChannelFormatDesc cudaCreateChannelDescNV12() 
# 418
{ 
# 419
int e = (((int)sizeof(char)) * 8); 
# 421
return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindNV12); 
# 422
} 
# 424
template< cudaChannelFormatKind > inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 425
{ 
# 426
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 427
} 
# 430
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X1> () 
# 431
{ 
# 432
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedNormalized8X1); 
# 433
} 
# 435
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X2> () 
# 436
{ 
# 437
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedNormalized8X2); 
# 438
} 
# 440
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X4> () 
# 441
{ 
# 442
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSignedNormalized8X4); 
# 443
} 
# 446
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X1> () 
# 447
{ 
# 448
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized8X1); 
# 449
} 
# 451
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X2> () 
# 452
{ 
# 453
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedNormalized8X2); 
# 454
} 
# 456
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X4> () 
# 457
{ 
# 458
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedNormalized8X4); 
# 459
} 
# 462
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X1> () 
# 463
{ 
# 464
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSignedNormalized16X1); 
# 465
} 
# 467
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X2> () 
# 468
{ 
# 469
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindSignedNormalized16X2); 
# 470
} 
# 472
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X4> () 
# 473
{ 
# 474
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSignedNormalized16X4); 
# 475
} 
# 478
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X1> () 
# 479
{ 
# 480
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized16X1); 
# 481
} 
# 483
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X2> () 
# 484
{ 
# 485
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsignedNormalized16X2); 
# 486
} 
# 488
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X4> () 
# 489
{ 
# 490
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsignedNormalized16X4); 
# 491
} 
# 494
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindNV12> () 
# 495
{ 
# 496
return cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindNV12); 
# 497
} 
# 500
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1> () 
# 501
{ 
# 502
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1); 
# 503
} 
# 506
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1SRGB> () 
# 507
{ 
# 508
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1SRGB); 
# 509
} 
# 512
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2> () 
# 513
{ 
# 514
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2); 
# 515
} 
# 518
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2SRGB> () 
# 519
{ 
# 520
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2SRGB); 
# 521
} 
# 524
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3> () 
# 525
{ 
# 526
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3); 
# 527
} 
# 530
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3SRGB> () 
# 531
{ 
# 532
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3SRGB); 
# 533
} 
# 536
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed4> () 
# 537
{ 
# 538
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed4); 
# 539
} 
# 542
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed4> () 
# 543
{ 
# 544
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedBlockCompressed4); 
# 545
} 
# 548
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed5> () 
# 549
{ 
# 550
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed5); 
# 551
} 
# 554
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed5> () 
# 555
{ 
# 556
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedBlockCompressed5); 
# 557
} 
# 560
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed6H> () 
# 561
{ 
# 562
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindUnsignedBlockCompressed6H); 
# 563
} 
# 566
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed6H> () 
# 567
{ 
# 568
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindSignedBlockCompressed6H); 
# 569
} 
# 572
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7> () 
# 573
{ 
# 574
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7); 
# 575
} 
# 578
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7SRGB> () 
# 579
{ 
# 580
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7SRGB); 
# 581
} 
# 79 "/usr/include/driver_functions.h" 3
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
# 80
{ 
# 81
cudaPitchedPtr s; 
# 83
(s.ptr) = d; 
# 84
(s.pitch) = p; 
# 85
(s.xsize) = xsz; 
# 86
(s.ysize) = ysz; 
# 88
return s; 
# 89
} 
# 106 "/usr/include/driver_functions.h" 3
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
# 107
{ 
# 108
cudaPos p; 
# 110
(p.x) = x; 
# 111
(p.y) = y; 
# 112
(p.z) = z; 
# 114
return p; 
# 115
} 
# 132 "/usr/include/driver_functions.h" 3
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
# 133
{ 
# 134
cudaExtent e; 
# 136
(e.width) = w; 
# 137
(e.height) = h; 
# 138
(e.depth) = d; 
# 140
return e; 
# 141
} 
# 73 "/usr/include/vector_functions.h" 3
static inline char1 make_char1(signed char x); 
# 75
static inline uchar1 make_uchar1(unsigned char x); 
# 77
static inline char2 make_char2(signed char x, signed char y); 
# 79
static inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
# 81
static inline char3 make_char3(signed char x, signed char y, signed char z); 
# 83
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
# 85
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
# 87
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
# 89
static inline short1 make_short1(short x); 
# 91
static inline ushort1 make_ushort1(unsigned short x); 
# 93
static inline short2 make_short2(short x, short y); 
# 95
static inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
# 97
static inline short3 make_short3(short x, short y, short z); 
# 99
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
# 101
static inline short4 make_short4(short x, short y, short z, short w); 
# 103
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
# 105
static inline int1 make_int1(int x); 
# 107
static inline uint1 make_uint1(unsigned x); 
# 109
static inline int2 make_int2(int x, int y); 
# 111
static inline uint2 make_uint2(unsigned x, unsigned y); 
# 113
static inline int3 make_int3(int x, int y, int z); 
# 115
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
# 117
static inline int4 make_int4(int x, int y, int z, int w); 
# 119
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
# 121
static inline long1 make_long1(long x); 
# 123
static inline ulong1 make_ulong1(unsigned long x); 
# 125
static inline long2 make_long2(long x, long y); 
# 127
static inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
# 129
static inline long3 make_long3(long x, long y, long z); 
# 131
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
# 133
static inline long4 make_long4(long x, long y, long z, long w); 
# 135
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
# 137
static inline float1 make_float1(float x); 
# 139
static inline float2 make_float2(float x, float y); 
# 141
static inline float3 make_float3(float x, float y, float z); 
# 143
static inline float4 make_float4(float x, float y, float z, float w); 
# 145
static inline longlong1 make_longlong1(long long x); 
# 147
static inline ulonglong1 make_ulonglong1(unsigned long long x); 
# 149
static inline longlong2 make_longlong2(long long x, long long y); 
# 151
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y); 
# 153
static inline longlong3 make_longlong3(long long x, long long y, long long z); 
# 155
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z); 
# 157
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w); 
# 159
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w); 
# 161
static inline double1 make_double1(double x); 
# 163
static inline double2 make_double2(double x, double y); 
# 165
static inline double3 make_double3(double x, double y, double z); 
# 167
static inline double4 make_double4(double x, double y, double z, double w); 
# 73 "/usr/include/vector_functions.hpp" 3
static inline char1 make_char1(signed char x) 
# 74
{ 
# 75
char1 t; (t.x) = x; return t; 
# 76
} 
# 78
static inline uchar1 make_uchar1(unsigned char x) 
# 79
{ 
# 80
uchar1 t; (t.x) = x; return t; 
# 81
} 
# 83
static inline char2 make_char2(signed char x, signed char y) 
# 84
{ 
# 85
char2 t; (t.x) = x; (t.y) = y; return t; 
# 86
} 
# 88
static inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
# 89
{ 
# 90
uchar2 t; (t.x) = x; (t.y) = y; return t; 
# 91
} 
# 93
static inline char3 make_char3(signed char x, signed char y, signed char z) 
# 94
{ 
# 95
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 96
} 
# 98
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
# 99
{ 
# 100
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 101
} 
# 103
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
# 104
{ 
# 105
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 106
} 
# 108
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
# 109
{ 
# 110
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 111
} 
# 113
static inline short1 make_short1(short x) 
# 114
{ 
# 115
short1 t; (t.x) = x; return t; 
# 116
} 
# 118
static inline ushort1 make_ushort1(unsigned short x) 
# 119
{ 
# 120
ushort1 t; (t.x) = x; return t; 
# 121
} 
# 123
static inline short2 make_short2(short x, short y) 
# 124
{ 
# 125
short2 t; (t.x) = x; (t.y) = y; return t; 
# 126
} 
# 128
static inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
# 129
{ 
# 130
ushort2 t; (t.x) = x; (t.y) = y; return t; 
# 131
} 
# 133
static inline short3 make_short3(short x, short y, short z) 
# 134
{ 
# 135
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 136
} 
# 138
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
# 139
{ 
# 140
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 141
} 
# 143
static inline short4 make_short4(short x, short y, short z, short w) 
# 144
{ 
# 145
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 146
} 
# 148
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
# 149
{ 
# 150
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 151
} 
# 153
static inline int1 make_int1(int x) 
# 154
{ 
# 155
int1 t; (t.x) = x; return t; 
# 156
} 
# 158
static inline uint1 make_uint1(unsigned x) 
# 159
{ 
# 160
uint1 t; (t.x) = x; return t; 
# 161
} 
# 163
static inline int2 make_int2(int x, int y) 
# 164
{ 
# 165
int2 t; (t.x) = x; (t.y) = y; return t; 
# 166
} 
# 168
static inline uint2 make_uint2(unsigned x, unsigned y) 
# 169
{ 
# 170
uint2 t; (t.x) = x; (t.y) = y; return t; 
# 171
} 
# 173
static inline int3 make_int3(int x, int y, int z) 
# 174
{ 
# 175
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 176
} 
# 178
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
# 179
{ 
# 180
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 181
} 
# 183
static inline int4 make_int4(int x, int y, int z, int w) 
# 184
{ 
# 185
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 186
} 
# 188
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
# 189
{ 
# 190
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 191
} 
# 193
static inline long1 make_long1(long x) 
# 194
{ 
# 195
long1 t; (t.x) = x; return t; 
# 196
} 
# 198
static inline ulong1 make_ulong1(unsigned long x) 
# 199
{ 
# 200
ulong1 t; (t.x) = x; return t; 
# 201
} 
# 203
static inline long2 make_long2(long x, long y) 
# 204
{ 
# 205
long2 t; (t.x) = x; (t.y) = y; return t; 
# 206
} 
# 208
static inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
# 209
{ 
# 210
ulong2 t; (t.x) = x; (t.y) = y; return t; 
# 211
} 
# 213
static inline long3 make_long3(long x, long y, long z) 
# 214
{ 
# 215
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 216
} 
# 218
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
# 219
{ 
# 220
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 221
} 
# 223
static inline long4 make_long4(long x, long y, long z, long w) 
# 224
{ 
# 225
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 226
} 
# 228
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
# 229
{ 
# 230
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 231
} 
# 233
static inline float1 make_float1(float x) 
# 234
{ 
# 235
float1 t; (t.x) = x; return t; 
# 236
} 
# 238
static inline float2 make_float2(float x, float y) 
# 239
{ 
# 240
float2 t; (t.x) = x; (t.y) = y; return t; 
# 241
} 
# 243
static inline float3 make_float3(float x, float y, float z) 
# 244
{ 
# 245
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 246
} 
# 248
static inline float4 make_float4(float x, float y, float z, float w) 
# 249
{ 
# 250
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 251
} 
# 253
static inline longlong1 make_longlong1(long long x) 
# 254
{ 
# 255
longlong1 t; (t.x) = x; return t; 
# 256
} 
# 258
static inline ulonglong1 make_ulonglong1(unsigned long long x) 
# 259
{ 
# 260
ulonglong1 t; (t.x) = x; return t; 
# 261
} 
# 263
static inline longlong2 make_longlong2(long long x, long long y) 
# 264
{ 
# 265
longlong2 t; (t.x) = x; (t.y) = y; return t; 
# 266
} 
# 268
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) 
# 269
{ 
# 270
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
# 271
} 
# 273
static inline longlong3 make_longlong3(long long x, long long y, long long z) 
# 274
{ 
# 275
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 276
} 
# 278
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z) 
# 279
{ 
# 280
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 281
} 
# 283
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w) 
# 284
{ 
# 285
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 286
} 
# 288
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w) 
# 289
{ 
# 290
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 291
} 
# 293
static inline double1 make_double1(double x) 
# 294
{ 
# 295
double1 t; (t.x) = x; return t; 
# 296
} 
# 298
static inline double2 make_double2(double x, double y) 
# 299
{ 
# 300
double2 t; (t.x) = x; (t.y) = y; return t; 
# 301
} 
# 303
static inline double3 make_double3(double x, double y, double z) 
# 304
{ 
# 305
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 306
} 
# 308
static inline double4 make_double4(double x, double y, double z, double w) 
# 309
{ 
# 310
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 311
} 
# 28 "/usr/include/string.h" 3
extern "C" {
# 43 "/usr/include/string.h" 3
extern void *memcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) noexcept(true)
# 44
 __attribute((__nonnull__(1, 2))); 
# 47
extern void *memmove(void * __dest, const void * __src, size_t __n) noexcept(true)
# 48
 __attribute((__nonnull__(1, 2))); 
# 54
extern void *memccpy(void *__restrict__ __dest, const void *__restrict__ __src, int __c, size_t __n) noexcept(true)
# 56
 __attribute((__nonnull__(1, 2))) __attribute((__access__(__write_only__ , 1 , 4 ))); 
# 61
extern void *memset(void * __s, int __c, size_t __n) noexcept(true) __attribute((__nonnull__(1))); 
# 64
extern int memcmp(const void * __s1, const void * __s2, size_t __n) noexcept(true)
# 65
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 80 "/usr/include/string.h" 3
extern int __memcmpeq(const void * __s1, const void * __s2, size_t __n) noexcept(true)
# 81
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 85
extern "C++" {
# 87
extern void *memchr(void * __s, int __c, size_t __n) noexcept(true) __asm__("memchr")
# 88
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 89
extern const void *memchr(const void * __s, int __c, size_t __n) noexcept(true) __asm__("memchr")
# 90
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 105 "/usr/include/string.h" 3
}
# 115 "/usr/include/string.h" 3
extern "C++" void *rawmemchr(void * __s, int __c) noexcept(true) __asm__("rawmemchr")
# 116
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 117
extern "C++" const void *rawmemchr(const void * __s, int __c) noexcept(true) __asm__("rawmemchr")
# 118
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 126
extern "C++" void *memrchr(void * __s, int __c, size_t __n) noexcept(true) __asm__("memrchr")
# 127
 __attribute((__pure__)) __attribute((__nonnull__(1)))
# 128
 __attribute((__access__(__read_only__ , 1 , 3 ))); 
# 129
extern "C++" const void *memrchr(const void * __s, int __c, size_t __n) noexcept(true) __asm__("memrchr")
# 130
 __attribute((__pure__)) __attribute((__nonnull__(1)))
# 131
 __attribute((__access__(__read_only__ , 1 , 3 ))); 
# 141 "/usr/include/string.h" 3
extern char *strcpy(char *__restrict__ __dest, const char *__restrict__ __src) noexcept(true)
# 142
 __attribute((__nonnull__(1, 2))); 
# 144
extern char *strncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) noexcept(true)
# 146
 __attribute((__nonnull__(1, 2))); 
# 149
extern char *strcat(char *__restrict__ __dest, const char *__restrict__ __src) noexcept(true)
# 150
 __attribute((__nonnull__(1, 2))); 
# 152
extern char *strncat(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) noexcept(true)
# 153
 __attribute((__nonnull__(1, 2))); 
# 156
extern int strcmp(const char * __s1, const char * __s2) noexcept(true)
# 157
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 159
extern int strncmp(const char * __s1, const char * __s2, size_t __n) noexcept(true)
# 160
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 163
extern int strcoll(const char * __s1, const char * __s2) noexcept(true)
# 164
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 166
extern size_t strxfrm(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) noexcept(true)
# 168
 __attribute((__nonnull__(2))) __attribute((__access__(__write_only__ , 1 , 3 ))); 
# 175
extern int strcoll_l(const char * __s1, const char * __s2, locale_t __l) noexcept(true)
# 176
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 179
extern size_t strxfrm_l(char * __dest, const char * __src, size_t __n, locale_t __l) noexcept(true)
# 180
 __attribute((__nonnull__(2, 4)))
# 181
 __attribute((__access__(__write_only__ , 1 , 3 ))); 
# 187
extern char *strdup(const char * __s) noexcept(true)
# 188
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 195
extern char *strndup(const char * __string, size_t __n) noexcept(true)
# 196
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 224 "/usr/include/string.h" 3
extern "C++" {
# 226
extern char *strchr(char * __s, int __c) noexcept(true) __asm__("strchr")
# 227
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 228
extern const char *strchr(const char * __s, int __c) noexcept(true) __asm__("strchr")
# 229
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 244 "/usr/include/string.h" 3
}
# 251
extern "C++" {
# 253
extern char *strrchr(char * __s, int __c) noexcept(true) __asm__("strrchr")
# 254
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 255
extern const char *strrchr(const char * __s, int __c) noexcept(true) __asm__("strrchr")
# 256
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 271 "/usr/include/string.h" 3
}
# 281 "/usr/include/string.h" 3
extern "C++" char *strchrnul(char * __s, int __c) noexcept(true) __asm__("strchrnul")
# 282
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 283
extern "C++" const char *strchrnul(const char * __s, int __c) noexcept(true) __asm__("strchrnul")
# 284
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 293 "/usr/include/string.h" 3
extern size_t strcspn(const char * __s, const char * __reject) noexcept(true)
# 294
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 297
extern size_t strspn(const char * __s, const char * __accept) noexcept(true)
# 298
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 301
extern "C++" {
# 303
extern char *strpbrk(char * __s, const char * __accept) noexcept(true) __asm__("strpbrk")
# 304
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 305
extern const char *strpbrk(const char * __s, const char * __accept) noexcept(true) __asm__("strpbrk")
# 306
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 321 "/usr/include/string.h" 3
}
# 328
extern "C++" {
# 330
extern char *strstr(char * __haystack, const char * __needle) noexcept(true) __asm__("strstr")
# 331
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 332
extern const char *strstr(const char * __haystack, const char * __needle) noexcept(true) __asm__("strstr")
# 333
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 348 "/usr/include/string.h" 3
}
# 356
extern char *strtok(char *__restrict__ __s, const char *__restrict__ __delim) noexcept(true)
# 357
 __attribute((__nonnull__(2))); 
# 361
extern char *__strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) noexcept(true)
# 364
 __attribute((__nonnull__(2, 3))); 
# 366
extern char *strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) noexcept(true)
# 368
 __attribute((__nonnull__(2, 3))); 
# 374
extern "C++" char *strcasestr(char * __haystack, const char * __needle) noexcept(true) __asm__("strcasestr")
# 375
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 376
extern "C++" const char *strcasestr(const char * __haystack, const char * __needle) noexcept(true) __asm__("strcasestr")
# 378
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 389 "/usr/include/string.h" 3
extern void *memmem(const void * __haystack, size_t __haystacklen, const void * __needle, size_t __needlelen) noexcept(true)
# 391
 __attribute((__pure__)) __attribute((__nonnull__(1, 3)))
# 392
 __attribute((__access__(__read_only__ , 1 , 2 )))
# 393
 __attribute((__access__(__read_only__ , 3 , 4 ))); 
# 397
extern void *__mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) noexcept(true)
# 399
 __attribute((__nonnull__(1, 2))); 
# 400
extern void *mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) noexcept(true)
# 402
 __attribute((__nonnull__(1, 2))); 
# 407
extern size_t strlen(const char * __s) noexcept(true)
# 408
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 413
extern size_t strnlen(const char * __string, size_t __maxlen) noexcept(true)
# 414
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 419
extern char *strerror(int __errnum) noexcept(true); 
# 444 "/usr/include/string.h" 3
extern char *strerror_r(int __errnum, char * __buf, size_t __buflen) noexcept(true)
# 445
 __attribute((__nonnull__(2))) __attribute((__access__(__write_only__ , 2 , 3 ))); 
# 450
extern const char *strerrordesc_np(int __err) noexcept(true); 
# 452
extern const char *strerrorname_np(int __err) noexcept(true); 
# 458
extern char *strerror_l(int __errnum, locale_t __l) noexcept(true); 
# 30 "/usr/include/strings.h" 3
extern "C" {
# 34
extern int bcmp(const void * __s1, const void * __s2, size_t __n) noexcept(true)
# 35
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 38
extern void bcopy(const void * __src, void * __dest, size_t __n) noexcept(true)
# 39
 __attribute((__nonnull__(1, 2))); 
# 42
extern void bzero(void * __s, size_t __n) noexcept(true) __attribute((__nonnull__(1))); 
# 46
extern "C++" {
# 48
extern char *index(char * __s, int __c) noexcept(true) __asm__("index")
# 49
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 50
extern const char *index(const char * __s, int __c) noexcept(true) __asm__("index")
# 51
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 66 "/usr/include/strings.h" 3
}
# 74
extern "C++" {
# 76
extern char *rindex(char * __s, int __c) noexcept(true) __asm__("rindex")
# 77
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 78
extern const char *rindex(const char * __s, int __c) noexcept(true) __asm__("rindex")
# 79
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 94 "/usr/include/strings.h" 3
}
# 104 "/usr/include/strings.h" 3
extern int ffs(int __i) noexcept(true) __attribute((const)); 
# 110
extern int ffsl(long __l) noexcept(true) __attribute((const)); 
# 111
__extension__ extern int ffsll(long long __ll) noexcept(true)
# 112
 __attribute((const)); 
# 116
extern int strcasecmp(const char * __s1, const char * __s2) noexcept(true)
# 117
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 120
extern int strncasecmp(const char * __s1, const char * __s2, size_t __n) noexcept(true)
# 121
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 128
extern int strcasecmp_l(const char * __s1, const char * __s2, locale_t __loc) noexcept(true)
# 129
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 133
extern int strncasecmp_l(const char * __s1, const char * __s2, size_t __n, locale_t __loc) noexcept(true)
# 135
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 4))); 
# 138
}
# 466 "/usr/include/string.h" 3
extern void explicit_bzero(void * __s, size_t __n) noexcept(true) __attribute((__nonnull__(1)))
# 467
 __attribute((__access__(__write_only__ , 1 , 2 ))); 
# 471
extern char *strsep(char **__restrict__ __stringp, const char *__restrict__ __delim) noexcept(true)
# 473
 __attribute((__nonnull__(1, 2))); 
# 478
extern char *strsignal(int __sig) noexcept(true); 
# 482
extern const char *sigabbrev_np(int __sig) noexcept(true); 
# 485
extern const char *sigdescr_np(int __sig) noexcept(true); 
# 489
extern char *__stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) noexcept(true)
# 490
 __attribute((__nonnull__(1, 2))); 
# 491
extern char *stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) noexcept(true)
# 492
 __attribute((__nonnull__(1, 2))); 
# 496
extern char *__stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) noexcept(true)
# 498
 __attribute((__nonnull__(1, 2))); 
# 499
extern char *stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) noexcept(true)
# 501
 __attribute((__nonnull__(1, 2))); 
# 506
extern size_t strlcpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) noexcept(true)
# 508
 __attribute((__nonnull__(1, 2))) __attribute((__access__(__write_only__ , 1 , 3 ))); 
# 512
extern size_t strlcat(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) noexcept(true)
# 514
 __attribute((__nonnull__(1, 2))) __attribute((__access__(__read_write__ , 1 , 3 ))); 
# 519
extern int strverscmp(const char * __s1, const char * __s2) noexcept(true)
# 520
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 523
extern char *strfry(char * __string) noexcept(true) __attribute((__nonnull__(1))); 
# 526
extern void *memfrob(void * __s, size_t __n) noexcept(true) __attribute((__nonnull__(1)))
# 527
 __attribute((__access__(__read_write__ , 1 , 2 ))); 
# 535
extern "C++" char *basename(char * __filename) noexcept(true) __asm__("basename")
# 536
 __attribute((__nonnull__(1))); 
# 537
extern "C++" const char *basename(const char * __filename) noexcept(true) __asm__("basename")
# 538
 __attribute((__nonnull__(1))); 
# 552 "/usr/include/string.h" 3
}
# 26 "/usr/include/x86_64-linux-gnu/bits/timex.h" 3
struct timex { 
# 58 "/usr/include/x86_64-linux-gnu/bits/timex.h" 3
unsigned modes; 
# 59
__syscall_slong_t offset; 
# 60
__syscall_slong_t freq; 
# 61
__syscall_slong_t maxerror; 
# 62
__syscall_slong_t esterror; 
# 63
int status; 
# 64
__syscall_slong_t constant; 
# 65
__syscall_slong_t precision; 
# 66
__syscall_slong_t tolerance; 
# 67
timeval time; 
# 68
__syscall_slong_t tick; 
# 69
__syscall_slong_t ppsfreq; 
# 70
__syscall_slong_t jitter; 
# 71
int shift; 
# 72
__syscall_slong_t stabil; 
# 73
__syscall_slong_t jitcnt; 
# 74
__syscall_slong_t calcnt; 
# 75
__syscall_slong_t errcnt; 
# 76
__syscall_slong_t stbcnt; 
# 78
int tai; 
# 81
int:32; int:32; int:32; int:32; 
# 82
int:32; int:32; int:32; int:32; 
# 83
int:32; int:32; int:32; 
# 85
}; 
# 75 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
extern "C" {
# 78
extern int clock_adjtime(__clockid_t __clock_id, timex * __utx) noexcept(true) __attribute((__nonnull__(2))); 
# 90 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
}
# 7 "/usr/include/x86_64-linux-gnu/bits/types/struct_tm.h" 3
struct tm { 
# 9
int tm_sec; 
# 10
int tm_min; 
# 11
int tm_hour; 
# 12
int tm_mday; 
# 13
int tm_mon; 
# 14
int tm_year; 
# 15
int tm_wday; 
# 16
int tm_yday; 
# 17
int tm_isdst; 
# 20
long tm_gmtoff; 
# 21
const char *tm_zone; 
# 26
}; 
# 8 "/usr/include/x86_64-linux-gnu/bits/types/struct_itimerspec.h" 3
struct itimerspec { 
# 10
timespec it_interval; 
# 11
timespec it_value; 
# 12
}; 
# 49 "/usr/include/time.h" 3
struct sigevent; 
# 68 "/usr/include/time.h" 3
extern "C" {
# 72
extern clock_t clock() noexcept(true); 
# 76
extern time_t time(time_t * __timer) noexcept(true); 
# 79
extern double difftime(time_t __time1, time_t __time0) noexcept(true)
# 80
 __attribute((const)); 
# 83
extern time_t mktime(tm * __tp) noexcept(true); 
# 100 "/usr/include/time.h" 3
extern size_t strftime(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp) noexcept(true)
# 103
 __attribute((__nonnull__(1, 3, 4))); 
# 108
extern char *strptime(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp) noexcept(true); 
# 117
extern size_t strftime_l(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp, locale_t __loc) noexcept(true); 
# 124
extern char *strptime_l(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp, locale_t __loc) noexcept(true); 
# 133
extern tm *gmtime(const time_t * __timer) noexcept(true); 
# 137
extern tm *localtime(const time_t * __timer) noexcept(true); 
# 155 "/usr/include/time.h" 3
extern tm *gmtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) noexcept(true); 
# 160
extern tm *localtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) noexcept(true); 
# 180 "/usr/include/time.h" 3
extern char *asctime(const tm * __tp) noexcept(true); 
# 184
extern char *ctime(const time_t * __timer) noexcept(true); 
# 198 "/usr/include/time.h" 3
extern char *asctime_r(const tm *__restrict__ __tp, char *__restrict__ __buf) noexcept(true); 
# 203
extern char *ctime_r(const time_t *__restrict__ __timer, char *__restrict__ __buf) noexcept(true); 
# 218 "/usr/include/time.h" 3
extern char *__tzname[2]; 
# 219
extern int __daylight; 
# 220
extern long __timezone; 
# 225
extern char *tzname[2]; 
# 229
extern void tzset() noexcept(true); 
# 233
extern int daylight; 
# 234
extern long timezone; 
# 247 "/usr/include/time.h" 3
extern time_t timegm(tm * __tp) noexcept(true); 
# 264 "/usr/include/time.h" 3
extern time_t timelocal(tm * __tp) noexcept(true); 
# 272
extern int dysize(int __year) noexcept(true) __attribute((const)); 
# 282 "/usr/include/time.h" 3
extern int nanosleep(const timespec * __requested_time, timespec * __remaining); 
# 286
extern int clock_getres(clockid_t __clock_id, timespec * __res) noexcept(true); 
# 289
extern int clock_gettime(clockid_t __clock_id, timespec * __tp) noexcept(true)
# 290
 __attribute((__nonnull__(2))); 
# 293
extern int clock_settime(clockid_t __clock_id, const timespec * __tp) noexcept(true)
# 294
 __attribute((__nonnull__(2))); 
# 324 "/usr/include/time.h" 3
extern int clock_nanosleep(clockid_t __clock_id, int __flags, const timespec * __req, timespec * __rem); 
# 339 "/usr/include/time.h" 3
extern int clock_getcpuclockid(pid_t __pid, clockid_t * __clock_id) noexcept(true); 
# 344
extern int timer_create(clockid_t __clock_id, sigevent *__restrict__ __evp, timer_t *__restrict__ __timerid) noexcept(true); 
# 349
extern int timer_delete(timer_t __timerid) noexcept(true); 
# 353
extern int timer_settime(timer_t __timerid, int __flags, const itimerspec *__restrict__ __value, itimerspec *__restrict__ __ovalue) noexcept(true); 
# 358
extern int timer_gettime(timer_t __timerid, itimerspec * __value) noexcept(true); 
# 377 "/usr/include/time.h" 3
extern int timer_getoverrun(timer_t __timerid) noexcept(true); 
# 384
extern int timespec_get(timespec * __ts, int __base) noexcept(true)
# 385
 __attribute((__nonnull__(1))); 
# 400 "/usr/include/time.h" 3
extern int timespec_getres(timespec * __ts, int __base) noexcept(true); 
# 426 "/usr/include/time.h" 3
extern int getdate_err; 
# 435 "/usr/include/time.h" 3
extern tm *getdate(const char * __string); 
# 449 "/usr/include/time.h" 3
extern int getdate_r(const char *__restrict__ __string, tm *__restrict__ __resbufp); 
# 453
}
# 88 "/usr/include/crt/common_functions.h" 3
extern "C" {
# 91
extern clock_t clock() noexcept(true); 
# 96
extern void *memset(void *, int, size_t) noexcept(true); 
# 97
extern void *memcpy(void *, const void *, size_t) noexcept(true); 
# 99
}
# 121 "/usr/include/crt/math_functions.h" 3
extern "C" {
# 219 "/usr/include/crt/math_functions.h" 3
extern int abs(int a) noexcept(true); 
# 227
extern long labs(long a) noexcept(true); 
# 235
extern long long llabs(long long a) noexcept(true); 
# 285 "/usr/include/crt/math_functions.h" 3
extern double fabs(double x) noexcept(true); 
# 328 "/usr/include/crt/math_functions.h" 3
extern float fabsf(float x) noexcept(true); 
# 338 "/usr/include/crt/math_functions.h" 3
extern inline int min(const int a, const int b); 
# 345
extern inline unsigned umin(const unsigned a, const unsigned b); 
# 352
extern inline long long llmin(const long long a, const long long b); 
# 359
extern inline unsigned long long ullmin(const unsigned long long a, const unsigned long long b); 
# 380 "/usr/include/crt/math_functions.h" 3
extern float fminf(float x, float y) noexcept(true); 
# 400 "/usr/include/crt/math_functions.h" 3
extern double fmin(double x, double y) noexcept(true); 
# 413 "/usr/include/crt/math_functions.h" 3
extern inline int max(const int a, const int b); 
# 421
extern inline unsigned umax(const unsigned a, const unsigned b); 
# 428
extern inline long long llmax(const long long a, const long long b); 
# 435
extern inline unsigned long long ullmax(const unsigned long long a, const unsigned long long b); 
# 456 "/usr/include/crt/math_functions.h" 3
extern float fmaxf(float x, float y) noexcept(true); 
# 476 "/usr/include/crt/math_functions.h" 3
extern double fmax(double, double) noexcept(true); 
# 520 "/usr/include/crt/math_functions.h" 3
extern double sin(double x) noexcept(true); 
# 553 "/usr/include/crt/math_functions.h" 3
extern double cos(double x) noexcept(true); 
# 572 "/usr/include/crt/math_functions.h" 3
extern void sincos(double x, double * sptr, double * cptr) noexcept(true); 
# 588 "/usr/include/crt/math_functions.h" 3
extern void sincosf(float x, float * sptr, float * cptr) noexcept(true); 
# 633 "/usr/include/crt/math_functions.h" 3
extern double tan(double x) noexcept(true); 
# 702 "/usr/include/crt/math_functions.h" 3
extern double sqrt(double x) noexcept(true); 
# 774 "/usr/include/crt/math_functions.h" 3
extern double rsqrt(double x); 
# 844 "/usr/include/crt/math_functions.h" 3
extern float rsqrtf(float x); 
# 900 "/usr/include/crt/math_functions.h" 3
extern double log2(double x) noexcept(true); 
# 965 "/usr/include/crt/math_functions.h" 3
extern double exp2(double x) noexcept(true); 
# 1030 "/usr/include/crt/math_functions.h" 3
extern float exp2f(float x) noexcept(true); 
# 1097 "/usr/include/crt/math_functions.h" 3
extern double exp10(double x) noexcept(true); 
# 1160 "/usr/include/crt/math_functions.h" 3
extern float exp10f(float x) noexcept(true); 
# 1253 "/usr/include/crt/math_functions.h" 3
extern double expm1(double x) noexcept(true); 
# 1345 "/usr/include/crt/math_functions.h" 3
extern float expm1f(float x) noexcept(true); 
# 1401 "/usr/include/crt/math_functions.h" 3
extern float log2f(float x) noexcept(true); 
# 1455 "/usr/include/crt/math_functions.h" 3
extern double log10(double x) noexcept(true); 
# 1525 "/usr/include/crt/math_functions.h" 3
extern double log(double x) noexcept(true); 
# 1621 "/usr/include/crt/math_functions.h" 3
extern double log1p(double x) noexcept(true); 
# 1720 "/usr/include/crt/math_functions.h" 3
extern float log1pf(float x) noexcept(true); 
# 1784 "/usr/include/crt/math_functions.h" 3
extern double floor(double x) noexcept(true); 
# 1863 "/usr/include/crt/math_functions.h" 3
extern double exp(double x) noexcept(true); 
# 1904 "/usr/include/crt/math_functions.h" 3
extern double cosh(double x) noexcept(true); 
# 1954 "/usr/include/crt/math_functions.h" 3
extern double sinh(double x) noexcept(true); 
# 2004 "/usr/include/crt/math_functions.h" 3
extern double tanh(double x) noexcept(true); 
# 2059 "/usr/include/crt/math_functions.h" 3
extern double acosh(double x) noexcept(true); 
# 2117 "/usr/include/crt/math_functions.h" 3
extern float acoshf(float x) noexcept(true); 
# 2170 "/usr/include/crt/math_functions.h" 3
extern double asinh(double x) noexcept(true); 
# 2223 "/usr/include/crt/math_functions.h" 3
extern float asinhf(float x) noexcept(true); 
# 2277 "/usr/include/crt/math_functions.h" 3
extern double atanh(double x) noexcept(true); 
# 2331 "/usr/include/crt/math_functions.h" 3
extern float atanhf(float x) noexcept(true); 
# 2380 "/usr/include/crt/math_functions.h" 3
extern double ldexp(double x, int exp) noexcept(true); 
# 2426 "/usr/include/crt/math_functions.h" 3
extern float ldexpf(float x, int exp) noexcept(true); 
# 2478 "/usr/include/crt/math_functions.h" 3
extern double logb(double x) noexcept(true); 
# 2533 "/usr/include/crt/math_functions.h" 3
extern float logbf(float x) noexcept(true); 
# 2573 "/usr/include/crt/math_functions.h" 3
extern int ilogb(double x) noexcept(true); 
# 2613 "/usr/include/crt/math_functions.h" 3
extern int ilogbf(float x) noexcept(true); 
# 2689 "/usr/include/crt/math_functions.h" 3
extern double scalbn(double x, int n) noexcept(true); 
# 2765 "/usr/include/crt/math_functions.h" 3
extern float scalbnf(float x, int n) noexcept(true); 
# 2841 "/usr/include/crt/math_functions.h" 3
extern double scalbln(double x, long n) noexcept(true); 
# 2917 "/usr/include/crt/math_functions.h" 3
extern float scalblnf(float x, long n) noexcept(true); 
# 2994 "/usr/include/crt/math_functions.h" 3
extern double frexp(double x, int * nptr) noexcept(true); 
# 3068 "/usr/include/crt/math_functions.h" 3
extern float frexpf(float x, int * nptr) noexcept(true); 
# 3120 "/usr/include/crt/math_functions.h" 3
extern double round(double x) noexcept(true); 
# 3175 "/usr/include/crt/math_functions.h" 3
extern float roundf(float x) noexcept(true); 
# 3193 "/usr/include/crt/math_functions.h" 3
extern long lround(double x) noexcept(true); 
# 3211 "/usr/include/crt/math_functions.h" 3
extern long lroundf(float x) noexcept(true); 
# 3229 "/usr/include/crt/math_functions.h" 3
extern long long llround(double x) noexcept(true); 
# 3247 "/usr/include/crt/math_functions.h" 3
extern long long llroundf(float x) noexcept(true); 
# 3375 "/usr/include/crt/math_functions.h" 3
extern float rintf(float x) noexcept(true); 
# 3392 "/usr/include/crt/math_functions.h" 3
extern long lrint(double x) noexcept(true); 
# 3409 "/usr/include/crt/math_functions.h" 3
extern long lrintf(float x) noexcept(true); 
# 3426 "/usr/include/crt/math_functions.h" 3
extern long long llrint(double x) noexcept(true); 
# 3443 "/usr/include/crt/math_functions.h" 3
extern long long llrintf(float x) noexcept(true); 
# 3496 "/usr/include/crt/math_functions.h" 3
extern double nearbyint(double x) noexcept(true); 
# 3549 "/usr/include/crt/math_functions.h" 3
extern float nearbyintf(float x) noexcept(true); 
# 3611 "/usr/include/crt/math_functions.h" 3
extern double ceil(double x) noexcept(true); 
# 3661 "/usr/include/crt/math_functions.h" 3
extern double trunc(double x) noexcept(true); 
# 3714 "/usr/include/crt/math_functions.h" 3
extern float truncf(float x) noexcept(true); 
# 3740 "/usr/include/crt/math_functions.h" 3
extern double fdim(double x, double y) noexcept(true); 
# 3766 "/usr/include/crt/math_functions.h" 3
extern float fdimf(float x, float y) noexcept(true); 
# 4066 "/usr/include/crt/math_functions.h" 3
extern double atan2(double y, double x) noexcept(true); 
# 4137 "/usr/include/crt/math_functions.h" 3
extern double atan(double x) noexcept(true); 
# 4160 "/usr/include/crt/math_functions.h" 3
extern double acos(double x) noexcept(true); 
# 4211 "/usr/include/crt/math_functions.h" 3
extern double asin(double x) noexcept(true); 
# 4279 "/usr/include/crt/math_functions.h" 3
extern double hypot(double x, double y) noexcept(true); 
# 4402 "/usr/include/crt/math_functions.h" 3
extern float hypotf(float x, float y) noexcept(true); 
# 5188 "/usr/include/crt/math_functions.h" 3
extern double cbrt(double x) noexcept(true); 
# 5274 "/usr/include/crt/math_functions.h" 3
extern float cbrtf(float x) noexcept(true); 
# 5329 "/usr/include/crt/math_functions.h" 3
extern double rcbrt(double x); 
# 5379 "/usr/include/crt/math_functions.h" 3
extern float rcbrtf(float x); 
# 5439 "/usr/include/crt/math_functions.h" 3
extern double sinpi(double x); 
# 5499 "/usr/include/crt/math_functions.h" 3
extern float sinpif(float x); 
# 5551 "/usr/include/crt/math_functions.h" 3
extern double cospi(double x); 
# 5603 "/usr/include/crt/math_functions.h" 3
extern float cospif(float x); 
# 5633 "/usr/include/crt/math_functions.h" 3
extern void sincospi(double x, double * sptr, double * cptr); 
# 5663 "/usr/include/crt/math_functions.h" 3
extern void sincospif(float x, float * sptr, float * cptr); 
# 5996 "/usr/include/crt/math_functions.h" 3
extern double pow(double x, double y) noexcept(true); 
# 6052 "/usr/include/crt/math_functions.h" 3
extern double modf(double x, double * iptr) noexcept(true); 
# 6111 "/usr/include/crt/math_functions.h" 3
extern double fmod(double x, double y) noexcept(true); 
# 6207 "/usr/include/crt/math_functions.h" 3
extern double remainder(double x, double y) noexcept(true); 
# 6306 "/usr/include/crt/math_functions.h" 3
extern float remainderf(float x, float y) noexcept(true); 
# 6378 "/usr/include/crt/math_functions.h" 3
extern double remquo(double x, double y, int * quo) noexcept(true); 
# 6450 "/usr/include/crt/math_functions.h" 3
extern float remquof(float x, float y, int * quo) noexcept(true); 
# 6491 "/usr/include/crt/math_functions.h" 3
extern double j0(double x) noexcept(true); 
# 6533 "/usr/include/crt/math_functions.h" 3
extern float j0f(float x) noexcept(true); 
# 6602 "/usr/include/crt/math_functions.h" 3
extern double j1(double x) noexcept(true); 
# 6671 "/usr/include/crt/math_functions.h" 3
extern float j1f(float x) noexcept(true); 
# 6714 "/usr/include/crt/math_functions.h" 3
extern double jn(int n, double x) noexcept(true); 
# 6757 "/usr/include/crt/math_functions.h" 3
extern float jnf(int n, float x) noexcept(true); 
# 6818 "/usr/include/crt/math_functions.h" 3
extern double y0(double x) noexcept(true); 
# 6879 "/usr/include/crt/math_functions.h" 3
extern float y0f(float x) noexcept(true); 
# 6940 "/usr/include/crt/math_functions.h" 3
extern double y1(double x) noexcept(true); 
# 7001 "/usr/include/crt/math_functions.h" 3
extern float y1f(float x) noexcept(true); 
# 7064 "/usr/include/crt/math_functions.h" 3
extern double yn(int n, double x) noexcept(true); 
# 7127 "/usr/include/crt/math_functions.h" 3
extern float ynf(int n, float x) noexcept(true); 
# 7316 "/usr/include/crt/math_functions.h" 3
extern double erf(double x) noexcept(true); 
# 7398 "/usr/include/crt/math_functions.h" 3
extern float erff(float x) noexcept(true); 
# 7470 "/usr/include/crt/math_functions.h" 3
extern double erfinv(double x); 
# 7535 "/usr/include/crt/math_functions.h" 3
extern float erfinvf(float x); 
# 7574 "/usr/include/crt/math_functions.h" 3
extern double erfc(double x) noexcept(true); 
# 7612 "/usr/include/crt/math_functions.h" 3
extern float erfcf(float x) noexcept(true); 
# 7729 "/usr/include/crt/math_functions.h" 3
extern double lgamma(double x) noexcept(true); 
# 7791 "/usr/include/crt/math_functions.h" 3
extern double erfcinv(double x); 
# 7846 "/usr/include/crt/math_functions.h" 3
extern float erfcinvf(float x); 
# 7914 "/usr/include/crt/math_functions.h" 3
extern double normcdfinv(double x); 
# 7982 "/usr/include/crt/math_functions.h" 3
extern float normcdfinvf(float x); 
# 8025 "/usr/include/crt/math_functions.h" 3
extern double normcdf(double x); 
# 8068 "/usr/include/crt/math_functions.h" 3
extern float normcdff(float x); 
# 8132 "/usr/include/crt/math_functions.h" 3
extern double erfcx(double x); 
# 8196 "/usr/include/crt/math_functions.h" 3
extern float erfcxf(float x); 
# 8315 "/usr/include/crt/math_functions.h" 3
extern float lgammaf(float x) noexcept(true); 
# 8413 "/usr/include/crt/math_functions.h" 3
extern double tgamma(double x) noexcept(true); 
# 8511 "/usr/include/crt/math_functions.h" 3
extern float tgammaf(float x) noexcept(true); 
# 8524 "/usr/include/crt/math_functions.h" 3
extern double copysign(double x, double y) noexcept(true); 
# 8537 "/usr/include/crt/math_functions.h" 3
extern float copysignf(float x, float y) noexcept(true); 
# 8556 "/usr/include/crt/math_functions.h" 3
extern double nextafter(double x, double y) noexcept(true); 
# 8575 "/usr/include/crt/math_functions.h" 3
extern float nextafterf(float x, float y) noexcept(true); 
# 8591 "/usr/include/crt/math_functions.h" 3
extern double nan(const char * tagp) noexcept(true); 
# 8607 "/usr/include/crt/math_functions.h" 3
extern float nanf(const char * tagp) noexcept(true); 
# 8614
extern int __isinff(float) noexcept(true); 
# 8615
extern int __isnanf(float) noexcept(true); 
# 8625 "/usr/include/crt/math_functions.h" 3
extern int __finite(double) noexcept(true); 
# 8626
extern int __finitef(float) noexcept(true); 
# 8627
extern int __signbit(double) noexcept(true); 
# 8628
extern int __isnan(double) noexcept(true); 
# 8629
extern int __isinf(double) noexcept(true); 
# 8632
extern int __signbitf(float) noexcept(true); 
# 8791 "/usr/include/crt/math_functions.h" 3
extern double fma(double x, double y, double z) noexcept(true); 
# 8949 "/usr/include/crt/math_functions.h" 3
extern float fmaf(float x, float y, float z) noexcept(true); 
# 8960 "/usr/include/crt/math_functions.h" 3
extern int __signbitl(long double) noexcept(true); 
# 8966
extern int __finitel(long double) noexcept(true); 
# 8967
extern int __isinfl(long double) noexcept(true); 
# 8968
extern int __isnanl(long double) noexcept(true); 
# 9018 "/usr/include/crt/math_functions.h" 3
extern float acosf(float x) noexcept(true); 
# 9077 "/usr/include/crt/math_functions.h" 3
extern float asinf(float x) noexcept(true); 
# 9157 "/usr/include/crt/math_functions.h" 3
extern float atanf(float x) noexcept(true); 
# 9454 "/usr/include/crt/math_functions.h" 3
extern float atan2f(float y, float x) noexcept(true); 
# 9488 "/usr/include/crt/math_functions.h" 3
extern float cosf(float x) noexcept(true); 
# 9530 "/usr/include/crt/math_functions.h" 3
extern float sinf(float x) noexcept(true); 
# 9572 "/usr/include/crt/math_functions.h" 3
extern float tanf(float x) noexcept(true); 
# 9613 "/usr/include/crt/math_functions.h" 3
extern float coshf(float x) noexcept(true); 
# 9663 "/usr/include/crt/math_functions.h" 3
extern float sinhf(float x) noexcept(true); 
# 9713 "/usr/include/crt/math_functions.h" 3
extern float tanhf(float x) noexcept(true); 
# 9765 "/usr/include/crt/math_functions.h" 3
extern float logf(float x) noexcept(true); 
# 9845 "/usr/include/crt/math_functions.h" 3
extern float expf(float x) noexcept(true); 
# 9897 "/usr/include/crt/math_functions.h" 3
extern float log10f(float x) noexcept(true); 
# 9952 "/usr/include/crt/math_functions.h" 3
extern float modff(float x, float * iptr) noexcept(true); 
# 10282 "/usr/include/crt/math_functions.h" 3
extern float powf(float x, float y) noexcept(true); 
# 10351 "/usr/include/crt/math_functions.h" 3
extern float sqrtf(float x) noexcept(true); 
# 10410 "/usr/include/crt/math_functions.h" 3
extern float ceilf(float x) noexcept(true); 
# 10471 "/usr/include/crt/math_functions.h" 3
extern float floorf(float x) noexcept(true); 
# 10529 "/usr/include/crt/math_functions.h" 3
extern float fmodf(float x, float y) noexcept(true); 
# 10544 "/usr/include/crt/math_functions.h" 3
}
# 67 "/usr/include/c++/12/bits/cpp_type_traits.h" 3
extern "C++" {
# 69
namespace std __attribute((__visibility__("default"))) { 
# 73
struct __true_type { }; 
# 74
struct __false_type { }; 
# 76
template< bool > 
# 77
struct __truth_type { 
# 78
typedef __false_type __type; }; 
# 81
template<> struct __truth_type< true>  { 
# 82
typedef __true_type __type; }; 
# 86
template< class _Sp, class _Tp> 
# 87
struct __traitor { 
# 89
enum { __value = ((bool)_Sp::__value) || ((bool)_Tp::__value)}; 
# 90
typedef typename __truth_type< __value> ::__type __type; 
# 91
}; 
# 94
template< class , class > 
# 95
struct __are_same { 
# 97
enum { __value}; 
# 98
typedef __false_type __type; 
# 99
}; 
# 101
template< class _Tp> 
# 102
struct __are_same< _Tp, _Tp>  { 
# 104
enum { __value = 1}; 
# 105
typedef __true_type __type; 
# 106
}; 
# 109
template< class _Tp> 
# 110
struct __is_void { 
# 112
enum { __value}; 
# 113
typedef __false_type __type; 
# 114
}; 
# 117
template<> struct __is_void< void>  { 
# 119
enum { __value = 1}; 
# 120
typedef __true_type __type; 
# 121
}; 
# 126
template< class _Tp> 
# 127
struct __is_integer { 
# 129
enum { __value}; 
# 130
typedef __false_type __type; 
# 131
}; 
# 138
template<> struct __is_integer< bool>  { 
# 140
enum { __value = 1}; 
# 141
typedef __true_type __type; 
# 142
}; 
# 145
template<> struct __is_integer< char>  { 
# 147
enum { __value = 1}; 
# 148
typedef __true_type __type; 
# 149
}; 
# 152
template<> struct __is_integer< signed char>  { 
# 154
enum { __value = 1}; 
# 155
typedef __true_type __type; 
# 156
}; 
# 159
template<> struct __is_integer< unsigned char>  { 
# 161
enum { __value = 1}; 
# 162
typedef __true_type __type; 
# 163
}; 
# 167
template<> struct __is_integer< wchar_t>  { 
# 169
enum { __value = 1}; 
# 170
typedef __true_type __type; 
# 171
}; 
# 185 "/usr/include/c++/12/bits/cpp_type_traits.h" 3
template<> struct __is_integer< char16_t>  { 
# 187
enum { __value = 1}; 
# 188
typedef __true_type __type; 
# 189
}; 
# 192
template<> struct __is_integer< char32_t>  { 
# 194
enum { __value = 1}; 
# 195
typedef __true_type __type; 
# 196
}; 
# 200
template<> struct __is_integer< short>  { 
# 202
enum { __value = 1}; 
# 203
typedef __true_type __type; 
# 204
}; 
# 207
template<> struct __is_integer< unsigned short>  { 
# 209
enum { __value = 1}; 
# 210
typedef __true_type __type; 
# 211
}; 
# 214
template<> struct __is_integer< int>  { 
# 216
enum { __value = 1}; 
# 217
typedef __true_type __type; 
# 218
}; 
# 221
template<> struct __is_integer< unsigned>  { 
# 223
enum { __value = 1}; 
# 224
typedef __true_type __type; 
# 225
}; 
# 228
template<> struct __is_integer< long>  { 
# 230
enum { __value = 1}; 
# 231
typedef __true_type __type; 
# 232
}; 
# 235
template<> struct __is_integer< unsigned long>  { 
# 237
enum { __value = 1}; 
# 238
typedef __true_type __type; 
# 239
}; 
# 242
template<> struct __is_integer< long long>  { 
# 244
enum { __value = 1}; 
# 245
typedef __true_type __type; 
# 246
}; 
# 249
template<> struct __is_integer< unsigned long long>  { 
# 251
enum { __value = 1}; 
# 252
typedef __true_type __type; 
# 253
}; 
# 272 "/usr/include/c++/12/bits/cpp_type_traits.h" 3
template<> struct __is_integer< __int128>  { enum { __value = 1}; typedef __true_type __type; }; template<> struct __is_integer< unsigned __int128>  { enum { __value = 1}; typedef __true_type __type; }; 
# 289 "/usr/include/c++/12/bits/cpp_type_traits.h" 3
template< class _Tp> 
# 290
struct __is_floating { 
# 292
enum { __value}; 
# 293
typedef __false_type __type; 
# 294
}; 
# 298
template<> struct __is_floating< float>  { 
# 300
enum { __value = 1}; 
# 301
typedef __true_type __type; 
# 302
}; 
# 305
template<> struct __is_floating< double>  { 
# 307
enum { __value = 1}; 
# 308
typedef __true_type __type; 
# 309
}; 
# 312
template<> struct __is_floating< long double>  { 
# 314
enum { __value = 1}; 
# 315
typedef __true_type __type; 
# 316
}; 
# 321
template< class _Tp> 
# 322
struct __is_pointer { 
# 324
enum { __value}; 
# 325
typedef __false_type __type; 
# 326
}; 
# 328
template< class _Tp> 
# 329
struct __is_pointer< _Tp *>  { 
# 331
enum { __value = 1}; 
# 332
typedef __true_type __type; 
# 333
}; 
# 338
template< class _Tp> 
# 339
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> >  { 
# 341
}; 
# 346
template< class _Tp> 
# 347
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> >  { 
# 349
}; 
# 354
template< class _Tp> 
# 355
struct __is_char { 
# 357
enum { __value}; 
# 358
typedef __false_type __type; 
# 359
}; 
# 362
template<> struct __is_char< char>  { 
# 364
enum { __value = 1}; 
# 365
typedef __true_type __type; 
# 366
}; 
# 370
template<> struct __is_char< wchar_t>  { 
# 372
enum { __value = 1}; 
# 373
typedef __true_type __type; 
# 374
}; 
# 377
template< class _Tp> 
# 378
struct __is_byte { 
# 380
enum { __value}; 
# 381
typedef __false_type __type; 
# 382
}; 
# 385
template<> struct __is_byte< char>  { 
# 387
enum { __value = 1}; 
# 388
typedef __true_type __type; 
# 389
}; 
# 392
template<> struct __is_byte< signed char>  { 
# 394
enum { __value = 1}; 
# 395
typedef __true_type __type; 
# 396
}; 
# 399
template<> struct __is_byte< unsigned char>  { 
# 401
enum { __value = 1}; 
# 402
typedef __true_type __type; 
# 403
}; 
# 406
enum class byte: unsigned char; 
# 409
template<> struct __is_byte< byte>  { 
# 411
enum { __value = 1}; 
# 412
typedef __true_type __type; 
# 413
}; 
# 425 "/usr/include/c++/12/bits/cpp_type_traits.h" 3
template< class > struct iterator_traits; 
# 428
template< class _Tp> 
# 429
struct __is_nonvolatile_trivially_copyable { 
# 431
enum { __value = __is_trivially_copyable(_Tp)}; 
# 432
}; 
# 437
template< class _Tp> 
# 438
struct __is_nonvolatile_trivially_copyable< volatile _Tp>  { 
# 440
enum { __value}; 
# 441
}; 
# 444
template< class _OutputIter, class _InputIter> 
# 445
struct __memcpyable { 
# 447
enum { __value}; 
# 448
}; 
# 450
template< class _Tp> 
# 451
struct __memcpyable< _Tp *, _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 453
}; 
# 455
template< class _Tp> 
# 456
struct __memcpyable< _Tp *, const _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 458
}; 
# 465
template< class _Iter1, class _Iter2> 
# 466
struct __memcmpable { 
# 468
enum { __value}; 
# 469
}; 
# 472
template< class _Tp> 
# 473
struct __memcmpable< _Tp *, _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 475
}; 
# 477
template< class _Tp> 
# 478
struct __memcmpable< const _Tp *, _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 480
}; 
# 482
template< class _Tp> 
# 483
struct __memcmpable< _Tp *, const _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 485
}; 
# 493
template< class _Tp, bool _TreatAsBytes = __is_byte< _Tp> ::__value> 
# 500
struct __is_memcmp_ordered { 
# 502
static const bool __value = (((_Tp)(-1)) > ((_Tp)1)); 
# 503
}; 
# 505
template< class _Tp> 
# 506
struct __is_memcmp_ordered< _Tp, false>  { 
# 508
static const bool __value = false; 
# 509
}; 
# 512
template< class _Tp, class _Up, bool  = sizeof(_Tp) == sizeof(_Up)> 
# 513
struct __is_memcmp_ordered_with { 
# 515
static const bool __value = (__is_memcmp_ordered< _Tp> ::__value && __is_memcmp_ordered< _Up> ::__value); 
# 517
}; 
# 519
template< class _Tp, class _Up> 
# 520
struct __is_memcmp_ordered_with< _Tp, _Up, false>  { 
# 522
static const bool __value = false; 
# 523
}; 
# 535 "/usr/include/c++/12/bits/cpp_type_traits.h" 3
template<> struct __is_memcmp_ordered_with< byte, byte, true>  { 
# 536
static constexpr inline bool __value = true; }; 
# 538
template< class _Tp, bool _SameSize> 
# 539
struct __is_memcmp_ordered_with< _Tp, byte, _SameSize>  { 
# 540
static constexpr inline bool __value = false; }; 
# 542
template< class _Up, bool _SameSize> 
# 543
struct __is_memcmp_ordered_with< byte, _Up, _SameSize>  { 
# 544
static constexpr inline bool __value = false; }; 
# 550
template< class _Tp> 
# 551
struct __is_move_iterator { 
# 553
enum { __value}; 
# 554
typedef __false_type __type; 
# 555
}; 
# 559
template< class _Iterator> inline _Iterator 
# 562
__miter_base(_Iterator __it) 
# 563
{ return __it; } 
# 566
}
# 567
}
# 37 "/usr/include/c++/12/ext/type_traits.h" 3
extern "C++" {
# 39
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 44
template< bool , class > 
# 45
struct __enable_if { 
# 46
}; 
# 48
template< class _Tp> 
# 49
struct __enable_if< true, _Tp>  { 
# 50
typedef _Tp __type; }; 
# 54
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 55
struct __conditional_type { 
# 56
typedef _Iftrue __type; }; 
# 58
template< class _Iftrue, class _Iffalse> 
# 59
struct __conditional_type< false, _Iftrue, _Iffalse>  { 
# 60
typedef _Iffalse __type; }; 
# 64
template< class _Tp> 
# 65
struct __add_unsigned { 
# 68
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 71
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 72
}; 
# 75
template<> struct __add_unsigned< char>  { 
# 76
typedef unsigned char __type; }; 
# 79
template<> struct __add_unsigned< signed char>  { 
# 80
typedef unsigned char __type; }; 
# 83
template<> struct __add_unsigned< short>  { 
# 84
typedef unsigned short __type; }; 
# 87
template<> struct __add_unsigned< int>  { 
# 88
typedef unsigned __type; }; 
# 91
template<> struct __add_unsigned< long>  { 
# 92
typedef unsigned long __type; }; 
# 95
template<> struct __add_unsigned< long long>  { 
# 96
typedef unsigned long long __type; }; 
# 100
template<> struct __add_unsigned< bool> ; 
# 103
template<> struct __add_unsigned< wchar_t> ; 
# 107
template< class _Tp> 
# 108
struct __remove_unsigned { 
# 111
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 114
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 115
}; 
# 118
template<> struct __remove_unsigned< char>  { 
# 119
typedef signed char __type; }; 
# 122
template<> struct __remove_unsigned< unsigned char>  { 
# 123
typedef signed char __type; }; 
# 126
template<> struct __remove_unsigned< unsigned short>  { 
# 127
typedef short __type; }; 
# 130
template<> struct __remove_unsigned< unsigned>  { 
# 131
typedef int __type; }; 
# 134
template<> struct __remove_unsigned< unsigned long>  { 
# 135
typedef long __type; }; 
# 138
template<> struct __remove_unsigned< unsigned long long>  { 
# 139
typedef long long __type; }; 
# 143
template<> struct __remove_unsigned< bool> ; 
# 146
template<> struct __remove_unsigned< wchar_t> ; 
# 150
template< class _Type> constexpr bool 
# 153
__is_null_pointer(_Type *__ptr) 
# 154
{ return __ptr == 0; } 
# 156
template< class _Type> constexpr bool 
# 159
__is_null_pointer(_Type) 
# 160
{ return false; } 
# 164
constexpr bool __is_null_pointer(std::nullptr_t) 
# 165
{ return true; } 
# 170
template< class _Tp, bool  = std::template __is_integer< _Tp> ::__value> 
# 171
struct __promote { 
# 172
typedef double __type; }; 
# 177
template< class _Tp> 
# 178
struct __promote< _Tp, false>  { 
# 179
}; 
# 182
template<> struct __promote< long double>  { 
# 183
typedef long double __type; }; 
# 186
template<> struct __promote< double>  { 
# 187
typedef double __type; }; 
# 190
template<> struct __promote< float>  { 
# 191
typedef float __type; }; 
# 195
template< class ..._Tp> using __promoted_t = __decltype(((((typename __promote< _Tp> ::__type)0) + ... ))); 
# 200
template< class _Tp, class _Up> using __promote_2 = __promote< __promoted_t< _Tp, _Up> > ; 
# 203
template< class _Tp, class _Up, class _Vp> using __promote_3 = __promote< __promoted_t< _Tp, _Up, _Vp> > ; 
# 206
template< class _Tp, class _Up, class _Vp, class _Wp> using __promote_4 = __promote< __promoted_t< _Tp, _Up, _Vp, _Wp> > ; 
# 240 "/usr/include/c++/12/ext/type_traits.h" 3
}
# 241
}
# 34 "/usr/include/math.h" 3
extern "C" {
# 163 "/usr/include/math.h" 3
typedef float float_t; 
# 164
typedef double double_t; 
# 252 "/usr/include/math.h" 3
enum { 
# 253
FP_INT_UPWARD, 
# 256
FP_INT_DOWNWARD, 
# 259
FP_INT_TOWARDZERO, 
# 262
FP_INT_TONEARESTFROMZERO, 
# 265
FP_INT_TONEAREST
# 268
}; 
# 20 "/usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h" 3
extern int __fpclassify(double __value) noexcept(true)
# 21
 __attribute((const)); 
# 24
extern int __signbit(double __value) noexcept(true)
# 25
 __attribute((const)); 
# 29
extern int __isinf(double __value) noexcept(true)
# 30
 __attribute((const)); 
# 33
extern int __finite(double __value) noexcept(true)
# 34
 __attribute((const)); 
# 37
extern int __isnan(double __value) noexcept(true)
# 38
 __attribute((const)); 
# 41
extern int __iseqsig(double __x, double __y) noexcept(true); 
# 44
extern int __issignaling(double __value) noexcept(true)
# 45
 __attribute((const)); 
# 53 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern double acos(double __x) noexcept(true); extern double __acos(double __x) noexcept(true); 
# 55
extern double asin(double __x) noexcept(true); extern double __asin(double __x) noexcept(true); 
# 57
extern double atan(double __x) noexcept(true); extern double __atan(double __x) noexcept(true); 
# 59
extern double atan2(double __y, double __x) noexcept(true); extern double __atan2(double __y, double __x) noexcept(true); 
# 62
extern double cos(double __x) noexcept(true); extern double __cos(double __x) noexcept(true); 
# 64
extern double sin(double __x) noexcept(true); extern double __sin(double __x) noexcept(true); 
# 66
extern double tan(double __x) noexcept(true); extern double __tan(double __x) noexcept(true); 
# 71
extern double cosh(double __x) noexcept(true); extern double __cosh(double __x) noexcept(true); 
# 73
extern double sinh(double __x) noexcept(true); extern double __sinh(double __x) noexcept(true); 
# 75
extern double tanh(double __x) noexcept(true); extern double __tanh(double __x) noexcept(true); 
# 79
extern void sincos(double __x, double * __sinx, double * __cosx) noexcept(true); extern void __sincos(double __x, double * __sinx, double * __cosx) noexcept(true); 
# 85
extern double acosh(double __x) noexcept(true); extern double __acosh(double __x) noexcept(true); 
# 87
extern double asinh(double __x) noexcept(true); extern double __asinh(double __x) noexcept(true); 
# 89
extern double atanh(double __x) noexcept(true); extern double __atanh(double __x) noexcept(true); 
# 95
extern double exp(double __x) noexcept(true); extern double __exp(double __x) noexcept(true); 
# 98
extern double frexp(double __x, int * __exponent) noexcept(true); extern double __frexp(double __x, int * __exponent) noexcept(true); 
# 101
extern double ldexp(double __x, int __exponent) noexcept(true); extern double __ldexp(double __x, int __exponent) noexcept(true); 
# 104
extern double log(double __x) noexcept(true); extern double __log(double __x) noexcept(true); 
# 107
extern double log10(double __x) noexcept(true); extern double __log10(double __x) noexcept(true); 
# 110
extern double modf(double __x, double * __iptr) noexcept(true); extern double __modf(double __x, double * __iptr) noexcept(true) __attribute((__nonnull__(2))); 
# 114
extern double exp10(double __x) noexcept(true); extern double __exp10(double __x) noexcept(true); 
# 119
extern double expm1(double __x) noexcept(true); extern double __expm1(double __x) noexcept(true); 
# 122
extern double log1p(double __x) noexcept(true); extern double __log1p(double __x) noexcept(true); 
# 125
extern double logb(double __x) noexcept(true); extern double __logb(double __x) noexcept(true); 
# 130
extern double exp2(double __x) noexcept(true); extern double __exp2(double __x) noexcept(true); 
# 133
extern double log2(double __x) noexcept(true); extern double __log2(double __x) noexcept(true); 
# 140
extern double pow(double __x, double __y) noexcept(true); extern double __pow(double __x, double __y) noexcept(true); 
# 143
extern double sqrt(double __x) noexcept(true); extern double __sqrt(double __x) noexcept(true); 
# 147
extern double hypot(double __x, double __y) noexcept(true); extern double __hypot(double __x, double __y) noexcept(true); 
# 152
extern double cbrt(double __x) noexcept(true); extern double __cbrt(double __x) noexcept(true); 
# 159
extern double ceil(double __x) noexcept(true) __attribute((const)); extern double __ceil(double __x) noexcept(true) __attribute((const)); 
# 162
extern double fabs(double __x) noexcept(true) __attribute((const)); extern double __fabs(double __x) noexcept(true) __attribute((const)); 
# 165
extern double floor(double __x) noexcept(true) __attribute((const)); extern double __floor(double __x) noexcept(true) __attribute((const)); 
# 168
extern double fmod(double __x, double __y) noexcept(true); extern double __fmod(double __x, double __y) noexcept(true); 
# 183 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern int finite(double __value) noexcept(true)
# 184
 __attribute((const)); 
# 187
extern double drem(double __x, double __y) noexcept(true); extern double __drem(double __x, double __y) noexcept(true); 
# 191
extern double significand(double __x) noexcept(true); extern double __significand(double __x) noexcept(true); 
# 198
extern double copysign(double __x, double __y) noexcept(true) __attribute((const)); extern double __copysign(double __x, double __y) noexcept(true) __attribute((const)); 
# 203
extern double nan(const char * __tagb) noexcept(true); extern double __nan(const char * __tagb) noexcept(true); 
# 220 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern double j0(double) noexcept(true); extern double __j0(double) noexcept(true); 
# 221
extern double j1(double) noexcept(true); extern double __j1(double) noexcept(true); 
# 222
extern double jn(int, double) noexcept(true); extern double __jn(int, double) noexcept(true); 
# 223
extern double y0(double) noexcept(true); extern double __y0(double) noexcept(true); 
# 224
extern double y1(double) noexcept(true); extern double __y1(double) noexcept(true); 
# 225
extern double yn(int, double) noexcept(true); extern double __yn(int, double) noexcept(true); 
# 231
extern double erf(double) noexcept(true); extern double __erf(double) noexcept(true); 
# 232
extern double erfc(double) noexcept(true); extern double __erfc(double) noexcept(true); 
# 233
extern double lgamma(double) noexcept(true); extern double __lgamma(double) noexcept(true); 
# 238
extern double tgamma(double) noexcept(true); extern double __tgamma(double) noexcept(true); 
# 244
extern double gamma(double) noexcept(true); extern double __gamma(double) noexcept(true); 
# 252
extern double lgamma_r(double, int * __signgamp) noexcept(true); extern double __lgamma_r(double, int * __signgamp) noexcept(true); 
# 259
extern double rint(double __x) noexcept(true); extern double __rint(double __x) noexcept(true); 
# 262
extern double nextafter(double __x, double __y) noexcept(true); extern double __nextafter(double __x, double __y) noexcept(true); 
# 264
extern double nexttoward(double __x, long double __y) noexcept(true); extern double __nexttoward(double __x, long double __y) noexcept(true); 
# 269
extern double nextdown(double __x) noexcept(true); extern double __nextdown(double __x) noexcept(true); 
# 271
extern double nextup(double __x) noexcept(true); extern double __nextup(double __x) noexcept(true); 
# 275
extern double remainder(double __x, double __y) noexcept(true); extern double __remainder(double __x, double __y) noexcept(true); 
# 279
extern double scalbn(double __x, int __n) noexcept(true); extern double __scalbn(double __x, int __n) noexcept(true); 
# 283
extern int ilogb(double __x) noexcept(true); extern int __ilogb(double __x) noexcept(true); 
# 288
extern long llogb(double __x) noexcept(true); extern long __llogb(double __x) noexcept(true); 
# 293
extern double scalbln(double __x, long __n) noexcept(true); extern double __scalbln(double __x, long __n) noexcept(true); 
# 297
extern double nearbyint(double __x) noexcept(true); extern double __nearbyint(double __x) noexcept(true); 
# 301
extern double round(double __x) noexcept(true) __attribute((const)); extern double __round(double __x) noexcept(true) __attribute((const)); 
# 305
extern double trunc(double __x) noexcept(true) __attribute((const)); extern double __trunc(double __x) noexcept(true) __attribute((const)); 
# 310
extern double remquo(double __x, double __y, int * __quo) noexcept(true); extern double __remquo(double __x, double __y, int * __quo) noexcept(true); 
# 317
extern long lrint(double __x) noexcept(true); extern long __lrint(double __x) noexcept(true); 
# 319
__extension__ extern long long llrint(double __x) noexcept(true); extern long long __llrint(double __x) noexcept(true); 
# 323
extern long lround(double __x) noexcept(true); extern long __lround(double __x) noexcept(true); 
# 325
__extension__ extern long long llround(double __x) noexcept(true); extern long long __llround(double __x) noexcept(true); 
# 329
extern double fdim(double __x, double __y) noexcept(true); extern double __fdim(double __x, double __y) noexcept(true); 
# 333
extern double fmax(double __x, double __y) noexcept(true) __attribute((const)); extern double __fmax(double __x, double __y) noexcept(true) __attribute((const)); 
# 336
extern double fmin(double __x, double __y) noexcept(true) __attribute((const)); extern double __fmin(double __x, double __y) noexcept(true) __attribute((const)); 
# 340
extern double fma(double __x, double __y, double __z) noexcept(true); extern double __fma(double __x, double __y, double __z) noexcept(true); 
# 345
extern double roundeven(double __x) noexcept(true) __attribute((const)); extern double __roundeven(double __x) noexcept(true) __attribute((const)); 
# 349
extern __intmax_t fromfp(double __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfp(double __x, int __round, unsigned __width) noexcept(true); 
# 354
extern __uintmax_t ufromfp(double __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfp(double __x, int __round, unsigned __width) noexcept(true); 
# 360
extern __intmax_t fromfpx(double __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpx(double __x, int __round, unsigned __width) noexcept(true); 
# 366
extern __uintmax_t ufromfpx(double __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpx(double __x, int __round, unsigned __width) noexcept(true); 
# 370
extern int canonicalize(double * __cx, const double * __x) noexcept(true); 
# 377
extern double fmaxmag(double __x, double __y) noexcept(true) __attribute((const)); extern double __fmaxmag(double __x, double __y) noexcept(true) __attribute((const)); 
# 380
extern double fminmag(double __x, double __y) noexcept(true) __attribute((const)); extern double __fminmag(double __x, double __y) noexcept(true) __attribute((const)); 
# 385
extern double fmaximum(double __x, double __y) noexcept(true) __attribute((const)); extern double __fmaximum(double __x, double __y) noexcept(true) __attribute((const)); 
# 388
extern double fminimum(double __x, double __y) noexcept(true) __attribute((const)); extern double __fminimum(double __x, double __y) noexcept(true) __attribute((const)); 
# 391
extern double fmaximum_num(double __x, double __y) noexcept(true) __attribute((const)); extern double __fmaximum_num(double __x, double __y) noexcept(true) __attribute((const)); 
# 394
extern double fminimum_num(double __x, double __y) noexcept(true) __attribute((const)); extern double __fminimum_num(double __x, double __y) noexcept(true) __attribute((const)); 
# 397
extern double fmaximum_mag(double __x, double __y) noexcept(true) __attribute((const)); extern double __fmaximum_mag(double __x, double __y) noexcept(true) __attribute((const)); 
# 400
extern double fminimum_mag(double __x, double __y) noexcept(true) __attribute((const)); extern double __fminimum_mag(double __x, double __y) noexcept(true) __attribute((const)); 
# 403
extern double fmaximum_mag_num(double __x, double __y) noexcept(true) __attribute((const)); extern double __fmaximum_mag_num(double __x, double __y) noexcept(true) __attribute((const)); 
# 406
extern double fminimum_mag_num(double __x, double __y) noexcept(true) __attribute((const)); extern double __fminimum_mag_num(double __x, double __y) noexcept(true) __attribute((const)); 
# 411
extern int totalorder(const double * __x, const double * __y) noexcept(true)
# 413
 __attribute((__pure__)); 
# 416
extern int totalordermag(const double * __x, const double * __y) noexcept(true)
# 418
 __attribute((__pure__)); 
# 421
extern double getpayload(const double * __x) noexcept(true); extern double __getpayload(const double * __x) noexcept(true); 
# 424
extern int setpayload(double * __x, double __payload) noexcept(true); 
# 427
extern int setpayloadsig(double * __x, double __payload) noexcept(true); 
# 435
extern double scalb(double __x, double __n) noexcept(true); extern double __scalb(double __x, double __n) noexcept(true); 
# 20 "/usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h" 3
extern int __fpclassifyf(float __value) noexcept(true)
# 21
 __attribute((const)); 
# 24
extern int __signbitf(float __value) noexcept(true)
# 25
 __attribute((const)); 
# 29
extern int __isinff(float __value) noexcept(true)
# 30
 __attribute((const)); 
# 33
extern int __finitef(float __value) noexcept(true)
# 34
 __attribute((const)); 
# 37
extern int __isnanf(float __value) noexcept(true)
# 38
 __attribute((const)); 
# 41
extern int __iseqsigf(float __x, float __y) noexcept(true); 
# 44
extern int __issignalingf(float __value) noexcept(true)
# 45
 __attribute((const)); 
# 53 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern float acosf(float __x) noexcept(true); extern float __acosf(float __x) noexcept(true); 
# 55
extern float asinf(float __x) noexcept(true); extern float __asinf(float __x) noexcept(true); 
# 57
extern float atanf(float __x) noexcept(true); extern float __atanf(float __x) noexcept(true); 
# 59
extern float atan2f(float __y, float __x) noexcept(true); extern float __atan2f(float __y, float __x) noexcept(true); 
# 62
extern float cosf(float __x) noexcept(true); 
# 64
extern float sinf(float __x) noexcept(true); 
# 66
extern float tanf(float __x) noexcept(true); 
# 71
extern float coshf(float __x) noexcept(true); extern float __coshf(float __x) noexcept(true); 
# 73
extern float sinhf(float __x) noexcept(true); extern float __sinhf(float __x) noexcept(true); 
# 75
extern float tanhf(float __x) noexcept(true); extern float __tanhf(float __x) noexcept(true); 
# 79
extern void sincosf(float __x, float * __sinx, float * __cosx) noexcept(true); 
# 85
extern float acoshf(float __x) noexcept(true); extern float __acoshf(float __x) noexcept(true); 
# 87
extern float asinhf(float __x) noexcept(true); extern float __asinhf(float __x) noexcept(true); 
# 89
extern float atanhf(float __x) noexcept(true); extern float __atanhf(float __x) noexcept(true); 
# 95
extern float expf(float __x) noexcept(true); 
# 98
extern float frexpf(float __x, int * __exponent) noexcept(true); extern float __frexpf(float __x, int * __exponent) noexcept(true); 
# 101
extern float ldexpf(float __x, int __exponent) noexcept(true); extern float __ldexpf(float __x, int __exponent) noexcept(true); 
# 104
extern float logf(float __x) noexcept(true); 
# 107
extern float log10f(float __x) noexcept(true); 
# 110
extern float modff(float __x, float * __iptr) noexcept(true); extern float __modff(float __x, float * __iptr) noexcept(true) __attribute((__nonnull__(2))); 
# 114
extern float exp10f(float __x) noexcept(true); 
# 119
extern float expm1f(float __x) noexcept(true); extern float __expm1f(float __x) noexcept(true); 
# 122
extern float log1pf(float __x) noexcept(true); extern float __log1pf(float __x) noexcept(true); 
# 125
extern float logbf(float __x) noexcept(true); extern float __logbf(float __x) noexcept(true); 
# 130
extern float exp2f(float __x) noexcept(true); extern float __exp2f(float __x) noexcept(true); 
# 133
extern float log2f(float __x) noexcept(true); 
# 140
extern float powf(float __x, float __y) noexcept(true); 
# 143
extern float sqrtf(float __x) noexcept(true); extern float __sqrtf(float __x) noexcept(true); 
# 147
extern float hypotf(float __x, float __y) noexcept(true); extern float __hypotf(float __x, float __y) noexcept(true); 
# 152
extern float cbrtf(float __x) noexcept(true); extern float __cbrtf(float __x) noexcept(true); 
# 159
extern float ceilf(float __x) noexcept(true) __attribute((const)); extern float __ceilf(float __x) noexcept(true) __attribute((const)); 
# 162
extern float fabsf(float __x) noexcept(true) __attribute((const)); extern float __fabsf(float __x) noexcept(true) __attribute((const)); 
# 165
extern float floorf(float __x) noexcept(true) __attribute((const)); extern float __floorf(float __x) noexcept(true) __attribute((const)); 
# 168
extern float fmodf(float __x, float __y) noexcept(true); extern float __fmodf(float __x, float __y) noexcept(true); 
# 177 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern int isinff(float __value) noexcept(true)
# 178
 __attribute((const)); 
# 183
extern int finitef(float __value) noexcept(true)
# 184
 __attribute((const)); 
# 187
extern float dremf(float __x, float __y) noexcept(true); extern float __dremf(float __x, float __y) noexcept(true); 
# 191
extern float significandf(float __x) noexcept(true); extern float __significandf(float __x) noexcept(true); 
# 198
extern float copysignf(float __x, float __y) noexcept(true) __attribute((const)); extern float __copysignf(float __x, float __y) noexcept(true) __attribute((const)); 
# 203
extern float nanf(const char * __tagb) noexcept(true); extern float __nanf(const char * __tagb) noexcept(true); 
# 213 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern int isnanf(float __value) noexcept(true)
# 214
 __attribute((const)); 
# 220
extern float j0f(float) noexcept(true); extern float __j0f(float) noexcept(true); 
# 221
extern float j1f(float) noexcept(true); extern float __j1f(float) noexcept(true); 
# 222
extern float jnf(int, float) noexcept(true); extern float __jnf(int, float) noexcept(true); 
# 223
extern float y0f(float) noexcept(true); extern float __y0f(float) noexcept(true); 
# 224
extern float y1f(float) noexcept(true); extern float __y1f(float) noexcept(true); 
# 225
extern float ynf(int, float) noexcept(true); extern float __ynf(int, float) noexcept(true); 
# 231
extern float erff(float) noexcept(true); extern float __erff(float) noexcept(true); 
# 232
extern float erfcf(float) noexcept(true); extern float __erfcf(float) noexcept(true); 
# 233
extern float lgammaf(float) noexcept(true); extern float __lgammaf(float) noexcept(true); 
# 238
extern float tgammaf(float) noexcept(true); extern float __tgammaf(float) noexcept(true); 
# 244
extern float gammaf(float) noexcept(true); extern float __gammaf(float) noexcept(true); 
# 252
extern float lgammaf_r(float, int * __signgamp) noexcept(true); extern float __lgammaf_r(float, int * __signgamp) noexcept(true); 
# 259
extern float rintf(float __x) noexcept(true); extern float __rintf(float __x) noexcept(true); 
# 262
extern float nextafterf(float __x, float __y) noexcept(true); extern float __nextafterf(float __x, float __y) noexcept(true); 
# 264
extern float nexttowardf(float __x, long double __y) noexcept(true); extern float __nexttowardf(float __x, long double __y) noexcept(true); 
# 269
extern float nextdownf(float __x) noexcept(true); extern float __nextdownf(float __x) noexcept(true); 
# 271
extern float nextupf(float __x) noexcept(true); extern float __nextupf(float __x) noexcept(true); 
# 275
extern float remainderf(float __x, float __y) noexcept(true); extern float __remainderf(float __x, float __y) noexcept(true); 
# 279
extern float scalbnf(float __x, int __n) noexcept(true); extern float __scalbnf(float __x, int __n) noexcept(true); 
# 283
extern int ilogbf(float __x) noexcept(true); extern int __ilogbf(float __x) noexcept(true); 
# 288
extern long llogbf(float __x) noexcept(true); extern long __llogbf(float __x) noexcept(true); 
# 293
extern float scalblnf(float __x, long __n) noexcept(true); extern float __scalblnf(float __x, long __n) noexcept(true); 
# 297
extern float nearbyintf(float __x) noexcept(true); extern float __nearbyintf(float __x) noexcept(true); 
# 301
extern float roundf(float __x) noexcept(true) __attribute((const)); extern float __roundf(float __x) noexcept(true) __attribute((const)); 
# 305
extern float truncf(float __x) noexcept(true) __attribute((const)); extern float __truncf(float __x) noexcept(true) __attribute((const)); 
# 310
extern float remquof(float __x, float __y, int * __quo) noexcept(true); extern float __remquof(float __x, float __y, int * __quo) noexcept(true); 
# 317
extern long lrintf(float __x) noexcept(true); extern long __lrintf(float __x) noexcept(true); 
# 319
__extension__ extern long long llrintf(float __x) noexcept(true); extern long long __llrintf(float __x) noexcept(true); 
# 323
extern long lroundf(float __x) noexcept(true); extern long __lroundf(float __x) noexcept(true); 
# 325
__extension__ extern long long llroundf(float __x) noexcept(true); extern long long __llroundf(float __x) noexcept(true); 
# 329
extern float fdimf(float __x, float __y) noexcept(true); extern float __fdimf(float __x, float __y) noexcept(true); 
# 333
extern float fmaxf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fmaxf(float __x, float __y) noexcept(true) __attribute((const)); 
# 336
extern float fminf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fminf(float __x, float __y) noexcept(true) __attribute((const)); 
# 340
extern float fmaf(float __x, float __y, float __z) noexcept(true); extern float __fmaf(float __x, float __y, float __z) noexcept(true); 
# 345
extern float roundevenf(float __x) noexcept(true) __attribute((const)); extern float __roundevenf(float __x) noexcept(true) __attribute((const)); 
# 349
extern __intmax_t fromfpf(float __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpf(float __x, int __round, unsigned __width) noexcept(true); 
# 354
extern __uintmax_t ufromfpf(float __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpf(float __x, int __round, unsigned __width) noexcept(true); 
# 360
extern __intmax_t fromfpxf(float __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpxf(float __x, int __round, unsigned __width) noexcept(true); 
# 366
extern __uintmax_t ufromfpxf(float __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpxf(float __x, int __round, unsigned __width) noexcept(true); 
# 370
extern int canonicalizef(float * __cx, const float * __x) noexcept(true); 
# 377
extern float fmaxmagf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fmaxmagf(float __x, float __y) noexcept(true) __attribute((const)); 
# 380
extern float fminmagf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fminmagf(float __x, float __y) noexcept(true) __attribute((const)); 
# 385
extern float fmaximumf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fmaximumf(float __x, float __y) noexcept(true) __attribute((const)); 
# 388
extern float fminimumf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fminimumf(float __x, float __y) noexcept(true) __attribute((const)); 
# 391
extern float fmaximum_numf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fmaximum_numf(float __x, float __y) noexcept(true) __attribute((const)); 
# 394
extern float fminimum_numf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fminimum_numf(float __x, float __y) noexcept(true) __attribute((const)); 
# 397
extern float fmaximum_magf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fmaximum_magf(float __x, float __y) noexcept(true) __attribute((const)); 
# 400
extern float fminimum_magf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fminimum_magf(float __x, float __y) noexcept(true) __attribute((const)); 
# 403
extern float fmaximum_mag_numf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fmaximum_mag_numf(float __x, float __y) noexcept(true) __attribute((const)); 
# 406
extern float fminimum_mag_numf(float __x, float __y) noexcept(true) __attribute((const)); extern float __fminimum_mag_numf(float __x, float __y) noexcept(true) __attribute((const)); 
# 411
extern int totalorderf(const float * __x, const float * __y) noexcept(true)
# 413
 __attribute((__pure__)); 
# 416
extern int totalordermagf(const float * __x, const float * __y) noexcept(true)
# 418
 __attribute((__pure__)); 
# 421
extern float getpayloadf(const float * __x) noexcept(true); extern float __getpayloadf(const float * __x) noexcept(true); 
# 424
extern int setpayloadf(float * __x, float __payload) noexcept(true); 
# 427
extern int setpayloadsigf(float * __x, float __payload) noexcept(true); 
# 435
extern float scalbf(float __x, float __n) noexcept(true); extern float __scalbf(float __x, float __n) noexcept(true); 
# 20 "/usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h" 3
extern int __fpclassifyl(long double __value) noexcept(true)
# 21
 __attribute((const)); 
# 24
extern int __signbitl(long double __value) noexcept(true)
# 25
 __attribute((const)); 
# 29
extern int __isinfl(long double __value) noexcept(true)
# 30
 __attribute((const)); 
# 33
extern int __finitel(long double __value) noexcept(true)
# 34
 __attribute((const)); 
# 37
extern int __isnanl(long double __value) noexcept(true)
# 38
 __attribute((const)); 
# 41
extern int __iseqsigl(long double __x, long double __y) noexcept(true); 
# 44
extern int __issignalingl(long double __value) noexcept(true)
# 45
 __attribute((const)); 
# 53 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern long double acosl(long double __x) noexcept(true); extern long double __acosl(long double __x) noexcept(true); 
# 55
extern long double asinl(long double __x) noexcept(true); extern long double __asinl(long double __x) noexcept(true); 
# 57
extern long double atanl(long double __x) noexcept(true); extern long double __atanl(long double __x) noexcept(true); 
# 59
extern long double atan2l(long double __y, long double __x) noexcept(true); extern long double __atan2l(long double __y, long double __x) noexcept(true); 
# 62
extern long double cosl(long double __x) noexcept(true); extern long double __cosl(long double __x) noexcept(true); 
# 64
extern long double sinl(long double __x) noexcept(true); extern long double __sinl(long double __x) noexcept(true); 
# 66
extern long double tanl(long double __x) noexcept(true); extern long double __tanl(long double __x) noexcept(true); 
# 71
extern long double coshl(long double __x) noexcept(true); extern long double __coshl(long double __x) noexcept(true); 
# 73
extern long double sinhl(long double __x) noexcept(true); extern long double __sinhl(long double __x) noexcept(true); 
# 75
extern long double tanhl(long double __x) noexcept(true); extern long double __tanhl(long double __x) noexcept(true); 
# 79
extern void sincosl(long double __x, long double * __sinx, long double * __cosx) noexcept(true); extern void __sincosl(long double __x, long double * __sinx, long double * __cosx) noexcept(true); 
# 85
extern long double acoshl(long double __x) noexcept(true); extern long double __acoshl(long double __x) noexcept(true); 
# 87
extern long double asinhl(long double __x) noexcept(true); extern long double __asinhl(long double __x) noexcept(true); 
# 89
extern long double atanhl(long double __x) noexcept(true); extern long double __atanhl(long double __x) noexcept(true); 
# 95
extern long double expl(long double __x) noexcept(true); extern long double __expl(long double __x) noexcept(true); 
# 98
extern long double frexpl(long double __x, int * __exponent) noexcept(true); extern long double __frexpl(long double __x, int * __exponent) noexcept(true); 
# 101
extern long double ldexpl(long double __x, int __exponent) noexcept(true); extern long double __ldexpl(long double __x, int __exponent) noexcept(true); 
# 104
extern long double logl(long double __x) noexcept(true); extern long double __logl(long double __x) noexcept(true); 
# 107
extern long double log10l(long double __x) noexcept(true); extern long double __log10l(long double __x) noexcept(true); 
# 110
extern long double modfl(long double __x, long double * __iptr) noexcept(true); extern long double __modfl(long double __x, long double * __iptr) noexcept(true) __attribute((__nonnull__(2))); 
# 114
extern long double exp10l(long double __x) noexcept(true); extern long double __exp10l(long double __x) noexcept(true); 
# 119
extern long double expm1l(long double __x) noexcept(true); extern long double __expm1l(long double __x) noexcept(true); 
# 122
extern long double log1pl(long double __x) noexcept(true); extern long double __log1pl(long double __x) noexcept(true); 
# 125
extern long double logbl(long double __x) noexcept(true); extern long double __logbl(long double __x) noexcept(true); 
# 130
extern long double exp2l(long double __x) noexcept(true); extern long double __exp2l(long double __x) noexcept(true); 
# 133
extern long double log2l(long double __x) noexcept(true); extern long double __log2l(long double __x) noexcept(true); 
# 140
extern long double powl(long double __x, long double __y) noexcept(true); extern long double __powl(long double __x, long double __y) noexcept(true); 
# 143
extern long double sqrtl(long double __x) noexcept(true); extern long double __sqrtl(long double __x) noexcept(true); 
# 147
extern long double hypotl(long double __x, long double __y) noexcept(true); extern long double __hypotl(long double __x, long double __y) noexcept(true); 
# 152
extern long double cbrtl(long double __x) noexcept(true); extern long double __cbrtl(long double __x) noexcept(true); 
# 159
extern long double ceill(long double __x) noexcept(true) __attribute((const)); extern long double __ceill(long double __x) noexcept(true) __attribute((const)); 
# 162
extern long double fabsl(long double __x) noexcept(true) __attribute((const)); extern long double __fabsl(long double __x) noexcept(true) __attribute((const)); 
# 165
extern long double floorl(long double __x) noexcept(true) __attribute((const)); extern long double __floorl(long double __x) noexcept(true) __attribute((const)); 
# 168
extern long double fmodl(long double __x, long double __y) noexcept(true); extern long double __fmodl(long double __x, long double __y) noexcept(true); 
# 177 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern int isinfl(long double __value) noexcept(true)
# 178
 __attribute((const)); 
# 183
extern int finitel(long double __value) noexcept(true)
# 184
 __attribute((const)); 
# 187
extern long double dreml(long double __x, long double __y) noexcept(true); extern long double __dreml(long double __x, long double __y) noexcept(true); 
# 191
extern long double significandl(long double __x) noexcept(true); extern long double __significandl(long double __x) noexcept(true); 
# 198
extern long double copysignl(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __copysignl(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 203
extern long double nanl(const char * __tagb) noexcept(true); extern long double __nanl(const char * __tagb) noexcept(true); 
# 213 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern int isnanl(long double __value) noexcept(true)
# 214
 __attribute((const)); 
# 220
extern long double j0l(long double) noexcept(true); extern long double __j0l(long double) noexcept(true); 
# 221
extern long double j1l(long double) noexcept(true); extern long double __j1l(long double) noexcept(true); 
# 222
extern long double jnl(int, long double) noexcept(true); extern long double __jnl(int, long double) noexcept(true); 
# 223
extern long double y0l(long double) noexcept(true); extern long double __y0l(long double) noexcept(true); 
# 224
extern long double y1l(long double) noexcept(true); extern long double __y1l(long double) noexcept(true); 
# 225
extern long double ynl(int, long double) noexcept(true); extern long double __ynl(int, long double) noexcept(true); 
# 231
extern long double erfl(long double) noexcept(true); extern long double __erfl(long double) noexcept(true); 
# 232
extern long double erfcl(long double) noexcept(true); extern long double __erfcl(long double) noexcept(true); 
# 233
extern long double lgammal(long double) noexcept(true); extern long double __lgammal(long double) noexcept(true); 
# 238
extern long double tgammal(long double) noexcept(true); extern long double __tgammal(long double) noexcept(true); 
# 244
extern long double gammal(long double) noexcept(true); extern long double __gammal(long double) noexcept(true); 
# 252
extern long double lgammal_r(long double, int * __signgamp) noexcept(true); extern long double __lgammal_r(long double, int * __signgamp) noexcept(true); 
# 259
extern long double rintl(long double __x) noexcept(true); extern long double __rintl(long double __x) noexcept(true); 
# 262
extern long double nextafterl(long double __x, long double __y) noexcept(true); extern long double __nextafterl(long double __x, long double __y) noexcept(true); 
# 264
extern long double nexttowardl(long double __x, long double __y) noexcept(true); extern long double __nexttowardl(long double __x, long double __y) noexcept(true); 
# 269
extern long double nextdownl(long double __x) noexcept(true); extern long double __nextdownl(long double __x) noexcept(true); 
# 271
extern long double nextupl(long double __x) noexcept(true); extern long double __nextupl(long double __x) noexcept(true); 
# 275
extern long double remainderl(long double __x, long double __y) noexcept(true); extern long double __remainderl(long double __x, long double __y) noexcept(true); 
# 279
extern long double scalbnl(long double __x, int __n) noexcept(true); extern long double __scalbnl(long double __x, int __n) noexcept(true); 
# 283
extern int ilogbl(long double __x) noexcept(true); extern int __ilogbl(long double __x) noexcept(true); 
# 288
extern long llogbl(long double __x) noexcept(true); extern long __llogbl(long double __x) noexcept(true); 
# 293
extern long double scalblnl(long double __x, long __n) noexcept(true); extern long double __scalblnl(long double __x, long __n) noexcept(true); 
# 297
extern long double nearbyintl(long double __x) noexcept(true); extern long double __nearbyintl(long double __x) noexcept(true); 
# 301
extern long double roundl(long double __x) noexcept(true) __attribute((const)); extern long double __roundl(long double __x) noexcept(true) __attribute((const)); 
# 305
extern long double truncl(long double __x) noexcept(true) __attribute((const)); extern long double __truncl(long double __x) noexcept(true) __attribute((const)); 
# 310
extern long double remquol(long double __x, long double __y, int * __quo) noexcept(true); extern long double __remquol(long double __x, long double __y, int * __quo) noexcept(true); 
# 317
extern long lrintl(long double __x) noexcept(true); extern long __lrintl(long double __x) noexcept(true); 
# 319
__extension__ extern long long llrintl(long double __x) noexcept(true); extern long long __llrintl(long double __x) noexcept(true); 
# 323
extern long lroundl(long double __x) noexcept(true); extern long __lroundl(long double __x) noexcept(true); 
# 325
__extension__ extern long long llroundl(long double __x) noexcept(true); extern long long __llroundl(long double __x) noexcept(true); 
# 329
extern long double fdiml(long double __x, long double __y) noexcept(true); extern long double __fdiml(long double __x, long double __y) noexcept(true); 
# 333
extern long double fmaxl(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fmaxl(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 336
extern long double fminl(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fminl(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 340
extern long double fmal(long double __x, long double __y, long double __z) noexcept(true); extern long double __fmal(long double __x, long double __y, long double __z) noexcept(true); 
# 345
extern long double roundevenl(long double __x) noexcept(true) __attribute((const)); extern long double __roundevenl(long double __x) noexcept(true) __attribute((const)); 
# 349
extern __intmax_t fromfpl(long double __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpl(long double __x, int __round, unsigned __width) noexcept(true); 
# 354
extern __uintmax_t ufromfpl(long double __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpl(long double __x, int __round, unsigned __width) noexcept(true); 
# 360
extern __intmax_t fromfpxl(long double __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpxl(long double __x, int __round, unsigned __width) noexcept(true); 
# 366
extern __uintmax_t ufromfpxl(long double __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpxl(long double __x, int __round, unsigned __width) noexcept(true); 
# 370
extern int canonicalizel(long double * __cx, const long double * __x) noexcept(true); 
# 377
extern long double fmaxmagl(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fmaxmagl(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 380
extern long double fminmagl(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fminmagl(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 385
extern long double fmaximuml(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fmaximuml(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 388
extern long double fminimuml(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fminimuml(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 391
extern long double fmaximum_numl(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fmaximum_numl(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 394
extern long double fminimum_numl(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fminimum_numl(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 397
extern long double fmaximum_magl(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fmaximum_magl(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 400
extern long double fminimum_magl(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fminimum_magl(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 403
extern long double fmaximum_mag_numl(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fmaximum_mag_numl(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 406
extern long double fminimum_mag_numl(long double __x, long double __y) noexcept(true) __attribute((const)); extern long double __fminimum_mag_numl(long double __x, long double __y) noexcept(true) __attribute((const)); 
# 411
extern int totalorderl(const long double * __x, const long double * __y) noexcept(true)
# 413
 __attribute((__pure__)); 
# 416
extern int totalordermagl(const long double * __x, const long double * __y) noexcept(true)
# 418
 __attribute((__pure__)); 
# 421
extern long double getpayloadl(const long double * __x) noexcept(true); extern long double __getpayloadl(const long double * __x) noexcept(true); 
# 424
extern int setpayloadl(long double * __x, long double __payload) noexcept(true); 
# 427
extern int setpayloadsigl(long double * __x, long double __payload) noexcept(true); 
# 435
extern long double scalbl(long double __x, long double __n) noexcept(true); extern long double __scalbl(long double __x, long double __n) noexcept(true); 
# 53 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float32 acosf32(_Float32 __x) noexcept(true); extern _Float32 __acosf32(_Float32 __x) noexcept(true); 
# 55
extern _Float32 asinf32(_Float32 __x) noexcept(true); extern _Float32 __asinf32(_Float32 __x) noexcept(true); 
# 57
extern _Float32 atanf32(_Float32 __x) noexcept(true); extern _Float32 __atanf32(_Float32 __x) noexcept(true); 
# 59
extern _Float32 atan2f32(_Float32 __y, _Float32 __x) noexcept(true); extern _Float32 __atan2f32(_Float32 __y, _Float32 __x) noexcept(true); 
# 62
extern _Float32 cosf32(_Float32 __x) noexcept(true); extern _Float32 __cosf32(_Float32 __x) noexcept(true); 
# 64
extern _Float32 sinf32(_Float32 __x) noexcept(true); extern _Float32 __sinf32(_Float32 __x) noexcept(true); 
# 66
extern _Float32 tanf32(_Float32 __x) noexcept(true); extern _Float32 __tanf32(_Float32 __x) noexcept(true); 
# 71
extern _Float32 coshf32(_Float32 __x) noexcept(true); extern _Float32 __coshf32(_Float32 __x) noexcept(true); 
# 73
extern _Float32 sinhf32(_Float32 __x) noexcept(true); extern _Float32 __sinhf32(_Float32 __x) noexcept(true); 
# 75
extern _Float32 tanhf32(_Float32 __x) noexcept(true); extern _Float32 __tanhf32(_Float32 __x) noexcept(true); 
# 79
extern void sincosf32(_Float32 __x, _Float32 * __sinx, _Float32 * __cosx) noexcept(true); extern void __sincosf32(_Float32 __x, _Float32 * __sinx, _Float32 * __cosx) noexcept(true); 
# 85
extern _Float32 acoshf32(_Float32 __x) noexcept(true); extern _Float32 __acoshf32(_Float32 __x) noexcept(true); 
# 87
extern _Float32 asinhf32(_Float32 __x) noexcept(true); extern _Float32 __asinhf32(_Float32 __x) noexcept(true); 
# 89
extern _Float32 atanhf32(_Float32 __x) noexcept(true); extern _Float32 __atanhf32(_Float32 __x) noexcept(true); 
# 95
extern _Float32 expf32(_Float32 __x) noexcept(true); extern _Float32 __expf32(_Float32 __x) noexcept(true); 
# 98
extern _Float32 frexpf32(_Float32 __x, int * __exponent) noexcept(true); extern _Float32 __frexpf32(_Float32 __x, int * __exponent) noexcept(true); 
# 101
extern _Float32 ldexpf32(_Float32 __x, int __exponent) noexcept(true); extern _Float32 __ldexpf32(_Float32 __x, int __exponent) noexcept(true); 
# 104
extern _Float32 logf32(_Float32 __x) noexcept(true); extern _Float32 __logf32(_Float32 __x) noexcept(true); 
# 107
extern _Float32 log10f32(_Float32 __x) noexcept(true); extern _Float32 __log10f32(_Float32 __x) noexcept(true); 
# 110
extern _Float32 modff32(_Float32 __x, _Float32 * __iptr) noexcept(true); extern _Float32 __modff32(_Float32 __x, _Float32 * __iptr) noexcept(true) __attribute((__nonnull__(2))); 
# 114
extern _Float32 exp10f32(_Float32 __x) noexcept(true); extern _Float32 __exp10f32(_Float32 __x) noexcept(true); 
# 119
extern _Float32 expm1f32(_Float32 __x) noexcept(true); extern _Float32 __expm1f32(_Float32 __x) noexcept(true); 
# 122
extern _Float32 log1pf32(_Float32 __x) noexcept(true); extern _Float32 __log1pf32(_Float32 __x) noexcept(true); 
# 125
extern _Float32 logbf32(_Float32 __x) noexcept(true); extern _Float32 __logbf32(_Float32 __x) noexcept(true); 
# 130
extern _Float32 exp2f32(_Float32 __x) noexcept(true); extern _Float32 __exp2f32(_Float32 __x) noexcept(true); 
# 133
extern _Float32 log2f32(_Float32 __x) noexcept(true); extern _Float32 __log2f32(_Float32 __x) noexcept(true); 
# 140
extern _Float32 powf32(_Float32 __x, _Float32 __y) noexcept(true); extern _Float32 __powf32(_Float32 __x, _Float32 __y) noexcept(true); 
# 143
extern _Float32 sqrtf32(_Float32 __x) noexcept(true); extern _Float32 __sqrtf32(_Float32 __x) noexcept(true); 
# 147
extern _Float32 hypotf32(_Float32 __x, _Float32 __y) noexcept(true); extern _Float32 __hypotf32(_Float32 __x, _Float32 __y) noexcept(true); 
# 152
extern _Float32 cbrtf32(_Float32 __x) noexcept(true); extern _Float32 __cbrtf32(_Float32 __x) noexcept(true); 
# 159
extern _Float32 ceilf32(_Float32 __x) noexcept(true) __attribute((const)); extern _Float32 __ceilf32(_Float32 __x) noexcept(true) __attribute((const)); 
# 162
extern _Float32 fabsf32(_Float32 __x) noexcept(true) __attribute((const)); extern _Float32 __fabsf32(_Float32 __x) noexcept(true) __attribute((const)); 
# 165
extern _Float32 floorf32(_Float32 __x) noexcept(true) __attribute((const)); extern _Float32 __floorf32(_Float32 __x) noexcept(true) __attribute((const)); 
# 168
extern _Float32 fmodf32(_Float32 __x, _Float32 __y) noexcept(true); extern _Float32 __fmodf32(_Float32 __x, _Float32 __y) noexcept(true); 
# 198 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float32 copysignf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __copysignf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 203
extern _Float32 nanf32(const char * __tagb) noexcept(true); extern _Float32 __nanf32(const char * __tagb) noexcept(true); 
# 220 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float32 j0f32(_Float32) noexcept(true); extern _Float32 __j0f32(_Float32) noexcept(true); 
# 221
extern _Float32 j1f32(_Float32) noexcept(true); extern _Float32 __j1f32(_Float32) noexcept(true); 
# 222
extern _Float32 jnf32(int, _Float32) noexcept(true); extern _Float32 __jnf32(int, _Float32) noexcept(true); 
# 223
extern _Float32 y0f32(_Float32) noexcept(true); extern _Float32 __y0f32(_Float32) noexcept(true); 
# 224
extern _Float32 y1f32(_Float32) noexcept(true); extern _Float32 __y1f32(_Float32) noexcept(true); 
# 225
extern _Float32 ynf32(int, _Float32) noexcept(true); extern _Float32 __ynf32(int, _Float32) noexcept(true); 
# 231
extern _Float32 erff32(_Float32) noexcept(true); extern _Float32 __erff32(_Float32) noexcept(true); 
# 232
extern _Float32 erfcf32(_Float32) noexcept(true); extern _Float32 __erfcf32(_Float32) noexcept(true); 
# 233
extern _Float32 lgammaf32(_Float32) noexcept(true); extern _Float32 __lgammaf32(_Float32) noexcept(true); 
# 238
extern _Float32 tgammaf32(_Float32) noexcept(true); extern _Float32 __tgammaf32(_Float32) noexcept(true); 
# 252 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float32 lgammaf32_r(_Float32, int * __signgamp) noexcept(true); extern _Float32 __lgammaf32_r(_Float32, int * __signgamp) noexcept(true); 
# 259
extern _Float32 rintf32(_Float32 __x) noexcept(true); extern _Float32 __rintf32(_Float32 __x) noexcept(true); 
# 262
extern _Float32 nextafterf32(_Float32 __x, _Float32 __y) noexcept(true); extern _Float32 __nextafterf32(_Float32 __x, _Float32 __y) noexcept(true); 
# 269
extern _Float32 nextdownf32(_Float32 __x) noexcept(true); extern _Float32 __nextdownf32(_Float32 __x) noexcept(true); 
# 271
extern _Float32 nextupf32(_Float32 __x) noexcept(true); extern _Float32 __nextupf32(_Float32 __x) noexcept(true); 
# 275
extern _Float32 remainderf32(_Float32 __x, _Float32 __y) noexcept(true); extern _Float32 __remainderf32(_Float32 __x, _Float32 __y) noexcept(true); 
# 279
extern _Float32 scalbnf32(_Float32 __x, int __n) noexcept(true); extern _Float32 __scalbnf32(_Float32 __x, int __n) noexcept(true); 
# 283
extern int ilogbf32(_Float32 __x) noexcept(true); extern int __ilogbf32(_Float32 __x) noexcept(true); 
# 288
extern long llogbf32(_Float32 __x) noexcept(true); extern long __llogbf32(_Float32 __x) noexcept(true); 
# 293
extern _Float32 scalblnf32(_Float32 __x, long __n) noexcept(true); extern _Float32 __scalblnf32(_Float32 __x, long __n) noexcept(true); 
# 297
extern _Float32 nearbyintf32(_Float32 __x) noexcept(true); extern _Float32 __nearbyintf32(_Float32 __x) noexcept(true); 
# 301
extern _Float32 roundf32(_Float32 __x) noexcept(true) __attribute((const)); extern _Float32 __roundf32(_Float32 __x) noexcept(true) __attribute((const)); 
# 305
extern _Float32 truncf32(_Float32 __x) noexcept(true) __attribute((const)); extern _Float32 __truncf32(_Float32 __x) noexcept(true) __attribute((const)); 
# 310
extern _Float32 remquof32(_Float32 __x, _Float32 __y, int * __quo) noexcept(true); extern _Float32 __remquof32(_Float32 __x, _Float32 __y, int * __quo) noexcept(true); 
# 317
extern long lrintf32(_Float32 __x) noexcept(true); extern long __lrintf32(_Float32 __x) noexcept(true); 
# 319
__extension__ extern long long llrintf32(_Float32 __x) noexcept(true); extern long long __llrintf32(_Float32 __x) noexcept(true); 
# 323
extern long lroundf32(_Float32 __x) noexcept(true); extern long __lroundf32(_Float32 __x) noexcept(true); 
# 325
__extension__ extern long long llroundf32(_Float32 __x) noexcept(true); extern long long __llroundf32(_Float32 __x) noexcept(true); 
# 329
extern _Float32 fdimf32(_Float32 __x, _Float32 __y) noexcept(true); extern _Float32 __fdimf32(_Float32 __x, _Float32 __y) noexcept(true); 
# 333
extern _Float32 fmaxf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fmaxf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 336
extern _Float32 fminf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fminf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 340
extern _Float32 fmaf32(_Float32 __x, _Float32 __y, _Float32 __z) noexcept(true); extern _Float32 __fmaf32(_Float32 __x, _Float32 __y, _Float32 __z) noexcept(true); 
# 345
extern _Float32 roundevenf32(_Float32 __x) noexcept(true) __attribute((const)); extern _Float32 __roundevenf32(_Float32 __x) noexcept(true) __attribute((const)); 
# 349
extern __intmax_t fromfpf32(_Float32 __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpf32(_Float32 __x, int __round, unsigned __width) noexcept(true); 
# 354
extern __uintmax_t ufromfpf32(_Float32 __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpf32(_Float32 __x, int __round, unsigned __width) noexcept(true); 
# 360
extern __intmax_t fromfpxf32(_Float32 __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpxf32(_Float32 __x, int __round, unsigned __width) noexcept(true); 
# 366
extern __uintmax_t ufromfpxf32(_Float32 __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpxf32(_Float32 __x, int __round, unsigned __width) noexcept(true); 
# 370
extern int canonicalizef32(_Float32 * __cx, const _Float32 * __x) noexcept(true); 
# 377
extern _Float32 fmaxmagf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fmaxmagf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 380
extern _Float32 fminmagf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fminmagf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 385
extern _Float32 fmaximumf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fmaximumf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 388
extern _Float32 fminimumf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fminimumf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 391
extern _Float32 fmaximum_numf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fmaximum_numf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 394
extern _Float32 fminimum_numf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fminimum_numf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 397
extern _Float32 fmaximum_magf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fmaximum_magf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 400
extern _Float32 fminimum_magf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fminimum_magf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 403
extern _Float32 fmaximum_mag_numf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fmaximum_mag_numf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 406
extern _Float32 fminimum_mag_numf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); extern _Float32 __fminimum_mag_numf32(_Float32 __x, _Float32 __y) noexcept(true) __attribute((const)); 
# 411
extern int totalorderf32(const _Float32 * __x, const _Float32 * __y) noexcept(true)
# 413
 __attribute((__pure__)); 
# 416
extern int totalordermagf32(const _Float32 * __x, const _Float32 * __y) noexcept(true)
# 418
 __attribute((__pure__)); 
# 421
extern _Float32 getpayloadf32(const _Float32 * __x) noexcept(true); extern _Float32 __getpayloadf32(const _Float32 * __x) noexcept(true); 
# 424
extern int setpayloadf32(_Float32 * __x, _Float32 __payload) noexcept(true); 
# 427
extern int setpayloadsigf32(_Float32 * __x, _Float32 __payload) noexcept(true); 
# 53 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float64 acosf64(_Float64 __x) noexcept(true); extern _Float64 __acosf64(_Float64 __x) noexcept(true); 
# 55
extern _Float64 asinf64(_Float64 __x) noexcept(true); extern _Float64 __asinf64(_Float64 __x) noexcept(true); 
# 57
extern _Float64 atanf64(_Float64 __x) noexcept(true); extern _Float64 __atanf64(_Float64 __x) noexcept(true); 
# 59
extern _Float64 atan2f64(_Float64 __y, _Float64 __x) noexcept(true); extern _Float64 __atan2f64(_Float64 __y, _Float64 __x) noexcept(true); 
# 62
extern _Float64 cosf64(_Float64 __x) noexcept(true); extern _Float64 __cosf64(_Float64 __x) noexcept(true); 
# 64
extern _Float64 sinf64(_Float64 __x) noexcept(true); extern _Float64 __sinf64(_Float64 __x) noexcept(true); 
# 66
extern _Float64 tanf64(_Float64 __x) noexcept(true); extern _Float64 __tanf64(_Float64 __x) noexcept(true); 
# 71
extern _Float64 coshf64(_Float64 __x) noexcept(true); extern _Float64 __coshf64(_Float64 __x) noexcept(true); 
# 73
extern _Float64 sinhf64(_Float64 __x) noexcept(true); extern _Float64 __sinhf64(_Float64 __x) noexcept(true); 
# 75
extern _Float64 tanhf64(_Float64 __x) noexcept(true); extern _Float64 __tanhf64(_Float64 __x) noexcept(true); 
# 79
extern void sincosf64(_Float64 __x, _Float64 * __sinx, _Float64 * __cosx) noexcept(true); extern void __sincosf64(_Float64 __x, _Float64 * __sinx, _Float64 * __cosx) noexcept(true); 
# 85
extern _Float64 acoshf64(_Float64 __x) noexcept(true); extern _Float64 __acoshf64(_Float64 __x) noexcept(true); 
# 87
extern _Float64 asinhf64(_Float64 __x) noexcept(true); extern _Float64 __asinhf64(_Float64 __x) noexcept(true); 
# 89
extern _Float64 atanhf64(_Float64 __x) noexcept(true); extern _Float64 __atanhf64(_Float64 __x) noexcept(true); 
# 95
extern _Float64 expf64(_Float64 __x) noexcept(true); extern _Float64 __expf64(_Float64 __x) noexcept(true); 
# 98
extern _Float64 frexpf64(_Float64 __x, int * __exponent) noexcept(true); extern _Float64 __frexpf64(_Float64 __x, int * __exponent) noexcept(true); 
# 101
extern _Float64 ldexpf64(_Float64 __x, int __exponent) noexcept(true); extern _Float64 __ldexpf64(_Float64 __x, int __exponent) noexcept(true); 
# 104
extern _Float64 logf64(_Float64 __x) noexcept(true); extern _Float64 __logf64(_Float64 __x) noexcept(true); 
# 107
extern _Float64 log10f64(_Float64 __x) noexcept(true); extern _Float64 __log10f64(_Float64 __x) noexcept(true); 
# 110
extern _Float64 modff64(_Float64 __x, _Float64 * __iptr) noexcept(true); extern _Float64 __modff64(_Float64 __x, _Float64 * __iptr) noexcept(true) __attribute((__nonnull__(2))); 
# 114
extern _Float64 exp10f64(_Float64 __x) noexcept(true); extern _Float64 __exp10f64(_Float64 __x) noexcept(true); 
# 119
extern _Float64 expm1f64(_Float64 __x) noexcept(true); extern _Float64 __expm1f64(_Float64 __x) noexcept(true); 
# 122
extern _Float64 log1pf64(_Float64 __x) noexcept(true); extern _Float64 __log1pf64(_Float64 __x) noexcept(true); 
# 125
extern _Float64 logbf64(_Float64 __x) noexcept(true); extern _Float64 __logbf64(_Float64 __x) noexcept(true); 
# 130
extern _Float64 exp2f64(_Float64 __x) noexcept(true); extern _Float64 __exp2f64(_Float64 __x) noexcept(true); 
# 133
extern _Float64 log2f64(_Float64 __x) noexcept(true); extern _Float64 __log2f64(_Float64 __x) noexcept(true); 
# 140
extern _Float64 powf64(_Float64 __x, _Float64 __y) noexcept(true); extern _Float64 __powf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 143
extern _Float64 sqrtf64(_Float64 __x) noexcept(true); extern _Float64 __sqrtf64(_Float64 __x) noexcept(true); 
# 147
extern _Float64 hypotf64(_Float64 __x, _Float64 __y) noexcept(true); extern _Float64 __hypotf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 152
extern _Float64 cbrtf64(_Float64 __x) noexcept(true); extern _Float64 __cbrtf64(_Float64 __x) noexcept(true); 
# 159
extern _Float64 ceilf64(_Float64 __x) noexcept(true) __attribute((const)); extern _Float64 __ceilf64(_Float64 __x) noexcept(true) __attribute((const)); 
# 162
extern _Float64 fabsf64(_Float64 __x) noexcept(true) __attribute((const)); extern _Float64 __fabsf64(_Float64 __x) noexcept(true) __attribute((const)); 
# 165
extern _Float64 floorf64(_Float64 __x) noexcept(true) __attribute((const)); extern _Float64 __floorf64(_Float64 __x) noexcept(true) __attribute((const)); 
# 168
extern _Float64 fmodf64(_Float64 __x, _Float64 __y) noexcept(true); extern _Float64 __fmodf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 198 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float64 copysignf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __copysignf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 203
extern _Float64 nanf64(const char * __tagb) noexcept(true); extern _Float64 __nanf64(const char * __tagb) noexcept(true); 
# 220 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float64 j0f64(_Float64) noexcept(true); extern _Float64 __j0f64(_Float64) noexcept(true); 
# 221
extern _Float64 j1f64(_Float64) noexcept(true); extern _Float64 __j1f64(_Float64) noexcept(true); 
# 222
extern _Float64 jnf64(int, _Float64) noexcept(true); extern _Float64 __jnf64(int, _Float64) noexcept(true); 
# 223
extern _Float64 y0f64(_Float64) noexcept(true); extern _Float64 __y0f64(_Float64) noexcept(true); 
# 224
extern _Float64 y1f64(_Float64) noexcept(true); extern _Float64 __y1f64(_Float64) noexcept(true); 
# 225
extern _Float64 ynf64(int, _Float64) noexcept(true); extern _Float64 __ynf64(int, _Float64) noexcept(true); 
# 231
extern _Float64 erff64(_Float64) noexcept(true); extern _Float64 __erff64(_Float64) noexcept(true); 
# 232
extern _Float64 erfcf64(_Float64) noexcept(true); extern _Float64 __erfcf64(_Float64) noexcept(true); 
# 233
extern _Float64 lgammaf64(_Float64) noexcept(true); extern _Float64 __lgammaf64(_Float64) noexcept(true); 
# 238
extern _Float64 tgammaf64(_Float64) noexcept(true); extern _Float64 __tgammaf64(_Float64) noexcept(true); 
# 252 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float64 lgammaf64_r(_Float64, int * __signgamp) noexcept(true); extern _Float64 __lgammaf64_r(_Float64, int * __signgamp) noexcept(true); 
# 259
extern _Float64 rintf64(_Float64 __x) noexcept(true); extern _Float64 __rintf64(_Float64 __x) noexcept(true); 
# 262
extern _Float64 nextafterf64(_Float64 __x, _Float64 __y) noexcept(true); extern _Float64 __nextafterf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 269
extern _Float64 nextdownf64(_Float64 __x) noexcept(true); extern _Float64 __nextdownf64(_Float64 __x) noexcept(true); 
# 271
extern _Float64 nextupf64(_Float64 __x) noexcept(true); extern _Float64 __nextupf64(_Float64 __x) noexcept(true); 
# 275
extern _Float64 remainderf64(_Float64 __x, _Float64 __y) noexcept(true); extern _Float64 __remainderf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 279
extern _Float64 scalbnf64(_Float64 __x, int __n) noexcept(true); extern _Float64 __scalbnf64(_Float64 __x, int __n) noexcept(true); 
# 283
extern int ilogbf64(_Float64 __x) noexcept(true); extern int __ilogbf64(_Float64 __x) noexcept(true); 
# 288
extern long llogbf64(_Float64 __x) noexcept(true); extern long __llogbf64(_Float64 __x) noexcept(true); 
# 293
extern _Float64 scalblnf64(_Float64 __x, long __n) noexcept(true); extern _Float64 __scalblnf64(_Float64 __x, long __n) noexcept(true); 
# 297
extern _Float64 nearbyintf64(_Float64 __x) noexcept(true); extern _Float64 __nearbyintf64(_Float64 __x) noexcept(true); 
# 301
extern _Float64 roundf64(_Float64 __x) noexcept(true) __attribute((const)); extern _Float64 __roundf64(_Float64 __x) noexcept(true) __attribute((const)); 
# 305
extern _Float64 truncf64(_Float64 __x) noexcept(true) __attribute((const)); extern _Float64 __truncf64(_Float64 __x) noexcept(true) __attribute((const)); 
# 310
extern _Float64 remquof64(_Float64 __x, _Float64 __y, int * __quo) noexcept(true); extern _Float64 __remquof64(_Float64 __x, _Float64 __y, int * __quo) noexcept(true); 
# 317
extern long lrintf64(_Float64 __x) noexcept(true); extern long __lrintf64(_Float64 __x) noexcept(true); 
# 319
__extension__ extern long long llrintf64(_Float64 __x) noexcept(true); extern long long __llrintf64(_Float64 __x) noexcept(true); 
# 323
extern long lroundf64(_Float64 __x) noexcept(true); extern long __lroundf64(_Float64 __x) noexcept(true); 
# 325
__extension__ extern long long llroundf64(_Float64 __x) noexcept(true); extern long long __llroundf64(_Float64 __x) noexcept(true); 
# 329
extern _Float64 fdimf64(_Float64 __x, _Float64 __y) noexcept(true); extern _Float64 __fdimf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 333
extern _Float64 fmaxf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fmaxf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 336
extern _Float64 fminf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fminf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 340
extern _Float64 fmaf64(_Float64 __x, _Float64 __y, _Float64 __z) noexcept(true); extern _Float64 __fmaf64(_Float64 __x, _Float64 __y, _Float64 __z) noexcept(true); 
# 345
extern _Float64 roundevenf64(_Float64 __x) noexcept(true) __attribute((const)); extern _Float64 __roundevenf64(_Float64 __x) noexcept(true) __attribute((const)); 
# 349
extern __intmax_t fromfpf64(_Float64 __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpf64(_Float64 __x, int __round, unsigned __width) noexcept(true); 
# 354
extern __uintmax_t ufromfpf64(_Float64 __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpf64(_Float64 __x, int __round, unsigned __width) noexcept(true); 
# 360
extern __intmax_t fromfpxf64(_Float64 __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpxf64(_Float64 __x, int __round, unsigned __width) noexcept(true); 
# 366
extern __uintmax_t ufromfpxf64(_Float64 __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpxf64(_Float64 __x, int __round, unsigned __width) noexcept(true); 
# 370
extern int canonicalizef64(_Float64 * __cx, const _Float64 * __x) noexcept(true); 
# 377
extern _Float64 fmaxmagf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fmaxmagf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 380
extern _Float64 fminmagf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fminmagf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 385
extern _Float64 fmaximumf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fmaximumf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 388
extern _Float64 fminimumf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fminimumf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 391
extern _Float64 fmaximum_numf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fmaximum_numf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 394
extern _Float64 fminimum_numf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fminimum_numf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 397
extern _Float64 fmaximum_magf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fmaximum_magf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 400
extern _Float64 fminimum_magf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fminimum_magf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 403
extern _Float64 fmaximum_mag_numf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fmaximum_mag_numf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 406
extern _Float64 fminimum_mag_numf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); extern _Float64 __fminimum_mag_numf64(_Float64 __x, _Float64 __y) noexcept(true) __attribute((const)); 
# 411
extern int totalorderf64(const _Float64 * __x, const _Float64 * __y) noexcept(true)
# 413
 __attribute((__pure__)); 
# 416
extern int totalordermagf64(const _Float64 * __x, const _Float64 * __y) noexcept(true)
# 418
 __attribute((__pure__)); 
# 421
extern _Float64 getpayloadf64(const _Float64 * __x) noexcept(true); extern _Float64 __getpayloadf64(const _Float64 * __x) noexcept(true); 
# 424
extern int setpayloadf64(_Float64 * __x, _Float64 __payload) noexcept(true); 
# 427
extern int setpayloadsigf64(_Float64 * __x, _Float64 __payload) noexcept(true); 
# 20 "/usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h" 3
extern int __fpclassifyf128(_Float128 __value) noexcept(true)
# 21
 __attribute((const)); 
# 24
extern int __signbitf128(_Float128 __value) noexcept(true)
# 25
 __attribute((const)); 
# 29
extern int __isinff128(_Float128 __value) noexcept(true)
# 30
 __attribute((const)); 
# 33
extern int __finitef128(_Float128 __value) noexcept(true)
# 34
 __attribute((const)); 
# 37
extern int __isnanf128(_Float128 __value) noexcept(true)
# 38
 __attribute((const)); 
# 41
extern int __iseqsigf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 44
extern int __issignalingf128(_Float128 __value) noexcept(true)
# 45
 __attribute((const)); 
# 53 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float128 acosf128(_Float128 __x) noexcept(true); extern _Float128 __acosf128(_Float128 __x) noexcept(true); 
# 55
extern _Float128 asinf128(_Float128 __x) noexcept(true); extern _Float128 __asinf128(_Float128 __x) noexcept(true); 
# 57
extern _Float128 atanf128(_Float128 __x) noexcept(true); extern _Float128 __atanf128(_Float128 __x) noexcept(true); 
# 59
extern _Float128 atan2f128(_Float128 __y, _Float128 __x) noexcept(true); extern _Float128 __atan2f128(_Float128 __y, _Float128 __x) noexcept(true); 
# 62
extern _Float128 cosf128(_Float128 __x) noexcept(true); extern _Float128 __cosf128(_Float128 __x) noexcept(true); 
# 64
extern _Float128 sinf128(_Float128 __x) noexcept(true); extern _Float128 __sinf128(_Float128 __x) noexcept(true); 
# 66
extern _Float128 tanf128(_Float128 __x) noexcept(true); extern _Float128 __tanf128(_Float128 __x) noexcept(true); 
# 71
extern _Float128 coshf128(_Float128 __x) noexcept(true); extern _Float128 __coshf128(_Float128 __x) noexcept(true); 
# 73
extern _Float128 sinhf128(_Float128 __x) noexcept(true); extern _Float128 __sinhf128(_Float128 __x) noexcept(true); 
# 75
extern _Float128 tanhf128(_Float128 __x) noexcept(true); extern _Float128 __tanhf128(_Float128 __x) noexcept(true); 
# 79
extern void sincosf128(_Float128 __x, _Float128 * __sinx, _Float128 * __cosx) noexcept(true); extern void __sincosf128(_Float128 __x, _Float128 * __sinx, _Float128 * __cosx) noexcept(true); 
# 85
extern _Float128 acoshf128(_Float128 __x) noexcept(true); extern _Float128 __acoshf128(_Float128 __x) noexcept(true); 
# 87
extern _Float128 asinhf128(_Float128 __x) noexcept(true); extern _Float128 __asinhf128(_Float128 __x) noexcept(true); 
# 89
extern _Float128 atanhf128(_Float128 __x) noexcept(true); extern _Float128 __atanhf128(_Float128 __x) noexcept(true); 
# 95
extern _Float128 expf128(_Float128 __x) noexcept(true); extern _Float128 __expf128(_Float128 __x) noexcept(true); 
# 98
extern _Float128 frexpf128(_Float128 __x, int * __exponent) noexcept(true); extern _Float128 __frexpf128(_Float128 __x, int * __exponent) noexcept(true); 
# 101
extern _Float128 ldexpf128(_Float128 __x, int __exponent) noexcept(true); extern _Float128 __ldexpf128(_Float128 __x, int __exponent) noexcept(true); 
# 104
extern _Float128 logf128(_Float128 __x) noexcept(true); extern _Float128 __logf128(_Float128 __x) noexcept(true); 
# 107
extern _Float128 log10f128(_Float128 __x) noexcept(true); extern _Float128 __log10f128(_Float128 __x) noexcept(true); 
# 110
extern _Float128 modff128(_Float128 __x, _Float128 * __iptr) noexcept(true); extern _Float128 __modff128(_Float128 __x, _Float128 * __iptr) noexcept(true) __attribute((__nonnull__(2))); 
# 114
extern _Float128 exp10f128(_Float128 __x) noexcept(true); extern _Float128 __exp10f128(_Float128 __x) noexcept(true); 
# 119
extern _Float128 expm1f128(_Float128 __x) noexcept(true); extern _Float128 __expm1f128(_Float128 __x) noexcept(true); 
# 122
extern _Float128 log1pf128(_Float128 __x) noexcept(true); extern _Float128 __log1pf128(_Float128 __x) noexcept(true); 
# 125
extern _Float128 logbf128(_Float128 __x) noexcept(true); extern _Float128 __logbf128(_Float128 __x) noexcept(true); 
# 130
extern _Float128 exp2f128(_Float128 __x) noexcept(true); extern _Float128 __exp2f128(_Float128 __x) noexcept(true); 
# 133
extern _Float128 log2f128(_Float128 __x) noexcept(true); extern _Float128 __log2f128(_Float128 __x) noexcept(true); 
# 140
extern _Float128 powf128(_Float128 __x, _Float128 __y) noexcept(true); extern _Float128 __powf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 143
extern _Float128 sqrtf128(_Float128 __x) noexcept(true); extern _Float128 __sqrtf128(_Float128 __x) noexcept(true); 
# 147
extern _Float128 hypotf128(_Float128 __x, _Float128 __y) noexcept(true); extern _Float128 __hypotf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 152
extern _Float128 cbrtf128(_Float128 __x) noexcept(true); extern _Float128 __cbrtf128(_Float128 __x) noexcept(true); 
# 159
extern _Float128 ceilf128(_Float128 __x) noexcept(true) __attribute((const)); extern _Float128 __ceilf128(_Float128 __x) noexcept(true) __attribute((const)); 
# 162
extern _Float128 fabsf128(_Float128 __x) noexcept(true) __attribute((const)); extern _Float128 __fabsf128(_Float128 __x) noexcept(true) __attribute((const)); 
# 165
extern _Float128 floorf128(_Float128 __x) noexcept(true) __attribute((const)); extern _Float128 __floorf128(_Float128 __x) noexcept(true) __attribute((const)); 
# 168
extern _Float128 fmodf128(_Float128 __x, _Float128 __y) noexcept(true); extern _Float128 __fmodf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 198 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float128 copysignf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __copysignf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 203
extern _Float128 nanf128(const char * __tagb) noexcept(true); extern _Float128 __nanf128(const char * __tagb) noexcept(true); 
# 220 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float128 j0f128(_Float128) noexcept(true); extern _Float128 __j0f128(_Float128) noexcept(true); 
# 221
extern _Float128 j1f128(_Float128) noexcept(true); extern _Float128 __j1f128(_Float128) noexcept(true); 
# 222
extern _Float128 jnf128(int, _Float128) noexcept(true); extern _Float128 __jnf128(int, _Float128) noexcept(true); 
# 223
extern _Float128 y0f128(_Float128) noexcept(true); extern _Float128 __y0f128(_Float128) noexcept(true); 
# 224
extern _Float128 y1f128(_Float128) noexcept(true); extern _Float128 __y1f128(_Float128) noexcept(true); 
# 225
extern _Float128 ynf128(int, _Float128) noexcept(true); extern _Float128 __ynf128(int, _Float128) noexcept(true); 
# 231
extern _Float128 erff128(_Float128) noexcept(true); extern _Float128 __erff128(_Float128) noexcept(true); 
# 232
extern _Float128 erfcf128(_Float128) noexcept(true); extern _Float128 __erfcf128(_Float128) noexcept(true); 
# 233
extern _Float128 lgammaf128(_Float128) noexcept(true); extern _Float128 __lgammaf128(_Float128) noexcept(true); 
# 238
extern _Float128 tgammaf128(_Float128) noexcept(true); extern _Float128 __tgammaf128(_Float128) noexcept(true); 
# 252 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float128 lgammaf128_r(_Float128, int * __signgamp) noexcept(true); extern _Float128 __lgammaf128_r(_Float128, int * __signgamp) noexcept(true); 
# 259
extern _Float128 rintf128(_Float128 __x) noexcept(true); extern _Float128 __rintf128(_Float128 __x) noexcept(true); 
# 262
extern _Float128 nextafterf128(_Float128 __x, _Float128 __y) noexcept(true); extern _Float128 __nextafterf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 269
extern _Float128 nextdownf128(_Float128 __x) noexcept(true); extern _Float128 __nextdownf128(_Float128 __x) noexcept(true); 
# 271
extern _Float128 nextupf128(_Float128 __x) noexcept(true); extern _Float128 __nextupf128(_Float128 __x) noexcept(true); 
# 275
extern _Float128 remainderf128(_Float128 __x, _Float128 __y) noexcept(true); extern _Float128 __remainderf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 279
extern _Float128 scalbnf128(_Float128 __x, int __n) noexcept(true); extern _Float128 __scalbnf128(_Float128 __x, int __n) noexcept(true); 
# 283
extern int ilogbf128(_Float128 __x) noexcept(true); extern int __ilogbf128(_Float128 __x) noexcept(true); 
# 288
extern long llogbf128(_Float128 __x) noexcept(true); extern long __llogbf128(_Float128 __x) noexcept(true); 
# 293
extern _Float128 scalblnf128(_Float128 __x, long __n) noexcept(true); extern _Float128 __scalblnf128(_Float128 __x, long __n) noexcept(true); 
# 297
extern _Float128 nearbyintf128(_Float128 __x) noexcept(true); extern _Float128 __nearbyintf128(_Float128 __x) noexcept(true); 
# 301
extern _Float128 roundf128(_Float128 __x) noexcept(true) __attribute((const)); extern _Float128 __roundf128(_Float128 __x) noexcept(true) __attribute((const)); 
# 305
extern _Float128 truncf128(_Float128 __x) noexcept(true) __attribute((const)); extern _Float128 __truncf128(_Float128 __x) noexcept(true) __attribute((const)); 
# 310
extern _Float128 remquof128(_Float128 __x, _Float128 __y, int * __quo) noexcept(true); extern _Float128 __remquof128(_Float128 __x, _Float128 __y, int * __quo) noexcept(true); 
# 317
extern long lrintf128(_Float128 __x) noexcept(true); extern long __lrintf128(_Float128 __x) noexcept(true); 
# 319
__extension__ extern long long llrintf128(_Float128 __x) noexcept(true); extern long long __llrintf128(_Float128 __x) noexcept(true); 
# 323
extern long lroundf128(_Float128 __x) noexcept(true); extern long __lroundf128(_Float128 __x) noexcept(true); 
# 325
__extension__ extern long long llroundf128(_Float128 __x) noexcept(true); extern long long __llroundf128(_Float128 __x) noexcept(true); 
# 329
extern _Float128 fdimf128(_Float128 __x, _Float128 __y) noexcept(true); extern _Float128 __fdimf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 333
extern _Float128 fmaxf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fmaxf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 336
extern _Float128 fminf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fminf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 340
extern _Float128 fmaf128(_Float128 __x, _Float128 __y, _Float128 __z) noexcept(true); extern _Float128 __fmaf128(_Float128 __x, _Float128 __y, _Float128 __z) noexcept(true); 
# 345
extern _Float128 roundevenf128(_Float128 __x) noexcept(true) __attribute((const)); extern _Float128 __roundevenf128(_Float128 __x) noexcept(true) __attribute((const)); 
# 349
extern __intmax_t fromfpf128(_Float128 __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpf128(_Float128 __x, int __round, unsigned __width) noexcept(true); 
# 354
extern __uintmax_t ufromfpf128(_Float128 __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpf128(_Float128 __x, int __round, unsigned __width) noexcept(true); 
# 360
extern __intmax_t fromfpxf128(_Float128 __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpxf128(_Float128 __x, int __round, unsigned __width) noexcept(true); 
# 366
extern __uintmax_t ufromfpxf128(_Float128 __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpxf128(_Float128 __x, int __round, unsigned __width) noexcept(true); 
# 370
extern int canonicalizef128(_Float128 * __cx, const _Float128 * __x) noexcept(true); 
# 377
extern _Float128 fmaxmagf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fmaxmagf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 380
extern _Float128 fminmagf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fminmagf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 385
extern _Float128 fmaximumf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fmaximumf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 388
extern _Float128 fminimumf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fminimumf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 391
extern _Float128 fmaximum_numf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fmaximum_numf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 394
extern _Float128 fminimum_numf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fminimum_numf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 397
extern _Float128 fmaximum_magf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fmaximum_magf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 400
extern _Float128 fminimum_magf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fminimum_magf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 403
extern _Float128 fmaximum_mag_numf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fmaximum_mag_numf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 406
extern _Float128 fminimum_mag_numf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); extern _Float128 __fminimum_mag_numf128(_Float128 __x, _Float128 __y) noexcept(true) __attribute((const)); 
# 411
extern int totalorderf128(const _Float128 * __x, const _Float128 * __y) noexcept(true)
# 413
 __attribute((__pure__)); 
# 416
extern int totalordermagf128(const _Float128 * __x, const _Float128 * __y) noexcept(true)
# 418
 __attribute((__pure__)); 
# 421
extern _Float128 getpayloadf128(const _Float128 * __x) noexcept(true); extern _Float128 __getpayloadf128(const _Float128 * __x) noexcept(true); 
# 424
extern int setpayloadf128(_Float128 * __x, _Float128 __payload) noexcept(true); 
# 427
extern int setpayloadsigf128(_Float128 * __x, _Float128 __payload) noexcept(true); 
# 53 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float32x acosf32x(_Float32x __x) noexcept(true); extern _Float32x __acosf32x(_Float32x __x) noexcept(true); 
# 55
extern _Float32x asinf32x(_Float32x __x) noexcept(true); extern _Float32x __asinf32x(_Float32x __x) noexcept(true); 
# 57
extern _Float32x atanf32x(_Float32x __x) noexcept(true); extern _Float32x __atanf32x(_Float32x __x) noexcept(true); 
# 59
extern _Float32x atan2f32x(_Float32x __y, _Float32x __x) noexcept(true); extern _Float32x __atan2f32x(_Float32x __y, _Float32x __x) noexcept(true); 
# 62
extern _Float32x cosf32x(_Float32x __x) noexcept(true); extern _Float32x __cosf32x(_Float32x __x) noexcept(true); 
# 64
extern _Float32x sinf32x(_Float32x __x) noexcept(true); extern _Float32x __sinf32x(_Float32x __x) noexcept(true); 
# 66
extern _Float32x tanf32x(_Float32x __x) noexcept(true); extern _Float32x __tanf32x(_Float32x __x) noexcept(true); 
# 71
extern _Float32x coshf32x(_Float32x __x) noexcept(true); extern _Float32x __coshf32x(_Float32x __x) noexcept(true); 
# 73
extern _Float32x sinhf32x(_Float32x __x) noexcept(true); extern _Float32x __sinhf32x(_Float32x __x) noexcept(true); 
# 75
extern _Float32x tanhf32x(_Float32x __x) noexcept(true); extern _Float32x __tanhf32x(_Float32x __x) noexcept(true); 
# 79
extern void sincosf32x(_Float32x __x, _Float32x * __sinx, _Float32x * __cosx) noexcept(true); extern void __sincosf32x(_Float32x __x, _Float32x * __sinx, _Float32x * __cosx) noexcept(true); 
# 85
extern _Float32x acoshf32x(_Float32x __x) noexcept(true); extern _Float32x __acoshf32x(_Float32x __x) noexcept(true); 
# 87
extern _Float32x asinhf32x(_Float32x __x) noexcept(true); extern _Float32x __asinhf32x(_Float32x __x) noexcept(true); 
# 89
extern _Float32x atanhf32x(_Float32x __x) noexcept(true); extern _Float32x __atanhf32x(_Float32x __x) noexcept(true); 
# 95
extern _Float32x expf32x(_Float32x __x) noexcept(true); extern _Float32x __expf32x(_Float32x __x) noexcept(true); 
# 98
extern _Float32x frexpf32x(_Float32x __x, int * __exponent) noexcept(true); extern _Float32x __frexpf32x(_Float32x __x, int * __exponent) noexcept(true); 
# 101
extern _Float32x ldexpf32x(_Float32x __x, int __exponent) noexcept(true); extern _Float32x __ldexpf32x(_Float32x __x, int __exponent) noexcept(true); 
# 104
extern _Float32x logf32x(_Float32x __x) noexcept(true); extern _Float32x __logf32x(_Float32x __x) noexcept(true); 
# 107
extern _Float32x log10f32x(_Float32x __x) noexcept(true); extern _Float32x __log10f32x(_Float32x __x) noexcept(true); 
# 110
extern _Float32x modff32x(_Float32x __x, _Float32x * __iptr) noexcept(true); extern _Float32x __modff32x(_Float32x __x, _Float32x * __iptr) noexcept(true) __attribute((__nonnull__(2))); 
# 114
extern _Float32x exp10f32x(_Float32x __x) noexcept(true); extern _Float32x __exp10f32x(_Float32x __x) noexcept(true); 
# 119
extern _Float32x expm1f32x(_Float32x __x) noexcept(true); extern _Float32x __expm1f32x(_Float32x __x) noexcept(true); 
# 122
extern _Float32x log1pf32x(_Float32x __x) noexcept(true); extern _Float32x __log1pf32x(_Float32x __x) noexcept(true); 
# 125
extern _Float32x logbf32x(_Float32x __x) noexcept(true); extern _Float32x __logbf32x(_Float32x __x) noexcept(true); 
# 130
extern _Float32x exp2f32x(_Float32x __x) noexcept(true); extern _Float32x __exp2f32x(_Float32x __x) noexcept(true); 
# 133
extern _Float32x log2f32x(_Float32x __x) noexcept(true); extern _Float32x __log2f32x(_Float32x __x) noexcept(true); 
# 140
extern _Float32x powf32x(_Float32x __x, _Float32x __y) noexcept(true); extern _Float32x __powf32x(_Float32x __x, _Float32x __y) noexcept(true); 
# 143
extern _Float32x sqrtf32x(_Float32x __x) noexcept(true); extern _Float32x __sqrtf32x(_Float32x __x) noexcept(true); 
# 147
extern _Float32x hypotf32x(_Float32x __x, _Float32x __y) noexcept(true); extern _Float32x __hypotf32x(_Float32x __x, _Float32x __y) noexcept(true); 
# 152
extern _Float32x cbrtf32x(_Float32x __x) noexcept(true); extern _Float32x __cbrtf32x(_Float32x __x) noexcept(true); 
# 159
extern _Float32x ceilf32x(_Float32x __x) noexcept(true) __attribute((const)); extern _Float32x __ceilf32x(_Float32x __x) noexcept(true) __attribute((const)); 
# 162
extern _Float32x fabsf32x(_Float32x __x) noexcept(true) __attribute((const)); extern _Float32x __fabsf32x(_Float32x __x) noexcept(true) __attribute((const)); 
# 165
extern _Float32x floorf32x(_Float32x __x) noexcept(true) __attribute((const)); extern _Float32x __floorf32x(_Float32x __x) noexcept(true) __attribute((const)); 
# 168
extern _Float32x fmodf32x(_Float32x __x, _Float32x __y) noexcept(true); extern _Float32x __fmodf32x(_Float32x __x, _Float32x __y) noexcept(true); 
# 198 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float32x copysignf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __copysignf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 203
extern _Float32x nanf32x(const char * __tagb) noexcept(true); extern _Float32x __nanf32x(const char * __tagb) noexcept(true); 
# 220 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float32x j0f32x(_Float32x) noexcept(true); extern _Float32x __j0f32x(_Float32x) noexcept(true); 
# 221
extern _Float32x j1f32x(_Float32x) noexcept(true); extern _Float32x __j1f32x(_Float32x) noexcept(true); 
# 222
extern _Float32x jnf32x(int, _Float32x) noexcept(true); extern _Float32x __jnf32x(int, _Float32x) noexcept(true); 
# 223
extern _Float32x y0f32x(_Float32x) noexcept(true); extern _Float32x __y0f32x(_Float32x) noexcept(true); 
# 224
extern _Float32x y1f32x(_Float32x) noexcept(true); extern _Float32x __y1f32x(_Float32x) noexcept(true); 
# 225
extern _Float32x ynf32x(int, _Float32x) noexcept(true); extern _Float32x __ynf32x(int, _Float32x) noexcept(true); 
# 231
extern _Float32x erff32x(_Float32x) noexcept(true); extern _Float32x __erff32x(_Float32x) noexcept(true); 
# 232
extern _Float32x erfcf32x(_Float32x) noexcept(true); extern _Float32x __erfcf32x(_Float32x) noexcept(true); 
# 233
extern _Float32x lgammaf32x(_Float32x) noexcept(true); extern _Float32x __lgammaf32x(_Float32x) noexcept(true); 
# 238
extern _Float32x tgammaf32x(_Float32x) noexcept(true); extern _Float32x __tgammaf32x(_Float32x) noexcept(true); 
# 252 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float32x lgammaf32x_r(_Float32x, int * __signgamp) noexcept(true); extern _Float32x __lgammaf32x_r(_Float32x, int * __signgamp) noexcept(true); 
# 259
extern _Float32x rintf32x(_Float32x __x) noexcept(true); extern _Float32x __rintf32x(_Float32x __x) noexcept(true); 
# 262
extern _Float32x nextafterf32x(_Float32x __x, _Float32x __y) noexcept(true); extern _Float32x __nextafterf32x(_Float32x __x, _Float32x __y) noexcept(true); 
# 269
extern _Float32x nextdownf32x(_Float32x __x) noexcept(true); extern _Float32x __nextdownf32x(_Float32x __x) noexcept(true); 
# 271
extern _Float32x nextupf32x(_Float32x __x) noexcept(true); extern _Float32x __nextupf32x(_Float32x __x) noexcept(true); 
# 275
extern _Float32x remainderf32x(_Float32x __x, _Float32x __y) noexcept(true); extern _Float32x __remainderf32x(_Float32x __x, _Float32x __y) noexcept(true); 
# 279
extern _Float32x scalbnf32x(_Float32x __x, int __n) noexcept(true); extern _Float32x __scalbnf32x(_Float32x __x, int __n) noexcept(true); 
# 283
extern int ilogbf32x(_Float32x __x) noexcept(true); extern int __ilogbf32x(_Float32x __x) noexcept(true); 
# 288
extern long llogbf32x(_Float32x __x) noexcept(true); extern long __llogbf32x(_Float32x __x) noexcept(true); 
# 293
extern _Float32x scalblnf32x(_Float32x __x, long __n) noexcept(true); extern _Float32x __scalblnf32x(_Float32x __x, long __n) noexcept(true); 
# 297
extern _Float32x nearbyintf32x(_Float32x __x) noexcept(true); extern _Float32x __nearbyintf32x(_Float32x __x) noexcept(true); 
# 301
extern _Float32x roundf32x(_Float32x __x) noexcept(true) __attribute((const)); extern _Float32x __roundf32x(_Float32x __x) noexcept(true) __attribute((const)); 
# 305
extern _Float32x truncf32x(_Float32x __x) noexcept(true) __attribute((const)); extern _Float32x __truncf32x(_Float32x __x) noexcept(true) __attribute((const)); 
# 310
extern _Float32x remquof32x(_Float32x __x, _Float32x __y, int * __quo) noexcept(true); extern _Float32x __remquof32x(_Float32x __x, _Float32x __y, int * __quo) noexcept(true); 
# 317
extern long lrintf32x(_Float32x __x) noexcept(true); extern long __lrintf32x(_Float32x __x) noexcept(true); 
# 319
__extension__ extern long long llrintf32x(_Float32x __x) noexcept(true); extern long long __llrintf32x(_Float32x __x) noexcept(true); 
# 323
extern long lroundf32x(_Float32x __x) noexcept(true); extern long __lroundf32x(_Float32x __x) noexcept(true); 
# 325
__extension__ extern long long llroundf32x(_Float32x __x) noexcept(true); extern long long __llroundf32x(_Float32x __x) noexcept(true); 
# 329
extern _Float32x fdimf32x(_Float32x __x, _Float32x __y) noexcept(true); extern _Float32x __fdimf32x(_Float32x __x, _Float32x __y) noexcept(true); 
# 333
extern _Float32x fmaxf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fmaxf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 336
extern _Float32x fminf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fminf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 340
extern _Float32x fmaf32x(_Float32x __x, _Float32x __y, _Float32x __z) noexcept(true); extern _Float32x __fmaf32x(_Float32x __x, _Float32x __y, _Float32x __z) noexcept(true); 
# 345
extern _Float32x roundevenf32x(_Float32x __x) noexcept(true) __attribute((const)); extern _Float32x __roundevenf32x(_Float32x __x) noexcept(true) __attribute((const)); 
# 349
extern __intmax_t fromfpf32x(_Float32x __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpf32x(_Float32x __x, int __round, unsigned __width) noexcept(true); 
# 354
extern __uintmax_t ufromfpf32x(_Float32x __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpf32x(_Float32x __x, int __round, unsigned __width) noexcept(true); 
# 360
extern __intmax_t fromfpxf32x(_Float32x __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpxf32x(_Float32x __x, int __round, unsigned __width) noexcept(true); 
# 366
extern __uintmax_t ufromfpxf32x(_Float32x __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpxf32x(_Float32x __x, int __round, unsigned __width) noexcept(true); 
# 370
extern int canonicalizef32x(_Float32x * __cx, const _Float32x * __x) noexcept(true); 
# 377
extern _Float32x fmaxmagf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fmaxmagf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 380
extern _Float32x fminmagf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fminmagf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 385
extern _Float32x fmaximumf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fmaximumf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 388
extern _Float32x fminimumf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fminimumf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 391
extern _Float32x fmaximum_numf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fmaximum_numf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 394
extern _Float32x fminimum_numf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fminimum_numf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 397
extern _Float32x fmaximum_magf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fmaximum_magf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 400
extern _Float32x fminimum_magf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fminimum_magf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 403
extern _Float32x fmaximum_mag_numf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fmaximum_mag_numf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 406
extern _Float32x fminimum_mag_numf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); extern _Float32x __fminimum_mag_numf32x(_Float32x __x, _Float32x __y) noexcept(true) __attribute((const)); 
# 411
extern int totalorderf32x(const _Float32x * __x, const _Float32x * __y) noexcept(true)
# 413
 __attribute((__pure__)); 
# 416
extern int totalordermagf32x(const _Float32x * __x, const _Float32x * __y) noexcept(true)
# 418
 __attribute((__pure__)); 
# 421
extern _Float32x getpayloadf32x(const _Float32x * __x) noexcept(true); extern _Float32x __getpayloadf32x(const _Float32x * __x) noexcept(true); 
# 424
extern int setpayloadf32x(_Float32x * __x, _Float32x __payload) noexcept(true); 
# 427
extern int setpayloadsigf32x(_Float32x * __x, _Float32x __payload) noexcept(true); 
# 53 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float64x acosf64x(_Float64x __x) noexcept(true); extern _Float64x __acosf64x(_Float64x __x) noexcept(true); 
# 55
extern _Float64x asinf64x(_Float64x __x) noexcept(true); extern _Float64x __asinf64x(_Float64x __x) noexcept(true); 
# 57
extern _Float64x atanf64x(_Float64x __x) noexcept(true); extern _Float64x __atanf64x(_Float64x __x) noexcept(true); 
# 59
extern _Float64x atan2f64x(_Float64x __y, _Float64x __x) noexcept(true); extern _Float64x __atan2f64x(_Float64x __y, _Float64x __x) noexcept(true); 
# 62
extern _Float64x cosf64x(_Float64x __x) noexcept(true); extern _Float64x __cosf64x(_Float64x __x) noexcept(true); 
# 64
extern _Float64x sinf64x(_Float64x __x) noexcept(true); extern _Float64x __sinf64x(_Float64x __x) noexcept(true); 
# 66
extern _Float64x tanf64x(_Float64x __x) noexcept(true); extern _Float64x __tanf64x(_Float64x __x) noexcept(true); 
# 71
extern _Float64x coshf64x(_Float64x __x) noexcept(true); extern _Float64x __coshf64x(_Float64x __x) noexcept(true); 
# 73
extern _Float64x sinhf64x(_Float64x __x) noexcept(true); extern _Float64x __sinhf64x(_Float64x __x) noexcept(true); 
# 75
extern _Float64x tanhf64x(_Float64x __x) noexcept(true); extern _Float64x __tanhf64x(_Float64x __x) noexcept(true); 
# 79
extern void sincosf64x(_Float64x __x, _Float64x * __sinx, _Float64x * __cosx) noexcept(true); extern void __sincosf64x(_Float64x __x, _Float64x * __sinx, _Float64x * __cosx) noexcept(true); 
# 85
extern _Float64x acoshf64x(_Float64x __x) noexcept(true); extern _Float64x __acoshf64x(_Float64x __x) noexcept(true); 
# 87
extern _Float64x asinhf64x(_Float64x __x) noexcept(true); extern _Float64x __asinhf64x(_Float64x __x) noexcept(true); 
# 89
extern _Float64x atanhf64x(_Float64x __x) noexcept(true); extern _Float64x __atanhf64x(_Float64x __x) noexcept(true); 
# 95
extern _Float64x expf64x(_Float64x __x) noexcept(true); extern _Float64x __expf64x(_Float64x __x) noexcept(true); 
# 98
extern _Float64x frexpf64x(_Float64x __x, int * __exponent) noexcept(true); extern _Float64x __frexpf64x(_Float64x __x, int * __exponent) noexcept(true); 
# 101
extern _Float64x ldexpf64x(_Float64x __x, int __exponent) noexcept(true); extern _Float64x __ldexpf64x(_Float64x __x, int __exponent) noexcept(true); 
# 104
extern _Float64x logf64x(_Float64x __x) noexcept(true); extern _Float64x __logf64x(_Float64x __x) noexcept(true); 
# 107
extern _Float64x log10f64x(_Float64x __x) noexcept(true); extern _Float64x __log10f64x(_Float64x __x) noexcept(true); 
# 110
extern _Float64x modff64x(_Float64x __x, _Float64x * __iptr) noexcept(true); extern _Float64x __modff64x(_Float64x __x, _Float64x * __iptr) noexcept(true) __attribute((__nonnull__(2))); 
# 114
extern _Float64x exp10f64x(_Float64x __x) noexcept(true); extern _Float64x __exp10f64x(_Float64x __x) noexcept(true); 
# 119
extern _Float64x expm1f64x(_Float64x __x) noexcept(true); extern _Float64x __expm1f64x(_Float64x __x) noexcept(true); 
# 122
extern _Float64x log1pf64x(_Float64x __x) noexcept(true); extern _Float64x __log1pf64x(_Float64x __x) noexcept(true); 
# 125
extern _Float64x logbf64x(_Float64x __x) noexcept(true); extern _Float64x __logbf64x(_Float64x __x) noexcept(true); 
# 130
extern _Float64x exp2f64x(_Float64x __x) noexcept(true); extern _Float64x __exp2f64x(_Float64x __x) noexcept(true); 
# 133
extern _Float64x log2f64x(_Float64x __x) noexcept(true); extern _Float64x __log2f64x(_Float64x __x) noexcept(true); 
# 140
extern _Float64x powf64x(_Float64x __x, _Float64x __y) noexcept(true); extern _Float64x __powf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 143
extern _Float64x sqrtf64x(_Float64x __x) noexcept(true); extern _Float64x __sqrtf64x(_Float64x __x) noexcept(true); 
# 147
extern _Float64x hypotf64x(_Float64x __x, _Float64x __y) noexcept(true); extern _Float64x __hypotf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 152
extern _Float64x cbrtf64x(_Float64x __x) noexcept(true); extern _Float64x __cbrtf64x(_Float64x __x) noexcept(true); 
# 159
extern _Float64x ceilf64x(_Float64x __x) noexcept(true) __attribute((const)); extern _Float64x __ceilf64x(_Float64x __x) noexcept(true) __attribute((const)); 
# 162
extern _Float64x fabsf64x(_Float64x __x) noexcept(true) __attribute((const)); extern _Float64x __fabsf64x(_Float64x __x) noexcept(true) __attribute((const)); 
# 165
extern _Float64x floorf64x(_Float64x __x) noexcept(true) __attribute((const)); extern _Float64x __floorf64x(_Float64x __x) noexcept(true) __attribute((const)); 
# 168
extern _Float64x fmodf64x(_Float64x __x, _Float64x __y) noexcept(true); extern _Float64x __fmodf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 198 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float64x copysignf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __copysignf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 203
extern _Float64x nanf64x(const char * __tagb) noexcept(true); extern _Float64x __nanf64x(const char * __tagb) noexcept(true); 
# 220 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float64x j0f64x(_Float64x) noexcept(true); extern _Float64x __j0f64x(_Float64x) noexcept(true); 
# 221
extern _Float64x j1f64x(_Float64x) noexcept(true); extern _Float64x __j1f64x(_Float64x) noexcept(true); 
# 222
extern _Float64x jnf64x(int, _Float64x) noexcept(true); extern _Float64x __jnf64x(int, _Float64x) noexcept(true); 
# 223
extern _Float64x y0f64x(_Float64x) noexcept(true); extern _Float64x __y0f64x(_Float64x) noexcept(true); 
# 224
extern _Float64x y1f64x(_Float64x) noexcept(true); extern _Float64x __y1f64x(_Float64x) noexcept(true); 
# 225
extern _Float64x ynf64x(int, _Float64x) noexcept(true); extern _Float64x __ynf64x(int, _Float64x) noexcept(true); 
# 231
extern _Float64x erff64x(_Float64x) noexcept(true); extern _Float64x __erff64x(_Float64x) noexcept(true); 
# 232
extern _Float64x erfcf64x(_Float64x) noexcept(true); extern _Float64x __erfcf64x(_Float64x) noexcept(true); 
# 233
extern _Float64x lgammaf64x(_Float64x) noexcept(true); extern _Float64x __lgammaf64x(_Float64x) noexcept(true); 
# 238
extern _Float64x tgammaf64x(_Float64x) noexcept(true); extern _Float64x __tgammaf64x(_Float64x) noexcept(true); 
# 252 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern _Float64x lgammaf64x_r(_Float64x, int * __signgamp) noexcept(true); extern _Float64x __lgammaf64x_r(_Float64x, int * __signgamp) noexcept(true); 
# 259
extern _Float64x rintf64x(_Float64x __x) noexcept(true); extern _Float64x __rintf64x(_Float64x __x) noexcept(true); 
# 262
extern _Float64x nextafterf64x(_Float64x __x, _Float64x __y) noexcept(true); extern _Float64x __nextafterf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 269
extern _Float64x nextdownf64x(_Float64x __x) noexcept(true); extern _Float64x __nextdownf64x(_Float64x __x) noexcept(true); 
# 271
extern _Float64x nextupf64x(_Float64x __x) noexcept(true); extern _Float64x __nextupf64x(_Float64x __x) noexcept(true); 
# 275
extern _Float64x remainderf64x(_Float64x __x, _Float64x __y) noexcept(true); extern _Float64x __remainderf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 279
extern _Float64x scalbnf64x(_Float64x __x, int __n) noexcept(true); extern _Float64x __scalbnf64x(_Float64x __x, int __n) noexcept(true); 
# 283
extern int ilogbf64x(_Float64x __x) noexcept(true); extern int __ilogbf64x(_Float64x __x) noexcept(true); 
# 288
extern long llogbf64x(_Float64x __x) noexcept(true); extern long __llogbf64x(_Float64x __x) noexcept(true); 
# 293
extern _Float64x scalblnf64x(_Float64x __x, long __n) noexcept(true); extern _Float64x __scalblnf64x(_Float64x __x, long __n) noexcept(true); 
# 297
extern _Float64x nearbyintf64x(_Float64x __x) noexcept(true); extern _Float64x __nearbyintf64x(_Float64x __x) noexcept(true); 
# 301
extern _Float64x roundf64x(_Float64x __x) noexcept(true) __attribute((const)); extern _Float64x __roundf64x(_Float64x __x) noexcept(true) __attribute((const)); 
# 305
extern _Float64x truncf64x(_Float64x __x) noexcept(true) __attribute((const)); extern _Float64x __truncf64x(_Float64x __x) noexcept(true) __attribute((const)); 
# 310
extern _Float64x remquof64x(_Float64x __x, _Float64x __y, int * __quo) noexcept(true); extern _Float64x __remquof64x(_Float64x __x, _Float64x __y, int * __quo) noexcept(true); 
# 317
extern long lrintf64x(_Float64x __x) noexcept(true); extern long __lrintf64x(_Float64x __x) noexcept(true); 
# 319
__extension__ extern long long llrintf64x(_Float64x __x) noexcept(true); extern long long __llrintf64x(_Float64x __x) noexcept(true); 
# 323
extern long lroundf64x(_Float64x __x) noexcept(true); extern long __lroundf64x(_Float64x __x) noexcept(true); 
# 325
__extension__ extern long long llroundf64x(_Float64x __x) noexcept(true); extern long long __llroundf64x(_Float64x __x) noexcept(true); 
# 329
extern _Float64x fdimf64x(_Float64x __x, _Float64x __y) noexcept(true); extern _Float64x __fdimf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 333
extern _Float64x fmaxf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fmaxf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 336
extern _Float64x fminf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fminf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 340
extern _Float64x fmaf64x(_Float64x __x, _Float64x __y, _Float64x __z) noexcept(true); extern _Float64x __fmaf64x(_Float64x __x, _Float64x __y, _Float64x __z) noexcept(true); 
# 345
extern _Float64x roundevenf64x(_Float64x __x) noexcept(true) __attribute((const)); extern _Float64x __roundevenf64x(_Float64x __x) noexcept(true) __attribute((const)); 
# 349
extern __intmax_t fromfpf64x(_Float64x __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpf64x(_Float64x __x, int __round, unsigned __width) noexcept(true); 
# 354
extern __uintmax_t ufromfpf64x(_Float64x __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpf64x(_Float64x __x, int __round, unsigned __width) noexcept(true); 
# 360
extern __intmax_t fromfpxf64x(_Float64x __x, int __round, unsigned __width) noexcept(true); extern __intmax_t __fromfpxf64x(_Float64x __x, int __round, unsigned __width) noexcept(true); 
# 366
extern __uintmax_t ufromfpxf64x(_Float64x __x, int __round, unsigned __width) noexcept(true); extern __uintmax_t __ufromfpxf64x(_Float64x __x, int __round, unsigned __width) noexcept(true); 
# 370
extern int canonicalizef64x(_Float64x * __cx, const _Float64x * __x) noexcept(true); 
# 377
extern _Float64x fmaxmagf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fmaxmagf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 380
extern _Float64x fminmagf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fminmagf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 385
extern _Float64x fmaximumf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fmaximumf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 388
extern _Float64x fminimumf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fminimumf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 391
extern _Float64x fmaximum_numf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fmaximum_numf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 394
extern _Float64x fminimum_numf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fminimum_numf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 397
extern _Float64x fmaximum_magf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fmaximum_magf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 400
extern _Float64x fminimum_magf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fminimum_magf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 403
extern _Float64x fmaximum_mag_numf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fmaximum_mag_numf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 406
extern _Float64x fminimum_mag_numf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); extern _Float64x __fminimum_mag_numf64x(_Float64x __x, _Float64x __y) noexcept(true) __attribute((const)); 
# 411
extern int totalorderf64x(const _Float64x * __x, const _Float64x * __y) noexcept(true)
# 413
 __attribute((__pure__)); 
# 416
extern int totalordermagf64x(const _Float64x * __x, const _Float64x * __y) noexcept(true)
# 418
 __attribute((__pure__)); 
# 421
extern _Float64x getpayloadf64x(const _Float64x * __x) noexcept(true); extern _Float64x __getpayloadf64x(const _Float64x * __x) noexcept(true); 
# 424
extern int setpayloadf64x(_Float64x * __x, _Float64x __payload) noexcept(true); 
# 427
extern int setpayloadsigf64x(_Float64x * __x, _Float64x __payload) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern float fadd(double __x, double __y) noexcept(true); 
# 27
extern float fdiv(double __x, double __y) noexcept(true); 
# 30
extern float ffma(double __x, double __y, double __z) noexcept(true); 
# 33
extern float fmul(double __x, double __y) noexcept(true); 
# 36
extern float fsqrt(double __x) noexcept(true); 
# 39
extern float fsub(double __x, double __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern float faddl(long double __x, long double __y) noexcept(true); 
# 27
extern float fdivl(long double __x, long double __y) noexcept(true); 
# 30
extern float ffmal(long double __x, long double __y, long double __z) noexcept(true); 
# 33
extern float fmull(long double __x, long double __y) noexcept(true); 
# 36
extern float fsqrtl(long double __x) noexcept(true); 
# 39
extern float fsubl(long double __x, long double __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern double daddl(long double __x, long double __y) noexcept(true); 
# 27
extern double ddivl(long double __x, long double __y) noexcept(true); 
# 30
extern double dfmal(long double __x, long double __y, long double __z) noexcept(true); 
# 33
extern double dmull(long double __x, long double __y) noexcept(true); 
# 36
extern double dsqrtl(long double __x) noexcept(true); 
# 39
extern double dsubl(long double __x, long double __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern _Float32 f32addf32x(_Float32x __x, _Float32x __y) noexcept(true); 
# 27
extern _Float32 f32divf32x(_Float32x __x, _Float32x __y) noexcept(true); 
# 30
extern _Float32 f32fmaf32x(_Float32x __x, _Float32x __y, _Float32x __z) noexcept(true); 
# 33
extern _Float32 f32mulf32x(_Float32x __x, _Float32x __y) noexcept(true); 
# 36
extern _Float32 f32sqrtf32x(_Float32x __x) noexcept(true); 
# 39
extern _Float32 f32subf32x(_Float32x __x, _Float32x __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern _Float32 f32addf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 27
extern _Float32 f32divf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 30
extern _Float32 f32fmaf64(_Float64 __x, _Float64 __y, _Float64 __z) noexcept(true); 
# 33
extern _Float32 f32mulf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 36
extern _Float32 f32sqrtf64(_Float64 __x) noexcept(true); 
# 39
extern _Float32 f32subf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern _Float32 f32addf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 27
extern _Float32 f32divf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 30
extern _Float32 f32fmaf64x(_Float64x __x, _Float64x __y, _Float64x __z) noexcept(true); 
# 33
extern _Float32 f32mulf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 36
extern _Float32 f32sqrtf64x(_Float64x __x) noexcept(true); 
# 39
extern _Float32 f32subf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern _Float32 f32addf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 27
extern _Float32 f32divf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 30
extern _Float32 f32fmaf128(_Float128 __x, _Float128 __y, _Float128 __z) noexcept(true); 
# 33
extern _Float32 f32mulf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 36
extern _Float32 f32sqrtf128(_Float128 __x) noexcept(true); 
# 39
extern _Float32 f32subf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern _Float32x f32xaddf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 27
extern _Float32x f32xdivf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 30
extern _Float32x f32xfmaf64(_Float64 __x, _Float64 __y, _Float64 __z) noexcept(true); 
# 33
extern _Float32x f32xmulf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 36
extern _Float32x f32xsqrtf64(_Float64 __x) noexcept(true); 
# 39
extern _Float32x f32xsubf64(_Float64 __x, _Float64 __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern _Float32x f32xaddf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 27
extern _Float32x f32xdivf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 30
extern _Float32x f32xfmaf64x(_Float64x __x, _Float64x __y, _Float64x __z) noexcept(true); 
# 33
extern _Float32x f32xmulf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 36
extern _Float32x f32xsqrtf64x(_Float64x __x) noexcept(true); 
# 39
extern _Float32x f32xsubf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern _Float32x f32xaddf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 27
extern _Float32x f32xdivf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 30
extern _Float32x f32xfmaf128(_Float128 __x, _Float128 __y, _Float128 __z) noexcept(true); 
# 33
extern _Float32x f32xmulf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 36
extern _Float32x f32xsqrtf128(_Float128 __x) noexcept(true); 
# 39
extern _Float32x f32xsubf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern _Float64 f64addf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 27
extern _Float64 f64divf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 30
extern _Float64 f64fmaf64x(_Float64x __x, _Float64x __y, _Float64x __z) noexcept(true); 
# 33
extern _Float64 f64mulf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 36
extern _Float64 f64sqrtf64x(_Float64x __x) noexcept(true); 
# 39
extern _Float64 f64subf64x(_Float64x __x, _Float64x __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern _Float64 f64addf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 27
extern _Float64 f64divf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 30
extern _Float64 f64fmaf128(_Float128 __x, _Float128 __y, _Float128 __z) noexcept(true); 
# 33
extern _Float64 f64mulf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 36
extern _Float64 f64sqrtf128(_Float128 __x) noexcept(true); 
# 39
extern _Float64 f64subf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 24 "/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h" 3
extern _Float64x f64xaddf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 27
extern _Float64x f64xdivf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 30
extern _Float64x f64xfmaf128(_Float128 __x, _Float128 __y, _Float128 __z) noexcept(true); 
# 33
extern _Float64x f64xmulf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 36
extern _Float64x f64xsqrtf128(_Float128 __x) noexcept(true); 
# 39
extern _Float64x f64xsubf128(_Float128 __x, _Float128 __y) noexcept(true); 
# 854 "/usr/include/math.h" 3
extern int signgam; 
# 935 "/usr/include/math.h" 3
enum { 
# 936
FP_NAN, 
# 939
FP_INFINITE, 
# 942
FP_ZERO, 
# 945
FP_SUBNORMAL, 
# 948
FP_NORMAL
# 951
}; 
# 23 "/usr/include/x86_64-linux-gnu/bits/iscanonical.h" 3
extern int __iscanonicall(long double __x) noexcept(true)
# 24
 __attribute((const)); 
# 46 "/usr/include/x86_64-linux-gnu/bits/iscanonical.h" 3
extern "C++" {
# 47
inline int iscanonical(float __val) { return (((void)((__typeof__(__val))__val)), 1); } 
# 48
inline int iscanonical(double __val) { return (((void)((__typeof__(__val))__val)), 1); } 
# 49
inline int iscanonical(long double __val) { return __iscanonicall(__val); } 
# 51
inline int iscanonical(_Float128 __val) { return (((void)((__typeof__(__val))__val)), 1); } 
# 53
}
# 1067 "/usr/include/math.h" 3
extern "C++" {
# 1068
inline int issignaling(float __val) { return __issignalingf(__val); } 
# 1069
inline int issignaling(double __val) { return __issignaling(__val); } 
# 1071
inline int issignaling(long double __val) 
# 1072
{ 
# 1076
return __issignalingl(__val); 
# 1078
} 
# 1082
inline int issignaling(_Float128 __val) { return __issignalingf128(__val); } 
# 1084
}
# 1098 "/usr/include/math.h" 3
extern "C++" {
# 1129 "/usr/include/math.h" 3
template< class __T> inline bool 
# 1130
iszero(__T __val) 
# 1131
{ 
# 1132
return __val == 0; 
# 1133
} 
# 1135
}
# 1364 "/usr/include/math.h" 3
extern "C++" {
# 1365
template< class > struct __iseqsig_type; 
# 1367
template<> struct __iseqsig_type< float>  { 
# 1369
static int __call(float __x, float __y) throw() 
# 1370
{ 
# 1371
return __iseqsigf(__x, __y); 
# 1372
} 
# 1373
}; 
# 1375
template<> struct __iseqsig_type< double>  { 
# 1377
static int __call(double __x, double __y) throw() 
# 1378
{ 
# 1379
return __iseqsig(__x, __y); 
# 1380
} 
# 1381
}; 
# 1383
template<> struct __iseqsig_type< long double>  { 
# 1385
static int __call(long double __x, long double __y) throw() 
# 1386
{ 
# 1388
return __iseqsigl(__x, __y); 
# 1392
} 
# 1393
}; 
# 1418 "/usr/include/math.h" 3
template<> struct __iseqsig_type< __float128>  { 
# 1420
static int __call(_Float128 __x, _Float128 __y) throw() 
# 1421
{ 
# 1423
return __iseqsigf128(__x, __y); 
# 1427
} 
# 1428
}; 
# 1455 "/usr/include/math.h" 3
template< class _T1, class _T2> inline int 
# 1457
iseqsig(_T1 __x, _T2 __y) throw() 
# 1458
{ 
# 1460
typedef __decltype(((__x + __y) + (0.0F))) _T3; 
# 1464
return __iseqsig_type< __decltype(((__x + __y) + (0.0F)))> ::__call(__x, __y); 
# 1465
} 
# 1467
}
# 1472
}
# 77 "/usr/include/c++/12/cmath" 3
extern "C++" {
# 79
namespace std __attribute((__visibility__("default"))) { 
# 83
using ::acos;
# 87
constexpr float acos(float __x) 
# 88
{ return __builtin_acosf(__x); } 
# 91
constexpr long double acos(long double __x) 
# 92
{ return __builtin_acosl(__x); } 
# 95
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 99
acos(_Tp __x) 
# 100
{ return __builtin_acos(__x); } 
# 102
using ::asin;
# 106
constexpr float asin(float __x) 
# 107
{ return __builtin_asinf(__x); } 
# 110
constexpr long double asin(long double __x) 
# 111
{ return __builtin_asinl(__x); } 
# 114
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 118
asin(_Tp __x) 
# 119
{ return __builtin_asin(__x); } 
# 121
using ::atan;
# 125
constexpr float atan(float __x) 
# 126
{ return __builtin_atanf(__x); } 
# 129
constexpr long double atan(long double __x) 
# 130
{ return __builtin_atanl(__x); } 
# 133
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 137
atan(_Tp __x) 
# 138
{ return __builtin_atan(__x); } 
# 140
using ::atan2;
# 144
constexpr float atan2(float __y, float __x) 
# 145
{ return __builtin_atan2f(__y, __x); } 
# 148
constexpr long double atan2(long double __y, long double __x) 
# 149
{ return __builtin_atan2l(__y, __x); } 
# 152
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 155
atan2(_Tp __y, _Up __x) 
# 156
{ 
# 157
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 158
return atan2((__type)__y, (__type)__x); 
# 159
} 
# 161
using ::ceil;
# 165
constexpr float ceil(float __x) 
# 166
{ return __builtin_ceilf(__x); } 
# 169
constexpr long double ceil(long double __x) 
# 170
{ return __builtin_ceill(__x); } 
# 173
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 177
ceil(_Tp __x) 
# 178
{ return __builtin_ceil(__x); } 
# 180
using ::cos;
# 184
constexpr float cos(float __x) 
# 185
{ return __builtin_cosf(__x); } 
# 188
constexpr long double cos(long double __x) 
# 189
{ return __builtin_cosl(__x); } 
# 192
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 196
cos(_Tp __x) 
# 197
{ return __builtin_cos(__x); } 
# 199
using ::cosh;
# 203
constexpr float cosh(float __x) 
# 204
{ return __builtin_coshf(__x); } 
# 207
constexpr long double cosh(long double __x) 
# 208
{ return __builtin_coshl(__x); } 
# 211
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 215
cosh(_Tp __x) 
# 216
{ return __builtin_cosh(__x); } 
# 218
using ::exp;
# 222
constexpr float exp(float __x) 
# 223
{ return __builtin_expf(__x); } 
# 226
constexpr long double exp(long double __x) 
# 227
{ return __builtin_expl(__x); } 
# 230
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 234
exp(_Tp __x) 
# 235
{ return __builtin_exp(__x); } 
# 237
using ::fabs;
# 241
constexpr float fabs(float __x) 
# 242
{ return __builtin_fabsf(__x); } 
# 245
constexpr long double fabs(long double __x) 
# 246
{ return __builtin_fabsl(__x); } 
# 249
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 253
fabs(_Tp __x) 
# 254
{ return __builtin_fabs(__x); } 
# 256
using ::floor;
# 260
constexpr float floor(float __x) 
# 261
{ return __builtin_floorf(__x); } 
# 264
constexpr long double floor(long double __x) 
# 265
{ return __builtin_floorl(__x); } 
# 268
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 272
floor(_Tp __x) 
# 273
{ return __builtin_floor(__x); } 
# 275
using ::fmod;
# 279
constexpr float fmod(float __x, float __y) 
# 280
{ return __builtin_fmodf(__x, __y); } 
# 283
constexpr long double fmod(long double __x, long double __y) 
# 284
{ return __builtin_fmodl(__x, __y); } 
# 287
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 290
fmod(_Tp __x, _Up __y) 
# 291
{ 
# 292
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 293
return fmod((__type)__x, (__type)__y); 
# 294
} 
# 296
using ::frexp;
# 300
inline float frexp(float __x, int *__exp) 
# 301
{ return __builtin_frexpf(__x, __exp); } 
# 304
inline long double frexp(long double __x, int *__exp) 
# 305
{ return __builtin_frexpl(__x, __exp); } 
# 308
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 312
frexp(_Tp __x, int *__exp) 
# 313
{ return __builtin_frexp(__x, __exp); } 
# 315
using ::ldexp;
# 319
constexpr float ldexp(float __x, int __exp) 
# 320
{ return __builtin_ldexpf(__x, __exp); } 
# 323
constexpr long double ldexp(long double __x, int __exp) 
# 324
{ return __builtin_ldexpl(__x, __exp); } 
# 327
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 331
ldexp(_Tp __x, int __exp) 
# 332
{ return __builtin_ldexp(__x, __exp); } 
# 334
using ::log;
# 338
constexpr float log(float __x) 
# 339
{ return __builtin_logf(__x); } 
# 342
constexpr long double log(long double __x) 
# 343
{ return __builtin_logl(__x); } 
# 346
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 350
log(_Tp __x) 
# 351
{ return __builtin_log(__x); } 
# 353
using ::log10;
# 357
constexpr float log10(float __x) 
# 358
{ return __builtin_log10f(__x); } 
# 361
constexpr long double log10(long double __x) 
# 362
{ return __builtin_log10l(__x); } 
# 365
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 369
log10(_Tp __x) 
# 370
{ return __builtin_log10(__x); } 
# 372
using ::modf;
# 376
inline float modf(float __x, float *__iptr) 
# 377
{ return __builtin_modff(__x, __iptr); } 
# 380
inline long double modf(long double __x, long double *__iptr) 
# 381
{ return __builtin_modfl(__x, __iptr); } 
# 384
using ::pow;
# 388
constexpr float pow(float __x, float __y) 
# 389
{ return __builtin_powf(__x, __y); } 
# 392
constexpr long double pow(long double __x, long double __y) 
# 393
{ return __builtin_powl(__x, __y); } 
# 412 "/usr/include/c++/12/cmath" 3
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 415
pow(_Tp __x, _Up __y) 
# 416
{ 
# 417
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 418
return pow((__type)__x, (__type)__y); 
# 419
} 
# 421
using ::sin;
# 425
constexpr float sin(float __x) 
# 426
{ return __builtin_sinf(__x); } 
# 429
constexpr long double sin(long double __x) 
# 430
{ return __builtin_sinl(__x); } 
# 433
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 437
sin(_Tp __x) 
# 438
{ return __builtin_sin(__x); } 
# 440
using ::sinh;
# 444
constexpr float sinh(float __x) 
# 445
{ return __builtin_sinhf(__x); } 
# 448
constexpr long double sinh(long double __x) 
# 449
{ return __builtin_sinhl(__x); } 
# 452
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 456
sinh(_Tp __x) 
# 457
{ return __builtin_sinh(__x); } 
# 459
using ::sqrt;
# 463
constexpr float sqrt(float __x) 
# 464
{ return __builtin_sqrtf(__x); } 
# 467
constexpr long double sqrt(long double __x) 
# 468
{ return __builtin_sqrtl(__x); } 
# 471
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 475
sqrt(_Tp __x) 
# 476
{ return __builtin_sqrt(__x); } 
# 478
using ::tan;
# 482
constexpr float tan(float __x) 
# 483
{ return __builtin_tanf(__x); } 
# 486
constexpr long double tan(long double __x) 
# 487
{ return __builtin_tanl(__x); } 
# 490
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 494
tan(_Tp __x) 
# 495
{ return __builtin_tan(__x); } 
# 497
using ::tanh;
# 501
constexpr float tanh(float __x) 
# 502
{ return __builtin_tanhf(__x); } 
# 505
constexpr long double tanh(long double __x) 
# 506
{ return __builtin_tanhl(__x); } 
# 509
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 513
tanh(_Tp __x) 
# 514
{ return __builtin_tanh(__x); } 
# 537 "/usr/include/c++/12/cmath" 3
constexpr int fpclassify(float __x) 
# 538
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 539
} 
# 542
constexpr int fpclassify(double __x) 
# 543
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 544
} 
# 547
constexpr int fpclassify(long double __x) 
# 548
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 549
} 
# 553
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 556
fpclassify(_Tp __x) 
# 557
{ return (__x != 0) ? 4 : 2; } 
# 562
constexpr bool isfinite(float __x) 
# 563
{ return __builtin_isfinite(__x); } 
# 566
constexpr bool isfinite(double __x) 
# 567
{ return __builtin_isfinite(__x); } 
# 570
constexpr bool isfinite(long double __x) 
# 571
{ return __builtin_isfinite(__x); } 
# 575
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 578
isfinite(_Tp __x) 
# 579
{ return true; } 
# 584
constexpr bool isinf(float __x) 
# 585
{ return __builtin_isinf(__x); } 
# 592
constexpr bool isinf(double __x) 
# 593
{ return __builtin_isinf(__x); } 
# 597
constexpr bool isinf(long double __x) 
# 598
{ return __builtin_isinf(__x); } 
# 602
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 605
isinf(_Tp __x) 
# 606
{ return false; } 
# 611
constexpr bool isnan(float __x) 
# 612
{ return __builtin_isnan(__x); } 
# 619
constexpr bool isnan(double __x) 
# 620
{ return __builtin_isnan(__x); } 
# 624
constexpr bool isnan(long double __x) 
# 625
{ return __builtin_isnan(__x); } 
# 629
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 632
isnan(_Tp __x) 
# 633
{ return false; } 
# 638
constexpr bool isnormal(float __x) 
# 639
{ return __builtin_isnormal(__x); } 
# 642
constexpr bool isnormal(double __x) 
# 643
{ return __builtin_isnormal(__x); } 
# 646
constexpr bool isnormal(long double __x) 
# 647
{ return __builtin_isnormal(__x); } 
# 651
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 654
isnormal(_Tp __x) 
# 655
{ return (__x != 0) ? true : false; } 
# 661
constexpr bool signbit(float __x) 
# 662
{ return __builtin_signbit(__x); } 
# 665
constexpr bool signbit(double __x) 
# 666
{ return __builtin_signbit(__x); } 
# 669
constexpr bool signbit(long double __x) 
# 670
{ return __builtin_signbit(__x); } 
# 674
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 677
signbit(_Tp __x) 
# 678
{ return (__x < 0) ? true : false; } 
# 683
constexpr bool isgreater(float __x, float __y) 
# 684
{ return __builtin_isgreater(__x, __y); } 
# 687
constexpr bool isgreater(double __x, double __y) 
# 688
{ return __builtin_isgreater(__x, __y); } 
# 691
constexpr bool isgreater(long double __x, long double __y) 
# 692
{ return __builtin_isgreater(__x, __y); } 
# 696
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 700
isgreater(_Tp __x, _Up __y) 
# 701
{ 
# 702
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 703
return __builtin_isgreater((__type)__x, (__type)__y); 
# 704
} 
# 709
constexpr bool isgreaterequal(float __x, float __y) 
# 710
{ return __builtin_isgreaterequal(__x, __y); } 
# 713
constexpr bool isgreaterequal(double __x, double __y) 
# 714
{ return __builtin_isgreaterequal(__x, __y); } 
# 717
constexpr bool isgreaterequal(long double __x, long double __y) 
# 718
{ return __builtin_isgreaterequal(__x, __y); } 
# 722
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 726
isgreaterequal(_Tp __x, _Up __y) 
# 727
{ 
# 728
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 729
return __builtin_isgreaterequal((__type)__x, (__type)__y); 
# 730
} 
# 735
constexpr bool isless(float __x, float __y) 
# 736
{ return __builtin_isless(__x, __y); } 
# 739
constexpr bool isless(double __x, double __y) 
# 740
{ return __builtin_isless(__x, __y); } 
# 743
constexpr bool isless(long double __x, long double __y) 
# 744
{ return __builtin_isless(__x, __y); } 
# 748
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 752
isless(_Tp __x, _Up __y) 
# 753
{ 
# 754
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 755
return __builtin_isless((__type)__x, (__type)__y); 
# 756
} 
# 761
constexpr bool islessequal(float __x, float __y) 
# 762
{ return __builtin_islessequal(__x, __y); } 
# 765
constexpr bool islessequal(double __x, double __y) 
# 766
{ return __builtin_islessequal(__x, __y); } 
# 769
constexpr bool islessequal(long double __x, long double __y) 
# 770
{ return __builtin_islessequal(__x, __y); } 
# 774
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 778
islessequal(_Tp __x, _Up __y) 
# 779
{ 
# 780
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 781
return __builtin_islessequal((__type)__x, (__type)__y); 
# 782
} 
# 787
constexpr bool islessgreater(float __x, float __y) 
# 788
{ return __builtin_islessgreater(__x, __y); } 
# 791
constexpr bool islessgreater(double __x, double __y) 
# 792
{ return __builtin_islessgreater(__x, __y); } 
# 795
constexpr bool islessgreater(long double __x, long double __y) 
# 796
{ return __builtin_islessgreater(__x, __y); } 
# 800
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 804
islessgreater(_Tp __x, _Up __y) 
# 805
{ 
# 806
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 807
return __builtin_islessgreater((__type)__x, (__type)__y); 
# 808
} 
# 813
constexpr bool isunordered(float __x, float __y) 
# 814
{ return __builtin_isunordered(__x, __y); } 
# 817
constexpr bool isunordered(double __x, double __y) 
# 818
{ return __builtin_isunordered(__x, __y); } 
# 821
constexpr bool isunordered(long double __x, long double __y) 
# 822
{ return __builtin_isunordered(__x, __y); } 
# 826
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 830
isunordered(_Tp __x, _Up __y) 
# 831
{ 
# 832
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 833
return __builtin_isunordered((__type)__x, (__type)__y); 
# 834
} 
# 1065 "/usr/include/c++/12/cmath" 3
using ::double_t;
# 1066
using ::float_t;
# 1069
using ::acosh;
# 1070
using ::acoshf;
# 1071
using ::acoshl;
# 1073
using ::asinh;
# 1074
using ::asinhf;
# 1075
using ::asinhl;
# 1077
using ::atanh;
# 1078
using ::atanhf;
# 1079
using ::atanhl;
# 1081
using ::cbrt;
# 1082
using ::cbrtf;
# 1083
using ::cbrtl;
# 1085
using ::copysign;
# 1086
using ::copysignf;
# 1087
using ::copysignl;
# 1089
using ::erf;
# 1090
using ::erff;
# 1091
using ::erfl;
# 1093
using ::erfc;
# 1094
using ::erfcf;
# 1095
using ::erfcl;
# 1097
using ::exp2;
# 1098
using ::exp2f;
# 1099
using ::exp2l;
# 1101
using ::expm1;
# 1102
using ::expm1f;
# 1103
using ::expm1l;
# 1105
using ::fdim;
# 1106
using ::fdimf;
# 1107
using ::fdiml;
# 1109
using ::fma;
# 1110
using ::fmaf;
# 1111
using ::fmal;
# 1113
using ::fmax;
# 1114
using ::fmaxf;
# 1115
using ::fmaxl;
# 1117
using ::fmin;
# 1118
using ::fminf;
# 1119
using ::fminl;
# 1121
using ::hypot;
# 1122
using ::hypotf;
# 1123
using ::hypotl;
# 1125
using ::ilogb;
# 1126
using ::ilogbf;
# 1127
using ::ilogbl;
# 1129
using ::lgamma;
# 1130
using ::lgammaf;
# 1131
using ::lgammal;
# 1134
using ::llrint;
# 1135
using ::llrintf;
# 1136
using ::llrintl;
# 1138
using ::llround;
# 1139
using ::llroundf;
# 1140
using ::llroundl;
# 1143
using ::log1p;
# 1144
using ::log1pf;
# 1145
using ::log1pl;
# 1147
using ::log2;
# 1148
using ::log2f;
# 1149
using ::log2l;
# 1151
using ::logb;
# 1152
using ::logbf;
# 1153
using ::logbl;
# 1155
using ::lrint;
# 1156
using ::lrintf;
# 1157
using ::lrintl;
# 1159
using ::lround;
# 1160
using ::lroundf;
# 1161
using ::lroundl;
# 1163
using ::nan;
# 1164
using ::nanf;
# 1165
using ::nanl;
# 1167
using ::nearbyint;
# 1168
using ::nearbyintf;
# 1169
using ::nearbyintl;
# 1171
using ::nextafter;
# 1172
using ::nextafterf;
# 1173
using ::nextafterl;
# 1175
using ::nexttoward;
# 1176
using ::nexttowardf;
# 1177
using ::nexttowardl;
# 1179
using ::remainder;
# 1180
using ::remainderf;
# 1181
using ::remainderl;
# 1183
using ::remquo;
# 1184
using ::remquof;
# 1185
using ::remquol;
# 1187
using ::rint;
# 1188
using ::rintf;
# 1189
using ::rintl;
# 1191
using ::round;
# 1192
using ::roundf;
# 1193
using ::roundl;
# 1195
using ::scalbln;
# 1196
using ::scalblnf;
# 1197
using ::scalblnl;
# 1199
using ::scalbn;
# 1200
using ::scalbnf;
# 1201
using ::scalbnl;
# 1203
using ::tgamma;
# 1204
using ::tgammaf;
# 1205
using ::tgammal;
# 1207
using ::trunc;
# 1208
using ::truncf;
# 1209
using ::truncl;
# 1214
constexpr float acosh(float __x) 
# 1215
{ return __builtin_acoshf(__x); } 
# 1218
constexpr long double acosh(long double __x) 
# 1219
{ return __builtin_acoshl(__x); } 
# 1223
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1226
acosh(_Tp __x) 
# 1227
{ return __builtin_acosh(__x); } 
# 1232
constexpr float asinh(float __x) 
# 1233
{ return __builtin_asinhf(__x); } 
# 1236
constexpr long double asinh(long double __x) 
# 1237
{ return __builtin_asinhl(__x); } 
# 1241
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1244
asinh(_Tp __x) 
# 1245
{ return __builtin_asinh(__x); } 
# 1250
constexpr float atanh(float __x) 
# 1251
{ return __builtin_atanhf(__x); } 
# 1254
constexpr long double atanh(long double __x) 
# 1255
{ return __builtin_atanhl(__x); } 
# 1259
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1262
atanh(_Tp __x) 
# 1263
{ return __builtin_atanh(__x); } 
# 1268
constexpr float cbrt(float __x) 
# 1269
{ return __builtin_cbrtf(__x); } 
# 1272
constexpr long double cbrt(long double __x) 
# 1273
{ return __builtin_cbrtl(__x); } 
# 1277
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1280
cbrt(_Tp __x) 
# 1281
{ return __builtin_cbrt(__x); } 
# 1286
constexpr float copysign(float __x, float __y) 
# 1287
{ return __builtin_copysignf(__x, __y); } 
# 1290
constexpr long double copysign(long double __x, long double __y) 
# 1291
{ return __builtin_copysignl(__x, __y); } 
# 1295
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1297
copysign(_Tp __x, _Up __y) 
# 1298
{ 
# 1299
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1300
return copysign((__type)__x, (__type)__y); 
# 1301
} 
# 1306
constexpr float erf(float __x) 
# 1307
{ return __builtin_erff(__x); } 
# 1310
constexpr long double erf(long double __x) 
# 1311
{ return __builtin_erfl(__x); } 
# 1315
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1318
erf(_Tp __x) 
# 1319
{ return __builtin_erf(__x); } 
# 1324
constexpr float erfc(float __x) 
# 1325
{ return __builtin_erfcf(__x); } 
# 1328
constexpr long double erfc(long double __x) 
# 1329
{ return __builtin_erfcl(__x); } 
# 1333
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1336
erfc(_Tp __x) 
# 1337
{ return __builtin_erfc(__x); } 
# 1342
constexpr float exp2(float __x) 
# 1343
{ return __builtin_exp2f(__x); } 
# 1346
constexpr long double exp2(long double __x) 
# 1347
{ return __builtin_exp2l(__x); } 
# 1351
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1354
exp2(_Tp __x) 
# 1355
{ return __builtin_exp2(__x); } 
# 1360
constexpr float expm1(float __x) 
# 1361
{ return __builtin_expm1f(__x); } 
# 1364
constexpr long double expm1(long double __x) 
# 1365
{ return __builtin_expm1l(__x); } 
# 1369
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1372
expm1(_Tp __x) 
# 1373
{ return __builtin_expm1(__x); } 
# 1378
constexpr float fdim(float __x, float __y) 
# 1379
{ return __builtin_fdimf(__x, __y); } 
# 1382
constexpr long double fdim(long double __x, long double __y) 
# 1383
{ return __builtin_fdiml(__x, __y); } 
# 1387
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1389
fdim(_Tp __x, _Up __y) 
# 1390
{ 
# 1391
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1392
return fdim((__type)__x, (__type)__y); 
# 1393
} 
# 1398
constexpr float fma(float __x, float __y, float __z) 
# 1399
{ return __builtin_fmaf(__x, __y, __z); } 
# 1402
constexpr long double fma(long double __x, long double __y, long double __z) 
# 1403
{ return __builtin_fmal(__x, __y, __z); } 
# 1407
template< class _Tp, class _Up, class _Vp> constexpr typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type 
# 1409
fma(_Tp __x, _Up __y, _Vp __z) 
# 1410
{ 
# 1411
typedef typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type __type; 
# 1412
return fma((__type)__x, (__type)__y, (__type)__z); 
# 1413
} 
# 1418
constexpr float fmax(float __x, float __y) 
# 1419
{ return __builtin_fmaxf(__x, __y); } 
# 1422
constexpr long double fmax(long double __x, long double __y) 
# 1423
{ return __builtin_fmaxl(__x, __y); } 
# 1427
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1429
fmax(_Tp __x, _Up __y) 
# 1430
{ 
# 1431
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1432
return fmax((__type)__x, (__type)__y); 
# 1433
} 
# 1438
constexpr float fmin(float __x, float __y) 
# 1439
{ return __builtin_fminf(__x, __y); } 
# 1442
constexpr long double fmin(long double __x, long double __y) 
# 1443
{ return __builtin_fminl(__x, __y); } 
# 1447
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1449
fmin(_Tp __x, _Up __y) 
# 1450
{ 
# 1451
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1452
return fmin((__type)__x, (__type)__y); 
# 1453
} 
# 1458
constexpr float hypot(float __x, float __y) 
# 1459
{ return __builtin_hypotf(__x, __y); } 
# 1462
constexpr long double hypot(long double __x, long double __y) 
# 1463
{ return __builtin_hypotl(__x, __y); } 
# 1467
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1469
hypot(_Tp __x, _Up __y) 
# 1470
{ 
# 1471
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1472
return hypot((__type)__x, (__type)__y); 
# 1473
} 
# 1478
constexpr int ilogb(float __x) 
# 1479
{ return __builtin_ilogbf(__x); } 
# 1482
constexpr int ilogb(long double __x) 
# 1483
{ return __builtin_ilogbl(__x); } 
# 1487
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 1491
ilogb(_Tp __x) 
# 1492
{ return __builtin_ilogb(__x); } 
# 1497
constexpr float lgamma(float __x) 
# 1498
{ return __builtin_lgammaf(__x); } 
# 1501
constexpr long double lgamma(long double __x) 
# 1502
{ return __builtin_lgammal(__x); } 
# 1506
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1509
lgamma(_Tp __x) 
# 1510
{ return __builtin_lgamma(__x); } 
# 1515
constexpr long long llrint(float __x) 
# 1516
{ return __builtin_llrintf(__x); } 
# 1519
constexpr long long llrint(long double __x) 
# 1520
{ return __builtin_llrintl(__x); } 
# 1524
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 1527
llrint(_Tp __x) 
# 1528
{ return __builtin_llrint(__x); } 
# 1533
constexpr long long llround(float __x) 
# 1534
{ return __builtin_llroundf(__x); } 
# 1537
constexpr long long llround(long double __x) 
# 1538
{ return __builtin_llroundl(__x); } 
# 1542
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 1545
llround(_Tp __x) 
# 1546
{ return __builtin_llround(__x); } 
# 1551
constexpr float log1p(float __x) 
# 1552
{ return __builtin_log1pf(__x); } 
# 1555
constexpr long double log1p(long double __x) 
# 1556
{ return __builtin_log1pl(__x); } 
# 1560
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1563
log1p(_Tp __x) 
# 1564
{ return __builtin_log1p(__x); } 
# 1570
constexpr float log2(float __x) 
# 1571
{ return __builtin_log2f(__x); } 
# 1574
constexpr long double log2(long double __x) 
# 1575
{ return __builtin_log2l(__x); } 
# 1579
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1582
log2(_Tp __x) 
# 1583
{ return __builtin_log2(__x); } 
# 1588
constexpr float logb(float __x) 
# 1589
{ return __builtin_logbf(__x); } 
# 1592
constexpr long double logb(long double __x) 
# 1593
{ return __builtin_logbl(__x); } 
# 1597
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1600
logb(_Tp __x) 
# 1601
{ return __builtin_logb(__x); } 
# 1606
constexpr long lrint(float __x) 
# 1607
{ return __builtin_lrintf(__x); } 
# 1610
constexpr long lrint(long double __x) 
# 1611
{ return __builtin_lrintl(__x); } 
# 1615
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 1618
lrint(_Tp __x) 
# 1619
{ return __builtin_lrint(__x); } 
# 1624
constexpr long lround(float __x) 
# 1625
{ return __builtin_lroundf(__x); } 
# 1628
constexpr long lround(long double __x) 
# 1629
{ return __builtin_lroundl(__x); } 
# 1633
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 1636
lround(_Tp __x) 
# 1637
{ return __builtin_lround(__x); } 
# 1642
constexpr float nearbyint(float __x) 
# 1643
{ return __builtin_nearbyintf(__x); } 
# 1646
constexpr long double nearbyint(long double __x) 
# 1647
{ return __builtin_nearbyintl(__x); } 
# 1651
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1654
nearbyint(_Tp __x) 
# 1655
{ return __builtin_nearbyint(__x); } 
# 1660
constexpr float nextafter(float __x, float __y) 
# 1661
{ return __builtin_nextafterf(__x, __y); } 
# 1664
constexpr long double nextafter(long double __x, long double __y) 
# 1665
{ return __builtin_nextafterl(__x, __y); } 
# 1669
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1671
nextafter(_Tp __x, _Up __y) 
# 1672
{ 
# 1673
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1674
return nextafter((__type)__x, (__type)__y); 
# 1675
} 
# 1680
constexpr float nexttoward(float __x, long double __y) 
# 1681
{ return __builtin_nexttowardf(__x, __y); } 
# 1684
constexpr long double nexttoward(long double __x, long double __y) 
# 1685
{ return __builtin_nexttowardl(__x, __y); } 
# 1689
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1692
nexttoward(_Tp __x, long double __y) 
# 1693
{ return __builtin_nexttoward(__x, __y); } 
# 1698
constexpr float remainder(float __x, float __y) 
# 1699
{ return __builtin_remainderf(__x, __y); } 
# 1702
constexpr long double remainder(long double __x, long double __y) 
# 1703
{ return __builtin_remainderl(__x, __y); } 
# 1707
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1709
remainder(_Tp __x, _Up __y) 
# 1710
{ 
# 1711
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1712
return remainder((__type)__x, (__type)__y); 
# 1713
} 
# 1718
inline float remquo(float __x, float __y, int *__pquo) 
# 1719
{ return __builtin_remquof(__x, __y, __pquo); } 
# 1722
inline long double remquo(long double __x, long double __y, int *__pquo) 
# 1723
{ return __builtin_remquol(__x, __y, __pquo); } 
# 1727
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1729
remquo(_Tp __x, _Up __y, int *__pquo) 
# 1730
{ 
# 1731
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1732
return remquo((__type)__x, (__type)__y, __pquo); 
# 1733
} 
# 1738
constexpr float rint(float __x) 
# 1739
{ return __builtin_rintf(__x); } 
# 1742
constexpr long double rint(long double __x) 
# 1743
{ return __builtin_rintl(__x); } 
# 1747
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1750
rint(_Tp __x) 
# 1751
{ return __builtin_rint(__x); } 
# 1756
constexpr float round(float __x) 
# 1757
{ return __builtin_roundf(__x); } 
# 1760
constexpr long double round(long double __x) 
# 1761
{ return __builtin_roundl(__x); } 
# 1765
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1768
round(_Tp __x) 
# 1769
{ return __builtin_round(__x); } 
# 1774
constexpr float scalbln(float __x, long __ex) 
# 1775
{ return __builtin_scalblnf(__x, __ex); } 
# 1778
constexpr long double scalbln(long double __x, long __ex) 
# 1779
{ return __builtin_scalblnl(__x, __ex); } 
# 1783
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1786
scalbln(_Tp __x, long __ex) 
# 1787
{ return __builtin_scalbln(__x, __ex); } 
# 1792
constexpr float scalbn(float __x, int __ex) 
# 1793
{ return __builtin_scalbnf(__x, __ex); } 
# 1796
constexpr long double scalbn(long double __x, int __ex) 
# 1797
{ return __builtin_scalbnl(__x, __ex); } 
# 1801
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1804
scalbn(_Tp __x, int __ex) 
# 1805
{ return __builtin_scalbn(__x, __ex); } 
# 1810
constexpr float tgamma(float __x) 
# 1811
{ return __builtin_tgammaf(__x); } 
# 1814
constexpr long double tgamma(long double __x) 
# 1815
{ return __builtin_tgammal(__x); } 
# 1819
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1822
tgamma(_Tp __x) 
# 1823
{ return __builtin_tgamma(__x); } 
# 1828
constexpr float trunc(float __x) 
# 1829
{ return __builtin_truncf(__x); } 
# 1832
constexpr long double trunc(long double __x) 
# 1833
{ return __builtin_truncl(__x); } 
# 1837
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1840
trunc(_Tp __x) 
# 1841
{ return __builtin_trunc(__x); } 
# 1852 "/usr/include/c++/12/cmath" 3
template< class _Tp> inline _Tp 
# 1854
__hypot3(_Tp __x, _Tp __y, _Tp __z) 
# 1855
{ 
# 1856
__x = std::abs(__x); 
# 1857
__y = std::abs(__y); 
# 1858
__z = std::abs(__z); 
# 1859
if (_Tp __a = (__x < __y) ? (__y < __z) ? __z : __y : ((__x < __z) ? __z : __x)) { 
# 1860
return __a * std::sqrt((((__x / __a) * (__x / __a)) + ((__y / __a) * (__y / __a))) + ((__z / __a) * (__z / __a))); } else { 
# 1864
return {}; }  
# 1865
} 
# 1868
inline float hypot(float __x, float __y, float __z) 
# 1869
{ return std::__hypot3< float> (__x, __y, __z); } 
# 1872
inline double hypot(double __x, double __y, double __z) 
# 1873
{ return std::__hypot3< double> (__x, __y, __z); } 
# 1876
inline long double hypot(long double __x, long double __y, long double __z) 
# 1877
{ return std::__hypot3< long double> (__x, __y, __z); } 
# 1879
template< class _Tp, class _Up, class _Vp> __gnu_cxx::__promoted_t< _Tp, _Up, _Vp>  
# 1881
hypot(_Tp __x, _Up __y, _Vp __z) 
# 1882
{ 
# 1883
using __type = __gnu_cxx::__promoted_t< _Tp, _Up, _Vp> ; 
# 1884
return std::__hypot3< __gnu_cxx::__promoted_t< _Tp, _Up, _Vp> > (__x, __y, __z); 
# 1885
} 
# 1932 "/usr/include/c++/12/cmath" 3
}
# 33 "/usr/include/c++/12/bits/specfun.h" 3
#pragma GCC visibility push ( default )
# 42 "/usr/include/c++/12/bits/functexcept.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 48
void __throw_bad_exception() __attribute((__noreturn__)); 
# 52
void __throw_bad_alloc() __attribute((__noreturn__)); 
# 55
void __throw_bad_array_new_length() __attribute((__noreturn__)); 
# 59
void __throw_bad_cast() __attribute((__noreturn__)); 
# 62
void __throw_bad_typeid() __attribute((__noreturn__)); 
# 66
void __throw_logic_error(const char *) __attribute((__noreturn__)); 
# 69
void __throw_domain_error(const char *) __attribute((__noreturn__)); 
# 72
void __throw_invalid_argument(const char *) __attribute((__noreturn__)); 
# 75
void __throw_length_error(const char *) __attribute((__noreturn__)); 
# 78
void __throw_out_of_range(const char *) __attribute((__noreturn__)); 
# 81
void __throw_out_of_range_fmt(const char *, ...) __attribute((__noreturn__))
# 82
 __attribute((__format__(__gnu_printf__, 1, 2))); 
# 85
void __throw_runtime_error(const char *) __attribute((__noreturn__)); 
# 88
void __throw_range_error(const char *) __attribute((__noreturn__)); 
# 91
void __throw_overflow_error(const char *) __attribute((__noreturn__)); 
# 94
void __throw_underflow_error(const char *) __attribute((__noreturn__)); 
# 98
void __throw_ios_failure(const char *) __attribute((__noreturn__)); 
# 101
void __throw_ios_failure(const char *, int) __attribute((__noreturn__)); 
# 105
void __throw_system_error(int) __attribute((__noreturn__)); 
# 109
void __throw_future_error(int) __attribute((__noreturn__)); 
# 113
void __throw_bad_function_call() __attribute((__noreturn__)); 
# 116
}
# 37 "/usr/include/c++/12/ext/numeric_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 50 "/usr/include/c++/12/ext/numeric_traits.h" 3
template< class _Tp> 
# 51
struct __is_integer_nonstrict : public std::__is_integer< _Tp>  { 
# 54
using std::__is_integer< _Tp> ::__value;
# 57
enum { __width = (__value) ? sizeof(_Tp) * (8) : (0)}; 
# 58
}; 
# 60
template< class _Value> 
# 61
struct __numeric_traits_integer { 
# 64
static_assert((__is_integer_nonstrict< _Value> ::__value), "invalid specialization");
# 70
static const bool __is_signed = (((_Value)(-1)) < 0); 
# 71
static const int __digits = (__is_integer_nonstrict< _Value> ::__width - __is_signed); 
# 75
static const _Value __max = (__is_signed ? (((((_Value)1) << (__digits - 1)) - 1) << 1) + 1 : (~((_Value)0))); 
# 78
static const _Value __min = (__is_signed ? (-__max) - 1 : ((_Value)0)); 
# 79
}; 
# 81
template< class _Value> const _Value __numeric_traits_integer< _Value> ::__min; 
# 84
template< class _Value> const _Value __numeric_traits_integer< _Value> ::__max; 
# 87
template< class _Value> const bool __numeric_traits_integer< _Value> ::__is_signed; 
# 90
template< class _Value> const int __numeric_traits_integer< _Value> ::__digits; 
# 137 "/usr/include/c++/12/ext/numeric_traits.h" 3
template< class _Tp> using __int_traits = __numeric_traits_integer< _Tp> ; 
# 157 "/usr/include/c++/12/ext/numeric_traits.h" 3
template< class _Value> 
# 158
struct __numeric_traits_floating { 
# 161
static const int __max_digits10 = ((2) + ((((std::template __are_same< _Value, float> ::__value) ? 24 : ((std::template __are_same< _Value, double> ::__value) ? 53 : 64)) * 643L) / (2136))); 
# 164
static const bool __is_signed = true; 
# 165
static const int __digits10 = ((std::template __are_same< _Value, float> ::__value) ? 6 : ((std::template __are_same< _Value, double> ::__value) ? 15 : 18)); 
# 166
static const int __max_exponent10 = ((std::template __are_same< _Value, float> ::__value) ? 38 : ((std::template __are_same< _Value, double> ::__value) ? 308 : 4932)); 
# 167
}; 
# 169
template< class _Value> const int __numeric_traits_floating< _Value> ::__max_digits10; 
# 172
template< class _Value> const bool __numeric_traits_floating< _Value> ::__is_signed; 
# 175
template< class _Value> const int __numeric_traits_floating< _Value> ::__digits10; 
# 178
template< class _Value> const int __numeric_traits_floating< _Value> ::__max_exponent10; 
# 186
template< class _Value> 
# 187
struct __numeric_traits : public __numeric_traits_integer< _Value>  { 
# 189
}; 
# 192
template<> struct __numeric_traits< float>  : public __numeric_traits_floating< float>  { 
# 194
}; 
# 197
template<> struct __numeric_traits< double>  : public __numeric_traits_floating< double>  { 
# 199
}; 
# 202
template<> struct __numeric_traits< long double>  : public __numeric_traits_floating< long double>  { 
# 204
}; 
# 239 "/usr/include/c++/12/ext/numeric_traits.h" 3
}
# 40 "/usr/include/c++/12/type_traits" 3
namespace std __attribute((__visibility__("default"))) { 
# 44
template< class _Tp> class reference_wrapper; 
# 61 "/usr/include/c++/12/type_traits" 3
template< class _Tp, _Tp __v> 
# 62
struct integral_constant { 
# 64
static constexpr inline _Tp value = (__v); 
# 65
typedef _Tp value_type; 
# 66
typedef integral_constant type; 
# 67
constexpr operator value_type() const noexcept { return value; } 
# 72
constexpr value_type operator()() const noexcept { return value; } 
# 74
}; 
# 82
using true_type = integral_constant< bool, true> ; 
# 85
using false_type = integral_constant< bool, false> ; 
# 89
template< bool __v> using __bool_constant = integral_constant< bool, __v> ; 
# 97
template< bool __v> using bool_constant = integral_constant< bool, __v> ; 
# 103
template< bool > 
# 104
struct __conditional { 
# 106
template< class _Tp, class > using type = _Tp; 
# 108
}; 
# 111
template<> struct __conditional< false>  { 
# 113
template< class , class _Up> using type = _Up; 
# 115
}; 
# 118
template< bool _Cond, class _If, class _Else> using __conditional_t = typename __conditional< _Cond> ::template type< _If, _Else> ; 
# 123
template< class _Type> 
# 124
struct __type_identity { 
# 125
using type = _Type; }; 
# 127
template< class _Tp> using __type_identity_t = typename __type_identity< _Tp> ::type; 
# 130
template< class ...> struct __or_; 
# 134
template<> struct __or_< >  : public false_type { 
# 136
}; 
# 138
template< class _B1> 
# 139
struct __or_< _B1>  : public _B1 { 
# 141
}; 
# 143
template< class _B1, class _B2> 
# 144
struct __or_< _B1, _B2>  : public __conditional_t< _B1::value, _B1, _B2>  { 
# 146
}; 
# 148
template< class _B1, class _B2, class _B3, class ..._Bn> 
# 149
struct __or_< _B1, _B2, _B3, _Bn...>  : public __conditional_t< _B1::value, _B1, std::__or_< _B2, _B3, _Bn...> >  { 
# 151
}; 
# 153
template< class ...> struct __and_; 
# 157
template<> struct __and_< >  : public true_type { 
# 159
}; 
# 161
template< class _B1> 
# 162
struct __and_< _B1>  : public _B1 { 
# 164
}; 
# 166
template< class _B1, class _B2> 
# 167
struct __and_< _B1, _B2>  : public __conditional_t< _B1::value, _B2, _B1>  { 
# 169
}; 
# 171
template< class _B1, class _B2, class _B3, class ..._Bn> 
# 172
struct __and_< _B1, _B2, _B3, _Bn...>  : public __conditional_t< _B1::value, std::__and_< _B2, _B3, _Bn...> , _B1>  { 
# 174
}; 
# 176
template< class _Pp> 
# 177
struct __not_ : public __bool_constant< !((bool)_Pp::value)>  { 
# 179
}; 
# 185
template< class ..._Bn> constexpr bool 
# 186
__or_v = (__or_< _Bn...> ::value); 
# 187
template< class ..._Bn> constexpr bool 
# 188
__and_v = (__and_< _Bn...> ::value); 
# 193
template< class ..._Bn> 
# 194
struct conjunction : public __and_< _Bn...>  { 
# 196
}; 
# 198
template< class ..._Bn> 
# 199
struct disjunction : public __or_< _Bn...>  { 
# 201
}; 
# 203
template< class _Pp> 
# 204
struct negation : public __not_< _Pp>  { 
# 206
}; 
# 211
template< class ..._Bn> constexpr bool 
# 212
conjunction_v = (conjunction< _Bn...> ::value); 
# 214
template< class ..._Bn> constexpr bool 
# 215
disjunction_v = (disjunction< _Bn...> ::value); 
# 217
template< class _Pp> constexpr bool 
# 218
negation_v = (negation< _Pp> ::value); 
# 224
template< class > struct is_reference; 
# 226
template< class > struct is_function; 
# 228
template< class > struct is_void; 
# 230
template< class > struct remove_cv; 
# 232
template< class > struct is_const; 
# 236
template< class > struct __is_array_unknown_bounds; 
# 242
template< class _Tp, size_t  = sizeof(_Tp)> constexpr true_type 
# 243
__is_complete_or_unbounded(__type_identity< _Tp> ) 
# 244
{ return {}; } 
# 246
template< class _TypeIdentity, class 
# 247
_NestedType = typename _TypeIdentity::type> constexpr typename __or_< is_reference< _NestedType> , is_function< _NestedType> , is_void< _NestedType> , __is_array_unknown_bounds< _NestedType> > ::type 
# 253
__is_complete_or_unbounded(_TypeIdentity) 
# 254
{ return {}; } 
# 261
template< class _Tp> 
# 262
struct __success_type { 
# 263
typedef _Tp type; }; 
# 265
struct __failure_type { 
# 266
}; 
# 269
template< class _Tp> using __remove_cv_t = typename remove_cv< _Tp> ::type; 
# 274
template< class > 
# 275
struct __is_void_helper : public false_type { 
# 276
}; 
# 279
template<> struct __is_void_helper< void>  : public true_type { 
# 280
}; 
# 284
template< class _Tp> 
# 285
struct is_void : public __is_void_helper< __remove_cv_t< _Tp> > ::type { 
# 287
}; 
# 290
template< class > 
# 291
struct __is_integral_helper : public false_type { 
# 292
}; 
# 295
template<> struct __is_integral_helper< bool>  : public true_type { 
# 296
}; 
# 299
template<> struct __is_integral_helper< char>  : public true_type { 
# 300
}; 
# 303
template<> struct __is_integral_helper< signed char>  : public true_type { 
# 304
}; 
# 307
template<> struct __is_integral_helper< unsigned char>  : public true_type { 
# 308
}; 
# 314
template<> struct __is_integral_helper< wchar_t>  : public true_type { 
# 315
}; 
# 324
template<> struct __is_integral_helper< char16_t>  : public true_type { 
# 325
}; 
# 328
template<> struct __is_integral_helper< char32_t>  : public true_type { 
# 329
}; 
# 332
template<> struct __is_integral_helper< short>  : public true_type { 
# 333
}; 
# 336
template<> struct __is_integral_helper< unsigned short>  : public true_type { 
# 337
}; 
# 340
template<> struct __is_integral_helper< int>  : public true_type { 
# 341
}; 
# 344
template<> struct __is_integral_helper< unsigned>  : public true_type { 
# 345
}; 
# 348
template<> struct __is_integral_helper< long>  : public true_type { 
# 349
}; 
# 352
template<> struct __is_integral_helper< unsigned long>  : public true_type { 
# 353
}; 
# 356
template<> struct __is_integral_helper< long long>  : public true_type { 
# 357
}; 
# 360
template<> struct __is_integral_helper< unsigned long long>  : public true_type { 
# 361
}; 
# 368
template<> struct __is_integral_helper< __int128>  : public true_type { 
# 369
}; 
# 373
template<> struct __is_integral_helper< unsigned __int128>  : public true_type { 
# 374
}; 
# 412 "/usr/include/c++/12/type_traits" 3
template< class _Tp> 
# 413
struct is_integral : public __is_integral_helper< __remove_cv_t< _Tp> > ::type { 
# 415
}; 
# 418
template< class > 
# 419
struct __is_floating_point_helper : public false_type { 
# 420
}; 
# 423
template<> struct __is_floating_point_helper< float>  : public true_type { 
# 424
}; 
# 427
template<> struct __is_floating_point_helper< double>  : public true_type { 
# 428
}; 
# 431
template<> struct __is_floating_point_helper< long double>  : public true_type { 
# 432
}; 
# 442 "/usr/include/c++/12/type_traits" 3
template< class _Tp> 
# 443
struct is_floating_point : public __is_floating_point_helper< __remove_cv_t< _Tp> > ::type { 
# 445
}; 
# 448
template< class > 
# 449
struct is_array : public false_type { 
# 450
}; 
# 452
template< class _Tp, size_t _Size> 
# 453
struct is_array< _Tp [_Size]>  : public true_type { 
# 454
}; 
# 456
template< class _Tp> 
# 457
struct is_array< _Tp []>  : public true_type { 
# 458
}; 
# 460
template< class > 
# 461
struct __is_pointer_helper : public false_type { 
# 462
}; 
# 464
template< class _Tp> 
# 465
struct __is_pointer_helper< _Tp *>  : public true_type { 
# 466
}; 
# 469
template< class _Tp> 
# 470
struct is_pointer : public __is_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 472
}; 
# 475
template< class > 
# 476
struct is_lvalue_reference : public false_type { 
# 477
}; 
# 479
template< class _Tp> 
# 480
struct is_lvalue_reference< _Tp &>  : public true_type { 
# 481
}; 
# 484
template< class > 
# 485
struct is_rvalue_reference : public false_type { 
# 486
}; 
# 488
template< class _Tp> 
# 489
struct is_rvalue_reference< _Tp &&>  : public true_type { 
# 490
}; 
# 492
template< class > 
# 493
struct __is_member_object_pointer_helper : public false_type { 
# 494
}; 
# 496
template< class _Tp, class _Cp> 
# 497
struct __is_member_object_pointer_helper< _Tp (_Cp::*)>  : public __not_< is_function< _Tp> > ::type { 
# 498
}; 
# 501
template< class _Tp> 
# 502
struct is_member_object_pointer : public __is_member_object_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 504
}; 
# 506
template< class > 
# 507
struct __is_member_function_pointer_helper : public false_type { 
# 508
}; 
# 510
template< class _Tp, class _Cp> 
# 511
struct __is_member_function_pointer_helper< _Tp (_Cp::*)>  : public is_function< _Tp> ::type { 
# 512
}; 
# 515
template< class _Tp> 
# 516
struct is_member_function_pointer : public __is_member_function_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 518
}; 
# 521
template< class _Tp> 
# 522
struct is_enum : public integral_constant< bool, __is_enum(_Tp)>  { 
# 524
}; 
# 527
template< class _Tp> 
# 528
struct is_union : public integral_constant< bool, __is_union(_Tp)>  { 
# 530
}; 
# 533
template< class _Tp> 
# 534
struct is_class : public integral_constant< bool, __is_class(_Tp)>  { 
# 536
}; 
# 539
template< class _Tp> 
# 540
struct is_function : public __bool_constant< !is_const< const _Tp> ::value>  { 
# 541
}; 
# 543
template< class _Tp> 
# 544
struct is_function< _Tp &>  : public false_type { 
# 545
}; 
# 547
template< class _Tp> 
# 548
struct is_function< _Tp &&>  : public false_type { 
# 549
}; 
# 553
template< class > 
# 554
struct __is_null_pointer_helper : public false_type { 
# 555
}; 
# 558
template<> struct __is_null_pointer_helper< __decltype((nullptr))>  : public true_type { 
# 559
}; 
# 562
template< class _Tp> 
# 563
struct is_null_pointer : public __is_null_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 565
}; 
# 569
template< class _Tp> 
# 570
struct __is_nullptr_t : public is_null_pointer< _Tp>  { 
# 572
} __attribute((__deprecated__("use \'std::is_null_pointer\' instead"))); 
# 577
template< class _Tp> 
# 578
struct is_reference : public __or_< is_lvalue_reference< _Tp> , is_rvalue_reference< _Tp> > ::type { 
# 581
}; 
# 584
template< class _Tp> 
# 585
struct is_arithmetic : public __or_< is_integral< _Tp> , is_floating_point< _Tp> > ::type { 
# 587
}; 
# 590
template< class _Tp> 
# 591
struct is_fundamental : public __or_< is_arithmetic< _Tp> , is_void< _Tp> , is_null_pointer< _Tp> > ::type { 
# 594
}; 
# 597
template< class _Tp> 
# 598
struct is_object : public __not_< __or_< is_function< _Tp> , is_reference< _Tp> , is_void< _Tp> > > ::type { 
# 601
}; 
# 603
template< class > struct is_member_pointer; 
# 607
template< class _Tp> 
# 608
struct is_scalar : public __or_< is_arithmetic< _Tp> , is_enum< _Tp> , is_pointer< _Tp> , is_member_pointer< _Tp> , is_null_pointer< _Tp> > ::type { 
# 611
}; 
# 614
template< class _Tp> 
# 615
struct is_compound : public __not_< is_fundamental< _Tp> > ::type { 
# 616
}; 
# 619
template< class _Tp> 
# 620
struct __is_member_pointer_helper : public false_type { 
# 621
}; 
# 623
template< class _Tp, class _Cp> 
# 624
struct __is_member_pointer_helper< _Tp (_Cp::*)>  : public true_type { 
# 625
}; 
# 629
template< class _Tp> 
# 630
struct is_member_pointer : public __is_member_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 632
}; 
# 634
template< class , class > struct is_same; 
# 638
template< class _Tp, class ..._Types> using __is_one_of = __or_< is_same< _Tp, _Types> ...> ; 
# 643
template< class _Tp> using __is_signed_integer = __is_one_of< __remove_cv_t< _Tp> , signed char, signed short, signed int, signed long, signed long long, signed __int128> ; 
# 663 "/usr/include/c++/12/type_traits" 3
template< class _Tp> using __is_unsigned_integer = __is_one_of< __remove_cv_t< _Tp> , unsigned char, unsigned short, unsigned, unsigned long, unsigned long long, unsigned __int128> ; 
# 682 "/usr/include/c++/12/type_traits" 3
template< class _Tp> using __is_standard_integer = __or_< __is_signed_integer< _Tp> , __is_unsigned_integer< _Tp> > ; 
# 687
template< class ...> using __void_t = void; 
# 691
template< class _Tp, class  = void> 
# 692
struct __is_referenceable : public false_type { 
# 694
}; 
# 696
template< class _Tp> 
# 697
struct __is_referenceable< _Tp, __void_t< _Tp &> >  : public true_type { 
# 699
}; 
# 705
template< class > 
# 706
struct is_const : public false_type { 
# 707
}; 
# 709
template< class _Tp> 
# 710
struct is_const< const _Tp>  : public true_type { 
# 711
}; 
# 714
template< class > 
# 715
struct is_volatile : public false_type { 
# 716
}; 
# 718
template< class _Tp> 
# 719
struct is_volatile< volatile _Tp>  : public true_type { 
# 720
}; 
# 723
template< class _Tp> 
# 724
struct is_trivial : public integral_constant< bool, __is_trivial(_Tp)>  { 
# 727
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 729
}; 
# 732
template< class _Tp> 
# 733
struct is_trivially_copyable : public integral_constant< bool, __is_trivially_copyable(_Tp)>  { 
# 736
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 738
}; 
# 741
template< class _Tp> 
# 742
struct is_standard_layout : public integral_constant< bool, __is_standard_layout(_Tp)>  { 
# 745
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 747
}; 
# 754
template< class _Tp> 
# 757
struct is_pod : public integral_constant< bool, __is_pod(_Tp)>  { 
# 760
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 762
}; 
# 768
template< class _Tp> 
# 771
struct
# 770
 [[__deprecated__]] is_literal_type : public integral_constant< bool, __is_literal_type(_Tp)>  { 
# 774
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 776
}; 
# 779
template< class _Tp> 
# 780
struct is_empty : public integral_constant< bool, __is_empty(_Tp)>  { 
# 782
}; 
# 785
template< class _Tp> 
# 786
struct is_polymorphic : public integral_constant< bool, __is_polymorphic(_Tp)>  { 
# 788
}; 
# 794
template< class _Tp> 
# 795
struct is_final : public integral_constant< bool, __is_final(_Tp)>  { 
# 797
}; 
# 801
template< class _Tp> 
# 802
struct is_abstract : public integral_constant< bool, __is_abstract(_Tp)>  { 
# 804
}; 
# 807
template< class _Tp, bool 
# 808
 = is_arithmetic< _Tp> ::value> 
# 809
struct __is_signed_helper : public false_type { 
# 810
}; 
# 812
template< class _Tp> 
# 813
struct __is_signed_helper< _Tp, true>  : public integral_constant< bool, ((_Tp)(-1)) < ((_Tp)0)>  { 
# 815
}; 
# 819
template< class _Tp> 
# 820
struct is_signed : public __is_signed_helper< _Tp> ::type { 
# 822
}; 
# 825
template< class _Tp> 
# 826
struct is_unsigned : public __and_< is_arithmetic< _Tp> , __not_< is_signed< _Tp> > >  { 
# 828
}; 
# 831
template< class _Tp, class _Up = _Tp &&> _Up __declval(int); 
# 835
template< class _Tp> _Tp __declval(long); 
# 840
template< class _Tp> auto declval() noexcept->__decltype((__declval< _Tp> (0))); 
# 843
template< class , unsigned  = 0U> struct extent; 
# 846
template< class > struct remove_all_extents; 
# 850
template< class _Tp> 
# 851
struct __is_array_known_bounds : public integral_constant< bool, (extent< _Tp> ::value > 0)>  { 
# 853
}; 
# 855
template< class _Tp> 
# 856
struct __is_array_unknown_bounds : public __and_< is_array< _Tp> , __not_< extent< _Tp> > >  { 
# 858
}; 
# 867 "/usr/include/c++/12/type_traits" 3
struct __do_is_destructible_impl { 
# 869
template< class _Tp, class  = __decltype((declval< _Tp &> ().~_Tp()))> static true_type __test(int); 
# 872
template< class > static false_type __test(...); 
# 874
}; 
# 876
template< class _Tp> 
# 877
struct __is_destructible_impl : public __do_is_destructible_impl { 
# 880
typedef __decltype((__test< _Tp> (0))) type; 
# 881
}; 
# 883
template< class _Tp, bool 
# 884
 = __or_< is_void< _Tp> , __is_array_unknown_bounds< _Tp> , is_function< _Tp> > ::value, bool 
# 887
 = __or_< is_reference< _Tp> , is_scalar< _Tp> > ::value> struct __is_destructible_safe; 
# 890
template< class _Tp> 
# 891
struct __is_destructible_safe< _Tp, false, false>  : public __is_destructible_impl< typename remove_all_extents< _Tp> ::type> ::type { 
# 894
}; 
# 896
template< class _Tp> 
# 897
struct __is_destructible_safe< _Tp, true, false>  : public false_type { 
# 898
}; 
# 900
template< class _Tp> 
# 901
struct __is_destructible_safe< _Tp, false, true>  : public true_type { 
# 902
}; 
# 906
template< class _Tp> 
# 907
struct is_destructible : public __is_destructible_safe< _Tp> ::type { 
# 910
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 912
}; 
# 920
struct __do_is_nt_destructible_impl { 
# 922
template< class _Tp> static __bool_constant< noexcept(declval< _Tp &> ().~_Tp())>  __test(int); 
# 926
template< class > static false_type __test(...); 
# 928
}; 
# 930
template< class _Tp> 
# 931
struct __is_nt_destructible_impl : public __do_is_nt_destructible_impl { 
# 934
typedef __decltype((__test< _Tp> (0))) type; 
# 935
}; 
# 937
template< class _Tp, bool 
# 938
 = __or_< is_void< _Tp> , __is_array_unknown_bounds< _Tp> , is_function< _Tp> > ::value, bool 
# 941
 = __or_< is_reference< _Tp> , is_scalar< _Tp> > ::value> struct __is_nt_destructible_safe; 
# 944
template< class _Tp> 
# 945
struct __is_nt_destructible_safe< _Tp, false, false>  : public __is_nt_destructible_impl< typename remove_all_extents< _Tp> ::type> ::type { 
# 948
}; 
# 950
template< class _Tp> 
# 951
struct __is_nt_destructible_safe< _Tp, true, false>  : public false_type { 
# 952
}; 
# 954
template< class _Tp> 
# 955
struct __is_nt_destructible_safe< _Tp, false, true>  : public true_type { 
# 956
}; 
# 960
template< class _Tp> 
# 961
struct is_nothrow_destructible : public __is_nt_destructible_safe< _Tp> ::type { 
# 964
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 966
}; 
# 969
template< class _Tp, class ..._Args> 
# 970
struct __is_constructible_impl : public __bool_constant< __is_constructible(_Tp, _Args...)>  { 
# 972
}; 
# 976
template< class _Tp, class ..._Args> 
# 977
struct is_constructible : public __is_constructible_impl< _Tp, _Args...>  { 
# 980
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 982
}; 
# 985
template< class _Tp> 
# 986
struct is_default_constructible : public __is_constructible_impl< _Tp> ::type { 
# 989
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 991
}; 
# 994
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_copy_constructible_impl; 
# 997
template< class _Tp> 
# 998
struct __is_copy_constructible_impl< _Tp, false>  : public false_type { 
# 999
}; 
# 1001
template< class _Tp> 
# 1002
struct __is_copy_constructible_impl< _Tp, true>  : public __is_constructible_impl< _Tp, const _Tp &>  { 
# 1004
}; 
# 1008
template< class _Tp> 
# 1009
struct is_copy_constructible : public __is_copy_constructible_impl< _Tp>  { 
# 1012
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1014
}; 
# 1017
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_move_constructible_impl; 
# 1020
template< class _Tp> 
# 1021
struct __is_move_constructible_impl< _Tp, false>  : public false_type { 
# 1022
}; 
# 1024
template< class _Tp> 
# 1025
struct __is_move_constructible_impl< _Tp, true>  : public __is_constructible_impl< _Tp, _Tp &&>  { 
# 1027
}; 
# 1031
template< class _Tp> 
# 1032
struct is_move_constructible : public __is_move_constructible_impl< _Tp>  { 
# 1035
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1037
}; 
# 1040
template< class _Tp, class ..._Args> using __is_nothrow_constructible_impl = __bool_constant< __is_nothrow_constructible(_Tp, _Args...)> ; 
# 1046
template< class _Tp, class ..._Args> 
# 1047
struct is_nothrow_constructible : public __is_nothrow_constructible_impl< _Tp, _Args...> ::type { 
# 1050
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1052
}; 
# 1055
template< class _Tp> 
# 1056
struct is_nothrow_default_constructible : public __bool_constant< __is_nothrow_constructible(_Tp)>  { 
# 1059
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1061
}; 
# 1064
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nothrow_copy_constructible_impl; 
# 1067
template< class _Tp> 
# 1068
struct __is_nothrow_copy_constructible_impl< _Tp, false>  : public false_type { 
# 1069
}; 
# 1071
template< class _Tp> 
# 1072
struct __is_nothrow_copy_constructible_impl< _Tp, true>  : public __is_nothrow_constructible_impl< _Tp, const _Tp &>  { 
# 1074
}; 
# 1078
template< class _Tp> 
# 1079
struct is_nothrow_copy_constructible : public __is_nothrow_copy_constructible_impl< _Tp> ::type { 
# 1082
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1084
}; 
# 1087
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nothrow_move_constructible_impl; 
# 1090
template< class _Tp> 
# 1091
struct __is_nothrow_move_constructible_impl< _Tp, false>  : public false_type { 
# 1092
}; 
# 1094
template< class _Tp> 
# 1095
struct __is_nothrow_move_constructible_impl< _Tp, true>  : public __is_nothrow_constructible_impl< _Tp, _Tp &&>  { 
# 1097
}; 
# 1101
template< class _Tp> 
# 1102
struct is_nothrow_move_constructible : public __is_nothrow_move_constructible_impl< _Tp> ::type { 
# 1105
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1107
}; 
# 1110
template< class _Tp, class _Up> 
# 1111
struct is_assignable : public __bool_constant< __is_assignable(_Tp, _Up)>  { 
# 1114
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1116
}; 
# 1118
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_copy_assignable_impl; 
# 1121
template< class _Tp> 
# 1122
struct __is_copy_assignable_impl< _Tp, false>  : public false_type { 
# 1123
}; 
# 1125
template< class _Tp> 
# 1126
struct __is_copy_assignable_impl< _Tp, true>  : public __bool_constant< __is_assignable(_Tp &, const _Tp &)>  { 
# 1128
}; 
# 1131
template< class _Tp> 
# 1132
struct is_copy_assignable : public __is_copy_assignable_impl< _Tp> ::type { 
# 1135
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1137
}; 
# 1139
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_move_assignable_impl; 
# 1142
template< class _Tp> 
# 1143
struct __is_move_assignable_impl< _Tp, false>  : public false_type { 
# 1144
}; 
# 1146
template< class _Tp> 
# 1147
struct __is_move_assignable_impl< _Tp, true>  : public __bool_constant< __is_assignable(_Tp &, _Tp &&)>  { 
# 1149
}; 
# 1152
template< class _Tp> 
# 1153
struct is_move_assignable : public __is_move_assignable_impl< _Tp> ::type { 
# 1156
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1158
}; 
# 1160
template< class _Tp, class _Up> using __is_nothrow_assignable_impl = __bool_constant< __is_nothrow_assignable(_Tp, _Up)> ; 
# 1165
template< class _Tp, class _Up> 
# 1166
struct is_nothrow_assignable : public __is_nothrow_assignable_impl< _Tp, _Up>  { 
# 1169
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1171
}; 
# 1173
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nt_copy_assignable_impl; 
# 1176
template< class _Tp> 
# 1177
struct __is_nt_copy_assignable_impl< _Tp, false>  : public false_type { 
# 1178
}; 
# 1180
template< class _Tp> 
# 1181
struct __is_nt_copy_assignable_impl< _Tp, true>  : public __is_nothrow_assignable_impl< _Tp &, const _Tp &>  { 
# 1183
}; 
# 1186
template< class _Tp> 
# 1187
struct is_nothrow_copy_assignable : public __is_nt_copy_assignable_impl< _Tp>  { 
# 1190
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1192
}; 
# 1194
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nt_move_assignable_impl; 
# 1197
template< class _Tp> 
# 1198
struct __is_nt_move_assignable_impl< _Tp, false>  : public false_type { 
# 1199
}; 
# 1201
template< class _Tp> 
# 1202
struct __is_nt_move_assignable_impl< _Tp, true>  : public __is_nothrow_assignable_impl< _Tp &, _Tp &&>  { 
# 1204
}; 
# 1207
template< class _Tp> 
# 1208
struct is_nothrow_move_assignable : public __is_nt_move_assignable_impl< _Tp>  { 
# 1211
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1213
}; 
# 1216
template< class _Tp, class ..._Args> 
# 1217
struct is_trivially_constructible : public __bool_constant< __is_trivially_constructible(_Tp, _Args...)>  { 
# 1220
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1222
}; 
# 1225
template< class _Tp> 
# 1226
struct is_trivially_default_constructible : public __bool_constant< __is_trivially_constructible(_Tp)>  { 
# 1229
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1231
}; 
# 1233
struct __do_is_implicitly_default_constructible_impl { 
# 1235
template< class _Tp> static void __helper(const _Tp &); 
# 1238
template< class _Tp> static true_type __test(const _Tp &, __decltype((__helper< const _Tp &> ({}))) * = 0); 
# 1242
static false_type __test(...); 
# 1243
}; 
# 1245
template< class _Tp> 
# 1246
struct __is_implicitly_default_constructible_impl : public __do_is_implicitly_default_constructible_impl { 
# 1249
typedef __decltype((__test(declval< _Tp> ()))) type; 
# 1250
}; 
# 1252
template< class _Tp> 
# 1253
struct __is_implicitly_default_constructible_safe : public __is_implicitly_default_constructible_impl< _Tp> ::type { 
# 1255
}; 
# 1257
template< class _Tp> 
# 1258
struct __is_implicitly_default_constructible : public __and_< __is_constructible_impl< _Tp> , __is_implicitly_default_constructible_safe< _Tp> >  { 
# 1261
}; 
# 1263
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_copy_constructible_impl; 
# 1266
template< class _Tp> 
# 1267
struct __is_trivially_copy_constructible_impl< _Tp, false>  : public false_type { 
# 1268
}; 
# 1270
template< class _Tp> 
# 1271
struct __is_trivially_copy_constructible_impl< _Tp, true>  : public __and_< __is_copy_constructible_impl< _Tp> , integral_constant< bool, __is_trivially_constructible(_Tp, const _Tp &)> >  { 
# 1275
}; 
# 1278
template< class _Tp> 
# 1279
struct is_trivially_copy_constructible : public __is_trivially_copy_constructible_impl< _Tp>  { 
# 1282
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1284
}; 
# 1286
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_move_constructible_impl; 
# 1289
template< class _Tp> 
# 1290
struct __is_trivially_move_constructible_impl< _Tp, false>  : public false_type { 
# 1291
}; 
# 1293
template< class _Tp> 
# 1294
struct __is_trivially_move_constructible_impl< _Tp, true>  : public __and_< __is_move_constructible_impl< _Tp> , integral_constant< bool, __is_trivially_constructible(_Tp, _Tp &&)> >  { 
# 1298
}; 
# 1301
template< class _Tp> 
# 1302
struct is_trivially_move_constructible : public __is_trivially_move_constructible_impl< _Tp>  { 
# 1305
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1307
}; 
# 1310
template< class _Tp, class _Up> 
# 1311
struct is_trivially_assignable : public __bool_constant< __is_trivially_assignable(_Tp, _Up)>  { 
# 1314
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1316
}; 
# 1318
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_copy_assignable_impl; 
# 1321
template< class _Tp> 
# 1322
struct __is_trivially_copy_assignable_impl< _Tp, false>  : public false_type { 
# 1323
}; 
# 1325
template< class _Tp> 
# 1326
struct __is_trivially_copy_assignable_impl< _Tp, true>  : public __bool_constant< __is_trivially_assignable(_Tp &, const _Tp &)>  { 
# 1328
}; 
# 1331
template< class _Tp> 
# 1332
struct is_trivially_copy_assignable : public __is_trivially_copy_assignable_impl< _Tp>  { 
# 1335
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1337
}; 
# 1339
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_move_assignable_impl; 
# 1342
template< class _Tp> 
# 1343
struct __is_trivially_move_assignable_impl< _Tp, false>  : public false_type { 
# 1344
}; 
# 1346
template< class _Tp> 
# 1347
struct __is_trivially_move_assignable_impl< _Tp, true>  : public __bool_constant< __is_trivially_assignable(_Tp &, _Tp &&)>  { 
# 1349
}; 
# 1352
template< class _Tp> 
# 1353
struct is_trivially_move_assignable : public __is_trivially_move_assignable_impl< _Tp>  { 
# 1356
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1358
}; 
# 1361
template< class _Tp> 
# 1362
struct is_trivially_destructible : public __and_< __is_destructible_safe< _Tp> , __bool_constant< __has_trivial_destructor(_Tp)> >  { 
# 1366
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1368
}; 
# 1372
template< class _Tp> 
# 1373
struct has_virtual_destructor : public integral_constant< bool, __has_virtual_destructor(_Tp)>  { 
# 1376
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1378
}; 
# 1384
template< class _Tp> 
# 1385
struct alignment_of : public integral_constant< unsigned long, __alignof__(_Tp)>  { 
# 1388
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1390
}; 
# 1393
template< class > 
# 1394
struct rank : public integral_constant< unsigned long, 0UL>  { 
# 1395
}; 
# 1397
template< class _Tp, size_t _Size> 
# 1398
struct rank< _Tp [_Size]>  : public integral_constant< unsigned long, 1 + std::rank< _Tp> ::value>  { 
# 1399
}; 
# 1401
template< class _Tp> 
# 1402
struct rank< _Tp []>  : public integral_constant< unsigned long, 1 + std::rank< _Tp> ::value>  { 
# 1403
}; 
# 1406
template< class , unsigned _Uint> 
# 1407
struct extent : public integral_constant< unsigned long, 0UL>  { 
# 1408
}; 
# 1410
template< class _Tp, unsigned _Uint, size_t _Size> 
# 1411
struct extent< _Tp [_Size], _Uint>  : public integral_constant< unsigned long, (_Uint == (0)) ? _Size : std::extent< _Tp, _Uint - (1)> ::value>  { 
# 1415
}; 
# 1417
template< class _Tp, unsigned _Uint> 
# 1418
struct extent< _Tp [], _Uint>  : public integral_constant< unsigned long, (_Uint == (0)) ? 0 : std::extent< _Tp, _Uint - (1)> ::value>  { 
# 1422
}; 
# 1428
template< class _Tp, class _Up> 
# 1429
struct is_same : public integral_constant< bool, __is_same(_Tp, _Up)>  { 
# 1435
}; 
# 1445 "/usr/include/c++/12/type_traits" 3
template< class _Base, class _Derived> 
# 1446
struct is_base_of : public integral_constant< bool, __is_base_of(_Base, _Derived)>  { 
# 1448
}; 
# 1450
template< class _From, class _To, bool 
# 1451
 = __or_< is_void< _From> , is_function< _To> , is_array< _To> > ::value> 
# 1453
struct __is_convertible_helper { 
# 1455
typedef typename is_void< _To> ::type type; 
# 1456
}; 
# 1458
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
template< class _From, class _To> 
# 1461
class __is_convertible_helper< _From, _To, false>  { 
# 1463
template< class _To1> static void __test_aux(_To1) noexcept; 
# 1466
template< class _From1, class _To1, class 
# 1467
 = __decltype((__test_aux< _To1> (std::declval< _From1> ())))> static true_type 
# 1466
__test(int); 
# 1471
template< class , class > static false_type __test(...); 
# 1476
public: typedef __decltype((__test< _From, _To> (0))) type; 
# 1477
}; 
#pragma GCC diagnostic pop
# 1481
template< class _From, class _To> 
# 1482
struct is_convertible : public __is_convertible_helper< _From, _To> ::type { 
# 1484
}; 
# 1487
template< class _ToElementType, class _FromElementType> using __is_array_convertible = is_convertible< _FromElementType (*)[], _ToElementType (*)[]> ; 
# 1491
template< class _From, class _To, bool 
# 1492
 = __or_< is_void< _From> , is_function< _To> , is_array< _To> > ::value> 
# 1494
struct __is_nt_convertible_helper : public is_void< _To>  { 
# 1496
}; 
# 1498
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
template< class _From, class _To> 
# 1501
class __is_nt_convertible_helper< _From, _To, false>  { 
# 1503
template< class _To1> static void __test_aux(_To1) noexcept; 
# 1506
template< class _From1, class _To1> static __bool_constant< noexcept(__test_aux< _To1> (std::declval< _From1> ()))>  __test(int); 
# 1511
template< class , class > static false_type __test(...); 
# 1516
public: using type = __decltype((__test< _From, _To> (0))); 
# 1517
}; 
#pragma GCC diagnostic pop
# 1537 "/usr/include/c++/12/type_traits" 3
template< class _Tp> 
# 1538
struct remove_const { 
# 1539
typedef _Tp type; }; 
# 1541
template< class _Tp> 
# 1542
struct remove_const< const _Tp>  { 
# 1543
typedef _Tp type; }; 
# 1546
template< class _Tp> 
# 1547
struct remove_volatile { 
# 1548
typedef _Tp type; }; 
# 1550
template< class _Tp> 
# 1551
struct remove_volatile< volatile _Tp>  { 
# 1552
typedef _Tp type; }; 
# 1555
template< class _Tp> 
# 1556
struct remove_cv { 
# 1557
using type = _Tp; }; 
# 1559
template< class _Tp> 
# 1560
struct remove_cv< const _Tp>  { 
# 1561
using type = _Tp; }; 
# 1563
template< class _Tp> 
# 1564
struct remove_cv< volatile _Tp>  { 
# 1565
using type = _Tp; }; 
# 1567
template< class _Tp> 
# 1568
struct remove_cv< const volatile _Tp>  { 
# 1569
using type = _Tp; }; 
# 1572
template< class _Tp> 
# 1573
struct add_const { 
# 1574
typedef const _Tp type; }; 
# 1577
template< class _Tp> 
# 1578
struct add_volatile { 
# 1579
typedef volatile _Tp type; }; 
# 1582
template< class _Tp> 
# 1583
struct add_cv { 
# 1586
typedef typename add_const< typename add_volatile< _Tp> ::type> ::type type; 
# 1587
}; 
# 1594
template< class _Tp> using remove_const_t = typename remove_const< _Tp> ::type; 
# 1598
template< class _Tp> using remove_volatile_t = typename remove_volatile< _Tp> ::type; 
# 1602
template< class _Tp> using remove_cv_t = typename remove_cv< _Tp> ::type; 
# 1606
template< class _Tp> using add_const_t = typename add_const< _Tp> ::type; 
# 1610
template< class _Tp> using add_volatile_t = typename add_volatile< _Tp> ::type; 
# 1614
template< class _Tp> using add_cv_t = typename add_cv< _Tp> ::type; 
# 1621
template< class _Tp> 
# 1622
struct remove_reference { 
# 1623
typedef _Tp type; }; 
# 1625
template< class _Tp> 
# 1626
struct remove_reference< _Tp &>  { 
# 1627
typedef _Tp type; }; 
# 1629
template< class _Tp> 
# 1630
struct remove_reference< _Tp &&>  { 
# 1631
typedef _Tp type; }; 
# 1633
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> 
# 1634
struct __add_lvalue_reference_helper { 
# 1635
typedef _Tp type; }; 
# 1637
template< class _Tp> 
# 1638
struct __add_lvalue_reference_helper< _Tp, true>  { 
# 1639
typedef _Tp &type; }; 
# 1642
template< class _Tp> 
# 1643
struct add_lvalue_reference : public __add_lvalue_reference_helper< _Tp>  { 
# 1645
}; 
# 1647
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> 
# 1648
struct __add_rvalue_reference_helper { 
# 1649
typedef _Tp type; }; 
# 1651
template< class _Tp> 
# 1652
struct __add_rvalue_reference_helper< _Tp, true>  { 
# 1653
typedef _Tp &&type; }; 
# 1656
template< class _Tp> 
# 1657
struct add_rvalue_reference : public __add_rvalue_reference_helper< _Tp>  { 
# 1659
}; 
# 1663
template< class _Tp> using remove_reference_t = typename remove_reference< _Tp> ::type; 
# 1667
template< class _Tp> using add_lvalue_reference_t = typename add_lvalue_reference< _Tp> ::type; 
# 1671
template< class _Tp> using add_rvalue_reference_t = typename add_rvalue_reference< _Tp> ::type; 
# 1680
template< class _Unqualified, bool _IsConst, bool _IsVol> struct __cv_selector; 
# 1683
template< class _Unqualified> 
# 1684
struct __cv_selector< _Unqualified, false, false>  { 
# 1685
typedef _Unqualified __type; }; 
# 1687
template< class _Unqualified> 
# 1688
struct __cv_selector< _Unqualified, false, true>  { 
# 1689
typedef volatile _Unqualified __type; }; 
# 1691
template< class _Unqualified> 
# 1692
struct __cv_selector< _Unqualified, true, false>  { 
# 1693
typedef const _Unqualified __type; }; 
# 1695
template< class _Unqualified> 
# 1696
struct __cv_selector< _Unqualified, true, true>  { 
# 1697
typedef const volatile _Unqualified __type; }; 
# 1699
template< class _Qualified, class _Unqualified, bool 
# 1700
_IsConst = is_const< _Qualified> ::value, bool 
# 1701
_IsVol = is_volatile< _Qualified> ::value> 
# 1702
class __match_cv_qualifiers { 
# 1704
typedef __cv_selector< _Unqualified, _IsConst, _IsVol>  __match; 
# 1707
public: typedef typename __cv_selector< _Unqualified, _IsConst, _IsVol> ::__type __type; 
# 1708
}; 
# 1711
template< class _Tp> 
# 1712
struct __make_unsigned { 
# 1713
typedef _Tp __type; }; 
# 1716
template<> struct __make_unsigned< char>  { 
# 1717
typedef unsigned char __type; }; 
# 1720
template<> struct __make_unsigned< signed char>  { 
# 1721
typedef unsigned char __type; }; 
# 1724
template<> struct __make_unsigned< short>  { 
# 1725
typedef unsigned short __type; }; 
# 1728
template<> struct __make_unsigned< int>  { 
# 1729
typedef unsigned __type; }; 
# 1732
template<> struct __make_unsigned< long>  { 
# 1733
typedef unsigned long __type; }; 
# 1736
template<> struct __make_unsigned< long long>  { 
# 1737
typedef unsigned long long __type; }; 
# 1742
template<> struct __make_unsigned< __int128>  { 
# 1743
typedef unsigned __int128 __type; }; 
# 1765 "/usr/include/c++/12/type_traits" 3
template< class _Tp, bool 
# 1766
_IsInt = is_integral< _Tp> ::value, bool 
# 1767
_IsEnum = is_enum< _Tp> ::value> class __make_unsigned_selector; 
# 1770
template< class _Tp> 
# 1771
class __make_unsigned_selector< _Tp, true, false>  { 
# 1773
using __unsigned_type = typename __make_unsigned< __remove_cv_t< _Tp> > ::__type; 
# 1777
public: using __type = typename __match_cv_qualifiers< _Tp, __unsigned_type> ::__type; 
# 1779
}; 
# 1781
class __make_unsigned_selector_base { 
# 1784
protected: template< class ...> struct _List { }; 
# 1786
template< class _Tp, class ..._Up> 
# 1787
struct _List< _Tp, _Up...>  : public __make_unsigned_selector_base::_List< _Up...>  { 
# 1788
static constexpr inline std::size_t __size = sizeof(_Tp); }; 
# 1790
template< size_t _Sz, class _Tp, bool  = _Sz <= _Tp::__size> struct __select; 
# 1793
template< size_t _Sz, class _Uint, class ..._UInts> 
# 1794
struct __select< _Sz, _List< _Uint, _UInts...> , true>  { 
# 1795
using __type = _Uint; }; 
# 1797
template< size_t _Sz, class _Uint, class ..._UInts> 
# 1798
struct __select< _Sz, _List< _Uint, _UInts...> , false>  : public __make_unsigned_selector_base::__select< _Sz, _List< _UInts...> >  { 
# 1800
}; 
# 1801
}; 
# 1804
template< class _Tp> 
# 1805
class __make_unsigned_selector< _Tp, false, true>  : private __make_unsigned_selector_base { 
# 1809
using _UInts = _List< unsigned char, unsigned short, unsigned, unsigned long, unsigned long long> ; 
# 1812
using __unsigned_type = typename __select< sizeof(_Tp), _List< unsigned char, unsigned short, unsigned, unsigned long, unsigned long long> > ::__type; 
# 1815
public: using __type = typename __match_cv_qualifiers< _Tp, __unsigned_type> ::__type; 
# 1817
}; 
# 1824
template<> struct __make_unsigned< wchar_t>  { 
# 1826
using __type = __make_unsigned_selector< wchar_t, false, true> ::__type; 
# 1828
}; 
# 1840 "/usr/include/c++/12/type_traits" 3
template<> struct __make_unsigned< char16_t>  { 
# 1842
using __type = __make_unsigned_selector< char16_t, false, true> ::__type; 
# 1844
}; 
# 1847
template<> struct __make_unsigned< char32_t>  { 
# 1849
using __type = __make_unsigned_selector< char32_t, false, true> ::__type; 
# 1851
}; 
# 1858
template< class _Tp> 
# 1859
struct make_unsigned { 
# 1860
typedef typename __make_unsigned_selector< _Tp> ::__type type; }; 
# 1864
template<> struct make_unsigned< bool> ; 
# 1869
template< class _Tp> 
# 1870
struct __make_signed { 
# 1871
typedef _Tp __type; }; 
# 1874
template<> struct __make_signed< char>  { 
# 1875
typedef signed char __type; }; 
# 1878
template<> struct __make_signed< unsigned char>  { 
# 1879
typedef signed char __type; }; 
# 1882
template<> struct __make_signed< unsigned short>  { 
# 1883
typedef signed short __type; }; 
# 1886
template<> struct __make_signed< unsigned>  { 
# 1887
typedef signed int __type; }; 
# 1890
template<> struct __make_signed< unsigned long>  { 
# 1891
typedef signed long __type; }; 
# 1894
template<> struct __make_signed< unsigned long long>  { 
# 1895
typedef signed long long __type; }; 
# 1900
template<> struct __make_signed< unsigned __int128>  { 
# 1901
typedef __int128 __type; }; 
# 1923 "/usr/include/c++/12/type_traits" 3
template< class _Tp, bool 
# 1924
_IsInt = is_integral< _Tp> ::value, bool 
# 1925
_IsEnum = is_enum< _Tp> ::value> class __make_signed_selector; 
# 1928
template< class _Tp> 
# 1929
class __make_signed_selector< _Tp, true, false>  { 
# 1931
using __signed_type = typename __make_signed< __remove_cv_t< _Tp> > ::__type; 
# 1935
public: using __type = typename __match_cv_qualifiers< _Tp, __signed_type> ::__type; 
# 1937
}; 
# 1940
template< class _Tp> 
# 1941
class __make_signed_selector< _Tp, false, true>  { 
# 1943
typedef typename __make_unsigned_selector< _Tp> ::__type __unsigned_type; 
# 1946
public: typedef typename std::__make_signed_selector< __unsigned_type> ::__type __type; 
# 1947
}; 
# 1954
template<> struct __make_signed< wchar_t>  { 
# 1956
using __type = __make_signed_selector< wchar_t, false, true> ::__type; 
# 1958
}; 
# 1970 "/usr/include/c++/12/type_traits" 3
template<> struct __make_signed< char16_t>  { 
# 1972
using __type = __make_signed_selector< char16_t, false, true> ::__type; 
# 1974
}; 
# 1977
template<> struct __make_signed< char32_t>  { 
# 1979
using __type = __make_signed_selector< char32_t, false, true> ::__type; 
# 1981
}; 
# 1988
template< class _Tp> 
# 1989
struct make_signed { 
# 1990
typedef typename __make_signed_selector< _Tp> ::__type type; }; 
# 1994
template<> struct make_signed< bool> ; 
# 1998
template< class _Tp> using make_signed_t = typename make_signed< _Tp> ::type; 
# 2002
template< class _Tp> using make_unsigned_t = typename make_unsigned< _Tp> ::type; 
# 2009
template< class _Tp> 
# 2010
struct remove_extent { 
# 2011
typedef _Tp type; }; 
# 2013
template< class _Tp, size_t _Size> 
# 2014
struct remove_extent< _Tp [_Size]>  { 
# 2015
typedef _Tp type; }; 
# 2017
template< class _Tp> 
# 2018
struct remove_extent< _Tp []>  { 
# 2019
typedef _Tp type; }; 
# 2022
template< class _Tp> 
# 2023
struct remove_all_extents { 
# 2024
typedef _Tp type; }; 
# 2026
template< class _Tp, size_t _Size> 
# 2027
struct remove_all_extents< _Tp [_Size]>  { 
# 2028
typedef typename std::remove_all_extents< _Tp> ::type type; }; 
# 2030
template< class _Tp> 
# 2031
struct remove_all_extents< _Tp []>  { 
# 2032
typedef typename std::remove_all_extents< _Tp> ::type type; }; 
# 2036
template< class _Tp> using remove_extent_t = typename remove_extent< _Tp> ::type; 
# 2040
template< class _Tp> using remove_all_extents_t = typename remove_all_extents< _Tp> ::type; 
# 2046
template< class _Tp, class > 
# 2047
struct __remove_pointer_helper { 
# 2048
typedef _Tp type; }; 
# 2050
template< class _Tp, class _Up> 
# 2051
struct __remove_pointer_helper< _Tp, _Up *>  { 
# 2052
typedef _Up type; }; 
# 2055
template< class _Tp> 
# 2056
struct remove_pointer : public __remove_pointer_helper< _Tp, __remove_cv_t< _Tp> >  { 
# 2058
}; 
# 2060
template< class _Tp, bool  = __or_< __is_referenceable< _Tp> , is_void< _Tp> > ::value> 
# 2062
struct __add_pointer_helper { 
# 2063
typedef _Tp type; }; 
# 2065
template< class _Tp> 
# 2066
struct __add_pointer_helper< _Tp, true>  { 
# 2067
typedef typename remove_reference< _Tp> ::type *type; }; 
# 2070
template< class _Tp> 
# 2071
struct add_pointer : public __add_pointer_helper< _Tp>  { 
# 2073
}; 
# 2077
template< class _Tp> using remove_pointer_t = typename remove_pointer< _Tp> ::type; 
# 2081
template< class _Tp> using add_pointer_t = typename add_pointer< _Tp> ::type; 
# 2085
template< size_t _Len> 
# 2086
struct __aligned_storage_msa { 
# 2088
union __type { 
# 2090
unsigned char __data[_Len]; 
# 2091
struct __attribute((__aligned__)) { } __align; 
# 2092
}; 
# 2093
}; 
# 2105 "/usr/include/c++/12/type_traits" 3
template< size_t _Len, size_t _Align = __alignof__(typename __aligned_storage_msa< _Len> ::__type)> 
# 2107
struct aligned_storage { 
# 2109
union type { 
# 2111
unsigned char __data[_Len]; 
# 2112
struct __attribute((__aligned__(_Align))) { } __align; 
# 2113
}; 
# 2114
}; 
# 2116
template< class ..._Types> 
# 2117
struct __strictest_alignment { 
# 2119
static const size_t _S_alignment = (0); 
# 2120
static const size_t _S_size = (0); 
# 2121
}; 
# 2123
template< class _Tp, class ..._Types> 
# 2124
struct __strictest_alignment< _Tp, _Types...>  { 
# 2126
static const size_t _S_alignment = ((__alignof__(_Tp) > __strictest_alignment< _Types...> ::_S_alignment) ? __alignof__(_Tp) : __strictest_alignment< _Types...> ::_S_alignment); 
# 2129
static const size_t _S_size = ((sizeof(_Tp) > __strictest_alignment< _Types...> ::_S_size) ? sizeof(_Tp) : __strictest_alignment< _Types...> ::_S_size); 
# 2132
}; 
# 2144 "/usr/include/c++/12/type_traits" 3
template< size_t _Len, class ..._Types> 
# 2145
struct aligned_union { 
# 2148
static_assert((sizeof...(_Types) != (0)), "At least one type is required");
# 2150
private: using __strictest = __strictest_alignment< _Types...> ; 
# 2151
static const size_t _S_len = ((_Len > __strictest::_S_size) ? _Len : __strictest::_S_size); 
# 2155
public: static const size_t alignment_value = (__strictest::_S_alignment); 
# 2157
typedef typename aligned_storage< _S_len, alignment_value> ::type type; 
# 2158
}; 
# 2160
template< size_t _Len, class ..._Types> const size_t aligned_union< _Len, _Types...> ::alignment_value; 
# 2167
template< class _Up, bool 
# 2168
_IsArray = is_array< _Up> ::value, bool 
# 2169
_IsFunction = is_function< _Up> ::value> struct __decay_selector; 
# 2173
template< class _Up> 
# 2174
struct __decay_selector< _Up, false, false>  { 
# 2175
typedef __remove_cv_t< _Up>  __type; }; 
# 2177
template< class _Up> 
# 2178
struct __decay_selector< _Up, true, false>  { 
# 2179
typedef typename remove_extent< _Up> ::type *__type; }; 
# 2181
template< class _Up> 
# 2182
struct __decay_selector< _Up, false, true>  { 
# 2183
typedef typename add_pointer< _Up> ::type __type; }; 
# 2187
template< class _Tp> 
# 2188
class decay { 
# 2190
typedef typename remove_reference< _Tp> ::type __remove_type; 
# 2193
public: typedef typename __decay_selector< __remove_type> ::__type type; 
# 2194
}; 
# 2199
template< class _Tp> 
# 2200
struct __strip_reference_wrapper { 
# 2202
typedef _Tp __type; 
# 2203
}; 
# 2205
template< class _Tp> 
# 2206
struct __strip_reference_wrapper< reference_wrapper< _Tp> >  { 
# 2208
typedef _Tp &__type; 
# 2209
}; 
# 2212
template< class _Tp> using __decay_t = typename decay< _Tp> ::type; 
# 2215
template< class _Tp> using __decay_and_strip = __strip_reference_wrapper< __decay_t< _Tp> > ; 
# 2221
template< bool , class _Tp = void> 
# 2222
struct enable_if { 
# 2223
}; 
# 2226
template< class _Tp> 
# 2227
struct enable_if< true, _Tp>  { 
# 2228
typedef _Tp type; }; 
# 2233
template< bool _Cond, class _Tp = void> using __enable_if_t = typename enable_if< _Cond, _Tp> ::type; 
# 2237
template< class ..._Cond> using _Require = __enable_if_t< __and_< _Cond...> ::value> ; 
# 2241
template< class _Tp> using __remove_cvref_t = typename remove_cv< typename remove_reference< _Tp> ::type> ::type; 
# 2248
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 2249
struct conditional { 
# 2250
typedef _Iftrue type; }; 
# 2253
template< class _Iftrue, class _Iffalse> 
# 2254
struct conditional< false, _Iftrue, _Iffalse>  { 
# 2255
typedef _Iffalse type; }; 
# 2258
template< class ..._Tp> struct common_type; 
# 2264
struct __do_common_type_impl { 
# 2266
template< class _Tp, class _Up> using __cond_t = __decltype((true ? std::declval< _Tp> () : std::declval< _Up> ())); 
# 2272
template< class _Tp, class _Up> static __success_type< __decay_t< __cond_t< _Tp, _Up> > >  _S_test(int); 
# 2284 "/usr/include/c++/12/type_traits" 3
template< class , class > static __failure_type _S_test_2(...); 
# 2288
template< class _Tp, class _Up> static __decltype((_S_test_2< _Tp, _Up> (0))) _S_test(...); 
# 2291
}; 
# 2295
template<> struct common_type< >  { 
# 2296
}; 
# 2299
template< class _Tp0> 
# 2300
struct common_type< _Tp0>  : public std::common_type< _Tp0, _Tp0>  { 
# 2302
}; 
# 2305
template< class _Tp1, class _Tp2, class 
# 2306
_Dp1 = __decay_t< _Tp1> , class _Dp2 = __decay_t< _Tp2> > 
# 2307
struct __common_type_impl { 
# 2311
using type = common_type< _Dp1, _Dp2> ; 
# 2312
}; 
# 2314
template< class _Tp1, class _Tp2> 
# 2315
struct __common_type_impl< _Tp1, _Tp2, _Tp1, _Tp2>  : private __do_common_type_impl { 
# 2320
using type = __decltype((_S_test< _Tp1, _Tp2> (0))); 
# 2321
}; 
# 2324
template< class _Tp1, class _Tp2> 
# 2325
struct common_type< _Tp1, _Tp2>  : public __common_type_impl< _Tp1, _Tp2> ::type { 
# 2327
}; 
# 2329
template< class ...> 
# 2330
struct __common_type_pack { 
# 2331
}; 
# 2333
template< class , class , class  = void> struct __common_type_fold; 
# 2337
template< class _Tp1, class _Tp2, class ..._Rp> 
# 2338
struct common_type< _Tp1, _Tp2, _Rp...>  : public __common_type_fold< std::common_type< _Tp1, _Tp2> , __common_type_pack< _Rp...> >  { 
# 2341
}; 
# 2346
template< class _CTp, class ..._Rp> 
# 2347
struct __common_type_fold< _CTp, __common_type_pack< _Rp...> , __void_t< typename _CTp::type> >  : public common_type< typename _CTp::type, _Rp...>  { 
# 2350
}; 
# 2353
template< class _CTp, class _Rp> 
# 2354
struct __common_type_fold< _CTp, _Rp, void>  { 
# 2355
}; 
# 2357
template< class _Tp, bool  = is_enum< _Tp> ::value> 
# 2358
struct __underlying_type_impl { 
# 2360
using type = __underlying_type(_Tp); 
# 2361
}; 
# 2363
template< class _Tp> 
# 2364
struct __underlying_type_impl< _Tp, false>  { 
# 2365
}; 
# 2369
template< class _Tp> 
# 2370
struct underlying_type : public __underlying_type_impl< _Tp>  { 
# 2372
}; 
# 2375
template< class _Tp> 
# 2376
struct __declval_protector { 
# 2378
static const bool __stop = false; 
# 2379
}; 
# 2386
template< class _Tp> auto 
# 2387
declval() noexcept->__decltype((__declval< _Tp> (0))) 
# 2388
{ 
# 2389
static_assert((__declval_protector< _Tp> ::__stop), "declval() must not be used!");
# 2391
return __declval< _Tp> (0); 
# 2392
} 
# 2395
template< class _Signature> struct result_of; 
# 2403
struct __invoke_memfun_ref { }; 
# 2404
struct __invoke_memfun_deref { }; 
# 2405
struct __invoke_memobj_ref { }; 
# 2406
struct __invoke_memobj_deref { }; 
# 2407
struct __invoke_other { }; 
# 2410
template< class _Tp, class _Tag> 
# 2411
struct __result_of_success : public __success_type< _Tp>  { 
# 2412
using __invoke_type = _Tag; }; 
# 2415
struct __result_of_memfun_ref_impl { 
# 2417
template< class _Fp, class _Tp1, class ..._Args> static __result_of_success< __decltype(((std::declval< _Tp1> ().*std::declval< _Fp> ())(std::declval< _Args> ()...))), __invoke_memfun_ref>  _S_test(int); 
# 2422
template< class ...> static __failure_type _S_test(...); 
# 2424
}; 
# 2426
template< class _MemPtr, class _Arg, class ..._Args> 
# 2427
struct __result_of_memfun_ref : private __result_of_memfun_ref_impl { 
# 2430
typedef __decltype((_S_test< _MemPtr, _Arg, _Args...> (0))) type; 
# 2431
}; 
# 2434
struct __result_of_memfun_deref_impl { 
# 2436
template< class _Fp, class _Tp1, class ..._Args> static __result_of_success< __decltype((((*std::declval< _Tp1> ()).*std::declval< _Fp> ())(std::declval< _Args> ()...))), __invoke_memfun_deref>  _S_test(int); 
# 2441
template< class ...> static __failure_type _S_test(...); 
# 2443
}; 
# 2445
template< class _MemPtr, class _Arg, class ..._Args> 
# 2446
struct __result_of_memfun_deref : private __result_of_memfun_deref_impl { 
# 2449
typedef __decltype((_S_test< _MemPtr, _Arg, _Args...> (0))) type; 
# 2450
}; 
# 2453
struct __result_of_memobj_ref_impl { 
# 2455
template< class _Fp, class _Tp1> static __result_of_success< __decltype((std::declval< _Tp1> ().*std::declval< _Fp> ())), __invoke_memobj_ref>  _S_test(int); 
# 2460
template< class , class > static __failure_type _S_test(...); 
# 2462
}; 
# 2464
template< class _MemPtr, class _Arg> 
# 2465
struct __result_of_memobj_ref : private __result_of_memobj_ref_impl { 
# 2468
typedef __decltype((_S_test< _MemPtr, _Arg> (0))) type; 
# 2469
}; 
# 2472
struct __result_of_memobj_deref_impl { 
# 2474
template< class _Fp, class _Tp1> static __result_of_success< __decltype(((*std::declval< _Tp1> ()).*std::declval< _Fp> ())), __invoke_memobj_deref>  _S_test(int); 
# 2479
template< class , class > static __failure_type _S_test(...); 
# 2481
}; 
# 2483
template< class _MemPtr, class _Arg> 
# 2484
struct __result_of_memobj_deref : private __result_of_memobj_deref_impl { 
# 2487
typedef __decltype((_S_test< _MemPtr, _Arg> (0))) type; 
# 2488
}; 
# 2490
template< class _MemPtr, class _Arg> struct __result_of_memobj; 
# 2493
template< class _Res, class _Class, class _Arg> 
# 2494
struct __result_of_memobj< _Res (_Class::*), _Arg>  { 
# 2496
typedef __remove_cvref_t< _Arg>  _Argval; 
# 2497
typedef _Res (_Class::*_MemPtr); 
# 2502
typedef typename __conditional_t< __or_< is_same< _Argval, _Class> , is_base_of< _Class, _Argval> > ::value, __result_of_memobj_ref< _MemPtr, _Arg> , __result_of_memobj_deref< _MemPtr, _Arg> > ::type type; 
# 2503
}; 
# 2505
template< class _MemPtr, class _Arg, class ..._Args> struct __result_of_memfun; 
# 2508
template< class _Res, class _Class, class _Arg, class ..._Args> 
# 2509
struct __result_of_memfun< _Res (_Class::*), _Arg, _Args...>  { 
# 2511
typedef typename remove_reference< _Arg> ::type _Argval; 
# 2512
typedef _Res (_Class::*_MemPtr); 
# 2516
typedef typename __conditional_t< is_base_of< _Class, _Argval> ::value, __result_of_memfun_ref< _MemPtr, _Arg, _Args...> , __result_of_memfun_deref< _MemPtr, _Arg, _Args...> > ::type type; 
# 2517
}; 
# 2524
template< class _Tp, class _Up = __remove_cvref_t< _Tp> > 
# 2525
struct __inv_unwrap { 
# 2527
using type = _Tp; 
# 2528
}; 
# 2530
template< class _Tp, class _Up> 
# 2531
struct __inv_unwrap< _Tp, reference_wrapper< _Up> >  { 
# 2533
using type = _Up &; 
# 2534
}; 
# 2536
template< bool , bool , class _Functor, class ..._ArgTypes> 
# 2537
struct __result_of_impl { 
# 2539
typedef __failure_type type; 
# 2540
}; 
# 2542
template< class _MemPtr, class _Arg> 
# 2543
struct __result_of_impl< true, false, _MemPtr, _Arg>  : public __result_of_memobj< __decay_t< _MemPtr> , typename __inv_unwrap< _Arg> ::type>  { 
# 2546
}; 
# 2548
template< class _MemPtr, class _Arg, class ..._Args> 
# 2549
struct __result_of_impl< false, true, _MemPtr, _Arg, _Args...>  : public __result_of_memfun< __decay_t< _MemPtr> , typename __inv_unwrap< _Arg> ::type, _Args...>  { 
# 2552
}; 
# 2555
struct __result_of_other_impl { 
# 2557
template< class _Fn, class ..._Args> static __result_of_success< __decltype((std::declval< _Fn> ()(std::declval< _Args> ()...))), __invoke_other>  _S_test(int); 
# 2562
template< class ...> static __failure_type _S_test(...); 
# 2564
}; 
# 2566
template< class _Functor, class ..._ArgTypes> 
# 2567
struct __result_of_impl< false, false, _Functor, _ArgTypes...>  : private __result_of_other_impl { 
# 2570
typedef __decltype((_S_test< _Functor, _ArgTypes...> (0))) type; 
# 2571
}; 
# 2574
template< class _Functor, class ..._ArgTypes> 
# 2575
struct __invoke_result : public __result_of_impl< is_member_object_pointer< typename remove_reference< _Functor> ::type> ::value, is_member_function_pointer< typename remove_reference< _Functor> ::type> ::value, _Functor, _ArgTypes...> ::type { 
# 2585
}; 
# 2588
template< class _Functor, class ..._ArgTypes> 
# 2589
struct result_of< _Functor (_ArgTypes ...)>  : public __invoke_result< _Functor, _ArgTypes...>  { 
# 2591
} __attribute((__deprecated__("use \'std::invoke_result\' instead"))); 
# 2595
template< size_t _Len, size_t _Align = __alignof__(typename __aligned_storage_msa< _Len> ::__type)> using aligned_storage_t = typename aligned_storage< _Len, _Align> ::type; 
# 2599
template< size_t _Len, class ..._Types> using aligned_union_t = typename aligned_union< _Len, _Types...> ::type; 
# 2603
template< class _Tp> using decay_t = typename decay< _Tp> ::type; 
# 2607
template< bool _Cond, class _Tp = void> using enable_if_t = typename enable_if< _Cond, _Tp> ::type; 
# 2611
template< bool _Cond, class _Iftrue, class _Iffalse> using conditional_t = typename conditional< _Cond, _Iftrue, _Iffalse> ::type; 
# 2615
template< class ..._Tp> using common_type_t = typename common_type< _Tp...> ::type; 
# 2619
template< class _Tp> using underlying_type_t = typename underlying_type< _Tp> ::type; 
# 2623
template< class _Tp> using result_of_t = typename result_of< _Tp> ::type; 
# 2630
template< class ...> using void_t = void; 
# 2657 "/usr/include/c++/12/type_traits" 3
template< class _Default, class _AlwaysVoid, 
# 2658
template< class ...>  class _Op, class ..._Args> 
# 2659
struct __detector { 
# 2661
using type = _Default; 
# 2662
using __is_detected = false_type; 
# 2663
}; 
# 2666
template< class _Default, template< class ...>  class _Op, class ...
# 2667
_Args> 
# 2668
struct __detector< _Default, __void_t< _Op< _Args...> > , _Op, _Args...>  { 
# 2670
using type = _Op< _Args...> ; 
# 2671
using __is_detected = true_type; 
# 2672
}; 
# 2674
template< class _Default, template< class ...>  class _Op, class ...
# 2675
_Args> using __detected_or = __detector< _Default, void, _Op, _Args...> ; 
# 2680
template< class _Default, template< class ...>  class _Op, class ...
# 2681
_Args> using __detected_or_t = typename __detected_or< _Default, _Op, _Args...> ::type; 
# 2699 "/usr/include/c++/12/type_traits" 3
template< class _Tp> struct __is_swappable; 
# 2702
template< class _Tp> struct __is_nothrow_swappable; 
# 2705
template< class > 
# 2706
struct __is_tuple_like_impl : public false_type { 
# 2707
}; 
# 2710
template< class _Tp> 
# 2711
struct __is_tuple_like : public __is_tuple_like_impl< __remove_cvref_t< _Tp> > ::type { 
# 2713
}; 
# 2716
template< class _Tp> inline _Require< __not_< __is_tuple_like< _Tp> > , is_move_constructible< _Tp> , is_move_assignable< _Tp> >  swap(_Tp &, _Tp &) noexcept(__and_< is_nothrow_move_constructible< _Tp> , is_nothrow_move_assignable< _Tp> > ::value); 
# 2726
template< class _Tp, size_t _Nm> inline __enable_if_t< __is_swappable< _Tp> ::value>  swap(_Tp (& __a)[_Nm], _Tp (& __b)[_Nm]) noexcept(__is_nothrow_swappable< _Tp> ::value); 
# 2734
namespace __swappable_details { 
# 2735
using std::swap;
# 2737
struct __do_is_swappable_impl { 
# 2739
template< class _Tp, class 
# 2740
 = __decltype((swap(std::declval< _Tp &> (), std::declval< _Tp &> ())))> static true_type 
# 2739
__test(int); 
# 2743
template< class > static false_type __test(...); 
# 2745
}; 
# 2747
struct __do_is_nothrow_swappable_impl { 
# 2749
template< class _Tp> static __bool_constant< noexcept(swap(std::declval< _Tp &> (), std::declval< _Tp &> ()))>  __test(int); 
# 2754
template< class > static false_type __test(...); 
# 2756
}; 
# 2758
}
# 2760
template< class _Tp> 
# 2761
struct __is_swappable_impl : public __swappable_details::__do_is_swappable_impl { 
# 2764
typedef __decltype((__test< _Tp> (0))) type; 
# 2765
}; 
# 2767
template< class _Tp> 
# 2768
struct __is_nothrow_swappable_impl : public __swappable_details::__do_is_nothrow_swappable_impl { 
# 2771
typedef __decltype((__test< _Tp> (0))) type; 
# 2772
}; 
# 2774
template< class _Tp> 
# 2775
struct __is_swappable : public __is_swappable_impl< _Tp> ::type { 
# 2777
}; 
# 2779
template< class _Tp> 
# 2780
struct __is_nothrow_swappable : public __is_nothrow_swappable_impl< _Tp> ::type { 
# 2782
}; 
# 2790
template< class _Tp> 
# 2791
struct is_swappable : public __is_swappable_impl< _Tp> ::type { 
# 2794
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 2796
}; 
# 2799
template< class _Tp> 
# 2800
struct is_nothrow_swappable : public __is_nothrow_swappable_impl< _Tp> ::type { 
# 2803
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 2805
}; 
# 2809
template< class _Tp> constexpr bool 
# 2810
is_swappable_v = (is_swappable< _Tp> ::value); 
# 2814
template< class _Tp> constexpr bool 
# 2815
is_nothrow_swappable_v = (is_nothrow_swappable< _Tp> ::value); 
# 2820
namespace __swappable_with_details { 
# 2821
using std::swap;
# 2823
struct __do_is_swappable_with_impl { 
# 2825
template< class _Tp, class _Up, class 
# 2826
 = __decltype((swap(std::declval< _Tp> (), std::declval< _Up> ()))), class 
# 2828
 = __decltype((swap(std::declval< _Up> (), std::declval< _Tp> ())))> static true_type 
# 2825
__test(int); 
# 2831
template< class , class > static false_type __test(...); 
# 2833
}; 
# 2835
struct __do_is_nothrow_swappable_with_impl { 
# 2837
template< class _Tp, class _Up> static __bool_constant< noexcept(swap(std::declval< _Tp> (), std::declval< _Up> ())) && noexcept(swap(std::declval< _Up> (), std::declval< _Tp> ()))>  __test(int); 
# 2844
template< class , class > static false_type __test(...); 
# 2846
}; 
# 2848
}
# 2850
template< class _Tp, class _Up> 
# 2851
struct __is_swappable_with_impl : public __swappable_with_details::__do_is_swappable_with_impl { 
# 2854
typedef __decltype((__test< _Tp, _Up> (0))) type; 
# 2855
}; 
# 2858
template< class _Tp> 
# 2859
struct __is_swappable_with_impl< _Tp &, _Tp &>  : public __swappable_details::__do_is_swappable_impl { 
# 2862
typedef __decltype((__test< _Tp &> (0))) type; 
# 2863
}; 
# 2865
template< class _Tp, class _Up> 
# 2866
struct __is_nothrow_swappable_with_impl : public __swappable_with_details::__do_is_nothrow_swappable_with_impl { 
# 2869
typedef __decltype((__test< _Tp, _Up> (0))) type; 
# 2870
}; 
# 2873
template< class _Tp> 
# 2874
struct __is_nothrow_swappable_with_impl< _Tp &, _Tp &>  : public __swappable_details::__do_is_nothrow_swappable_impl { 
# 2877
typedef __decltype((__test< _Tp &> (0))) type; 
# 2878
}; 
# 2882
template< class _Tp, class _Up> 
# 2883
struct is_swappable_with : public __is_swappable_with_impl< _Tp, _Up> ::type { 
# 2886
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "first template argument must be a complete class or an unbounded array");
# 2888
static_assert((std::__is_complete_or_unbounded(__type_identity< _Up> {})), "second template argument must be a complete class or an unbounded array");
# 2890
}; 
# 2893
template< class _Tp, class _Up> 
# 2894
struct is_nothrow_swappable_with : public __is_nothrow_swappable_with_impl< _Tp, _Up> ::type { 
# 2897
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "first template argument must be a complete class or an unbounded array");
# 2899
static_assert((std::__is_complete_or_unbounded(__type_identity< _Up> {})), "second template argument must be a complete class or an unbounded array");
# 2901
}; 
# 2905
template< class _Tp, class _Up> constexpr bool 
# 2906
is_swappable_with_v = (is_swappable_with< _Tp, _Up> ::value); 
# 2910
template< class _Tp, class _Up> constexpr bool 
# 2911
is_nothrow_swappable_with_v = (is_nothrow_swappable_with< _Tp, _Up> ::value); 
# 2922 "/usr/include/c++/12/type_traits" 3
template< class _Result, class _Ret, bool 
# 2923
 = is_void< _Ret> ::value, class  = void> 
# 2924
struct __is_invocable_impl : public false_type { 
# 2927
using __nothrow_type = false_type; 
# 2928
}; 
# 2931
template< class _Result, class _Ret> 
# 2932
struct __is_invocable_impl< _Result, _Ret, true, __void_t< typename _Result::type> >  : public true_type { 
# 2937
using __nothrow_type = true_type; 
# 2938
}; 
# 2940
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
# 2943
template< class _Result, class _Ret> 
# 2944
struct __is_invocable_impl< _Result, _Ret, false, __void_t< typename _Result::type> >  { 
# 2952
private: static typename _Result::type _S_get() noexcept; 
# 2954
template< class _Tp> static void _S_conv(_Tp) noexcept; 
# 2958
template< class _Tp, bool _Check_Noex = false, class 
# 2959
 = __decltype((_S_conv< _Tp> ((_S_get)()))), bool 
# 2960
_Noex = noexcept(_S_conv< _Tp> ((_S_get)()))> static __bool_constant< _Check_Noex ? _Noex : true>  
# 2958
_S_test(int); 
# 2964
template< class _Tp, bool  = false> static false_type _S_test(...); 
# 2970
public: using type = __decltype((_S_test< _Ret> (1))); 
# 2973
using __nothrow_type = __decltype((_S_test< _Ret, true> (1))); 
# 2974
}; 
#pragma GCC diagnostic pop
# 2977
template< class _Fn, class ..._ArgTypes> 
# 2978
struct __is_invocable : public __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , void> ::type { 
# 2980
}; 
# 2982
template< class _Fn, class _Tp, class ..._Args> constexpr bool 
# 2983
__call_is_nt(__invoke_memfun_ref) 
# 2984
{ 
# 2985
using _Up = typename __inv_unwrap< _Tp> ::type; 
# 2986
return noexcept((std::declval< typename __inv_unwrap< _Tp> ::type> ().*std::declval< _Fn> ())(std::declval< _Args> ()...)); 
# 2988
} 
# 2990
template< class _Fn, class _Tp, class ..._Args> constexpr bool 
# 2991
__call_is_nt(__invoke_memfun_deref) 
# 2992
{ 
# 2993
return noexcept(((*std::declval< _Tp> ()).*std::declval< _Fn> ())(std::declval< _Args> ()...)); 
# 2995
} 
# 2997
template< class _Fn, class _Tp> constexpr bool 
# 2998
__call_is_nt(__invoke_memobj_ref) 
# 2999
{ 
# 3000
using _Up = typename __inv_unwrap< _Tp> ::type; 
# 3001
return noexcept((std::declval< typename __inv_unwrap< _Tp> ::type> ().*std::declval< _Fn> ())); 
# 3002
} 
# 3004
template< class _Fn, class _Tp> constexpr bool 
# 3005
__call_is_nt(__invoke_memobj_deref) 
# 3006
{ 
# 3007
return noexcept(((*std::declval< _Tp> ()).*std::declval< _Fn> ())); 
# 3008
} 
# 3010
template< class _Fn, class ..._Args> constexpr bool 
# 3011
__call_is_nt(__invoke_other) 
# 3012
{ 
# 3013
return noexcept(std::declval< _Fn> ()(std::declval< _Args> ()...)); 
# 3014
} 
# 3016
template< class _Result, class _Fn, class ..._Args> 
# 3017
struct __call_is_nothrow : public __bool_constant< std::__call_is_nt< _Fn, _Args...> (typename _Result::__invoke_type{})>  { 
# 3021
}; 
# 3023
template< class _Fn, class ..._Args> using __call_is_nothrow_ = __call_is_nothrow< __invoke_result< _Fn, _Args...> , _Fn, _Args...> ; 
# 3028
template< class _Fn, class ..._Args> 
# 3029
struct __is_nothrow_invocable : public __and_< __is_invocable< _Fn, _Args...> , __call_is_nothrow_< _Fn, _Args...> > ::type { 
# 3032
}; 
# 3034
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
struct __nonesuchbase { }; 
# 3037
struct __nonesuch : private __nonesuchbase { 
# 3038
~__nonesuch() = delete;
# 3039
__nonesuch(const __nonesuch &) = delete;
# 3040
void operator=(const __nonesuch &) = delete;
# 3041
}; 
#pragma GCC diagnostic pop
# 3049
template< class _Functor, class ..._ArgTypes> 
# 3050
struct invoke_result : public __invoke_result< _Functor, _ArgTypes...>  { 
# 3053
static_assert((std::__is_complete_or_unbounded(__type_identity< _Functor> {})), "_Functor must be a complete class or an unbounded array");
# 3055
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 3058
}; 
# 3061
template< class _Fn, class ..._Args> using invoke_result_t = typename invoke_result< _Fn, _Args...> ::type; 
# 3065
template< class _Fn, class ..._ArgTypes> 
# 3066
struct is_invocable : public __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , void> ::type { 
# 3069
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 3071
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 3074
}; 
# 3077
template< class _Ret, class _Fn, class ..._ArgTypes> 
# 3078
struct is_invocable_r : public __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , _Ret> ::type { 
# 3081
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 3083
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 3086
static_assert((std::__is_complete_or_unbounded(__type_identity< _Ret> {})), "_Ret must be a complete class or an unbounded array");
# 3088
}; 
# 3091
template< class _Fn, class ..._ArgTypes> 
# 3092
struct is_nothrow_invocable : public __and_< __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , void> , __call_is_nothrow_< _Fn, _ArgTypes...> > ::type { 
# 3096
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 3098
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 3101
}; 
# 3104
template< class _Result, class _Ret> using __is_nt_invocable_impl = typename __is_invocable_impl< _Result, _Ret> ::__nothrow_type; 
# 3110
template< class _Ret, class _Fn, class ..._ArgTypes> 
# 3111
struct is_nothrow_invocable_r : public __and_< __is_nt_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , _Ret> , __call_is_nothrow_< _Fn, _ArgTypes...> > ::type { 
# 3115
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 3117
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 3120
static_assert((std::__is_complete_or_unbounded(__type_identity< _Ret> {})), "_Ret must be a complete class or an unbounded array");
# 3122
}; 
# 3141 "/usr/include/c++/12/type_traits" 3
template< class _Tp> constexpr bool 
# 3142
is_void_v = (is_void< _Tp> ::value); 
# 3143
template< class _Tp> constexpr bool 
# 3144
is_null_pointer_v = (is_null_pointer< _Tp> ::value); 
# 3145
template< class _Tp> constexpr bool 
# 3146
is_integral_v = (is_integral< _Tp> ::value); 
# 3147
template< class _Tp> constexpr bool 
# 3148
is_floating_point_v = (is_floating_point< _Tp> ::value); 
# 3149
template< class _Tp> constexpr bool 
# 3150
is_array_v = (is_array< _Tp> ::value); 
# 3151
template< class _Tp> constexpr bool 
# 3152
is_pointer_v = (is_pointer< _Tp> ::value); 
# 3153
template< class _Tp> constexpr bool 
# 3154
is_lvalue_reference_v = (is_lvalue_reference< _Tp> ::value); 
# 3156
template< class _Tp> constexpr bool 
# 3157
is_rvalue_reference_v = (is_rvalue_reference< _Tp> ::value); 
# 3159
template< class _Tp> constexpr bool 
# 3160
is_member_object_pointer_v = (is_member_object_pointer< _Tp> ::value); 
# 3162
template< class _Tp> constexpr bool 
# 3163
is_member_function_pointer_v = (is_member_function_pointer< _Tp> ::value); 
# 3165
template< class _Tp> constexpr bool 
# 3166
is_enum_v = (is_enum< _Tp> ::value); 
# 3167
template< class _Tp> constexpr bool 
# 3168
is_union_v = (is_union< _Tp> ::value); 
# 3169
template< class _Tp> constexpr bool 
# 3170
is_class_v = (is_class< _Tp> ::value); 
# 3171
template< class _Tp> constexpr bool 
# 3172
is_function_v = (is_function< _Tp> ::value); 
# 3173
template< class _Tp> constexpr bool 
# 3174
is_reference_v = (is_reference< _Tp> ::value); 
# 3175
template< class _Tp> constexpr bool 
# 3176
is_arithmetic_v = (is_arithmetic< _Tp> ::value); 
# 3177
template< class _Tp> constexpr bool 
# 3178
is_fundamental_v = (is_fundamental< _Tp> ::value); 
# 3179
template< class _Tp> constexpr bool 
# 3180
is_object_v = (is_object< _Tp> ::value); 
# 3181
template< class _Tp> constexpr bool 
# 3182
is_scalar_v = (is_scalar< _Tp> ::value); 
# 3183
template< class _Tp> constexpr bool 
# 3184
is_compound_v = (is_compound< _Tp> ::value); 
# 3185
template< class _Tp> constexpr bool 
# 3186
is_member_pointer_v = (is_member_pointer< _Tp> ::value); 
# 3187
template< class _Tp> constexpr bool 
# 3188
is_const_v = (is_const< _Tp> ::value); 
# 3189
template< class _Tp> constexpr bool 
# 3190
is_volatile_v = (is_volatile< _Tp> ::value); 
# 3191
template< class _Tp> constexpr bool 
# 3192
is_trivial_v = (is_trivial< _Tp> ::value); 
# 3193
template< class _Tp> constexpr bool 
# 3194
is_trivially_copyable_v = (is_trivially_copyable< _Tp> ::value); 
# 3196
template< class _Tp> constexpr bool 
# 3197
is_standard_layout_v = (is_standard_layout< _Tp> ::value); 
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template< class _Tp> constexpr bool 
# 3202
is_pod_v = (is_pod< _Tp> ::value); 
# 3203
template< class _Tp> 
# 3204
[[__deprecated__]] constexpr bool 
# 3205
is_literal_type_v = (is_literal_type< _Tp> ::value); 
#pragma GCC diagnostic pop
template< class _Tp> constexpr bool 
# 3208
is_empty_v = (is_empty< _Tp> ::value); 
# 3209
template< class _Tp> constexpr bool 
# 3210
is_polymorphic_v = (is_polymorphic< _Tp> ::value); 
# 3211
template< class _Tp> constexpr bool 
# 3212
is_abstract_v = (is_abstract< _Tp> ::value); 
# 3213
template< class _Tp> constexpr bool 
# 3214
is_final_v = (is_final< _Tp> ::value); 
# 3215
template< class _Tp> constexpr bool 
# 3216
is_signed_v = (is_signed< _Tp> ::value); 
# 3217
template< class _Tp> constexpr bool 
# 3218
is_unsigned_v = (is_unsigned< _Tp> ::value); 
# 3219
template< class _Tp, class ..._Args> constexpr bool 
# 3220
is_constructible_v = (is_constructible< _Tp, _Args...> ::value); 
# 3222
template< class _Tp> constexpr bool 
# 3223
is_default_constructible_v = (is_default_constructible< _Tp> ::value); 
# 3225
template< class _Tp> constexpr bool 
# 3226
is_copy_constructible_v = (is_copy_constructible< _Tp> ::value); 
# 3228
template< class _Tp> constexpr bool 
# 3229
is_move_constructible_v = (is_move_constructible< _Tp> ::value); 
# 3231
template< class _Tp, class _Up> constexpr bool 
# 3232
is_assignable_v = (is_assignable< _Tp, _Up> ::value); 
# 3233
template< class _Tp> constexpr bool 
# 3234
is_copy_assignable_v = (is_copy_assignable< _Tp> ::value); 
# 3235
template< class _Tp> constexpr bool 
# 3236
is_move_assignable_v = (is_move_assignable< _Tp> ::value); 
# 3237
template< class _Tp> constexpr bool 
# 3238
is_destructible_v = (is_destructible< _Tp> ::value); 
# 3239
template< class _Tp, class ..._Args> constexpr bool 
# 3240
is_trivially_constructible_v = (is_trivially_constructible< _Tp, _Args...> ::value); 
# 3242
template< class _Tp> constexpr bool 
# 3243
is_trivially_default_constructible_v = (is_trivially_default_constructible< _Tp> ::value); 
# 3245
template< class _Tp> constexpr bool 
# 3246
is_trivially_copy_constructible_v = (is_trivially_copy_constructible< _Tp> ::value); 
# 3248
template< class _Tp> constexpr bool 
# 3249
is_trivially_move_constructible_v = (is_trivially_move_constructible< _Tp> ::value); 
# 3251
template< class _Tp, class _Up> constexpr bool 
# 3252
is_trivially_assignable_v = (is_trivially_assignable< _Tp, _Up> ::value); 
# 3254
template< class _Tp> constexpr bool 
# 3255
is_trivially_copy_assignable_v = (is_trivially_copy_assignable< _Tp> ::value); 
# 3257
template< class _Tp> constexpr bool 
# 3258
is_trivially_move_assignable_v = (is_trivially_move_assignable< _Tp> ::value); 
# 3260
template< class _Tp> constexpr bool 
# 3261
is_trivially_destructible_v = (is_trivially_destructible< _Tp> ::value); 
# 3263
template< class _Tp, class ..._Args> constexpr bool 
# 3264
is_nothrow_constructible_v = (is_nothrow_constructible< _Tp, _Args...> ::value); 
# 3266
template< class _Tp> constexpr bool 
# 3267
is_nothrow_default_constructible_v = (is_nothrow_default_constructible< _Tp> ::value); 
# 3269
template< class _Tp> constexpr bool 
# 3270
is_nothrow_copy_constructible_v = (is_nothrow_copy_constructible< _Tp> ::value); 
# 3272
template< class _Tp> constexpr bool 
# 3273
is_nothrow_move_constructible_v = (is_nothrow_move_constructible< _Tp> ::value); 
# 3275
template< class _Tp, class _Up> constexpr bool 
# 3276
is_nothrow_assignable_v = (is_nothrow_assignable< _Tp, _Up> ::value); 
# 3278
template< class _Tp> constexpr bool 
# 3279
is_nothrow_copy_assignable_v = (is_nothrow_copy_assignable< _Tp> ::value); 
# 3281
template< class _Tp> constexpr bool 
# 3282
is_nothrow_move_assignable_v = (is_nothrow_move_assignable< _Tp> ::value); 
# 3284
template< class _Tp> constexpr bool 
# 3285
is_nothrow_destructible_v = (is_nothrow_destructible< _Tp> ::value); 
# 3287
template< class _Tp> constexpr bool 
# 3288
has_virtual_destructor_v = (has_virtual_destructor< _Tp> ::value); 
# 3290
template< class _Tp> constexpr size_t 
# 3291
alignment_of_v = (alignment_of< _Tp> ::value); 
# 3292
template< class _Tp> constexpr size_t 
# 3293
rank_v = (rank< _Tp> ::value); 
# 3294
template< class _Tp, unsigned _Idx = 0U> constexpr size_t 
# 3295
extent_v = (extent< _Tp, _Idx> ::value); 
# 3297
template< class _Tp, class _Up> constexpr bool 
# 3298
is_same_v = __is_same(_Tp, _Up); 
# 3303
template< class _Base, class _Derived> constexpr bool 
# 3304
is_base_of_v = (is_base_of< _Base, _Derived> ::value); 
# 3305
template< class _From, class _To> constexpr bool 
# 3306
is_convertible_v = (is_convertible< _From, _To> ::value); 
# 3307
template< class _Fn, class ..._Args> constexpr bool 
# 3308
is_invocable_v = (is_invocable< _Fn, _Args...> ::value); 
# 3309
template< class _Fn, class ..._Args> constexpr bool 
# 3310
is_nothrow_invocable_v = (is_nothrow_invocable< _Fn, _Args...> ::value); 
# 3312
template< class _Ret, class _Fn, class ..._Args> constexpr bool 
# 3313
is_invocable_r_v = (is_invocable_r< _Ret, _Fn, _Args...> ::value); 
# 3315
template< class _Ret, class _Fn, class ..._Args> constexpr bool 
# 3316
is_nothrow_invocable_r_v = (is_nothrow_invocable_r< _Ret, _Fn, _Args...> ::value); 
# 3324
template< class _Tp> 
# 3325
struct has_unique_object_representations : public bool_constant< __has_unique_object_representations(remove_cv_t< remove_all_extents_t< _Tp> > )>  { 
# 3330
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 3332
}; 
# 3335
template< class _Tp> constexpr bool 
# 3336
has_unique_object_representations_v = (has_unique_object_representations< _Tp> ::value); 
# 3344
template< class _Tp> 
# 3345
struct is_aggregate : public bool_constant< __is_aggregate(remove_cv_t< _Tp> )>  { 
# 3347
}; 
# 3350
template< class _Tp> constexpr bool 
# 3351
is_aggregate_v = (is_aggregate< _Tp> ::value); 
# 3726 "/usr/include/c++/12/type_traits" 3
}
# 38 "/usr/include/c++/12/bits/move.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 47
template< class _Tp> constexpr _Tp *
# 49
__addressof(_Tp &__r) noexcept 
# 50
{ return __builtin_addressof(__r); } 
# 55
}
# 59
namespace std __attribute((__visibility__("default"))) { 
# 74 "/usr/include/c++/12/bits/move.h" 3
template< class _Tp> 
# 75
[[__nodiscard__]] constexpr _Tp &&
# 77
forward(typename remove_reference< _Tp> ::type &__t) noexcept 
# 78
{ return static_cast< _Tp &&>(__t); } 
# 86
template< class _Tp> 
# 87
[[__nodiscard__]] constexpr _Tp &&
# 89
forward(typename remove_reference< _Tp> ::type &&__t) noexcept 
# 90
{ 
# 91
static_assert((!std::template is_lvalue_reference< _Tp> ::value), "std::forward must not be used to convert an rvalue to an lvalue");
# 93
return static_cast< _Tp &&>(__t); 
# 94
} 
# 101
template< class _Tp> 
# 102
[[__nodiscard__]] constexpr typename remove_reference< _Tp> ::type &&
# 104
move(_Tp &&__t) noexcept 
# 105
{ return static_cast< typename remove_reference< _Tp> ::type &&>(__t); } 
# 108
template< class _Tp> 
# 109
struct __move_if_noexcept_cond : public __and_< __not_< is_nothrow_move_constructible< _Tp> > , is_copy_constructible< _Tp> > ::type { 
# 111
}; 
# 121 "/usr/include/c++/12/bits/move.h" 3
template< class _Tp> 
# 122
[[__nodiscard__]] constexpr __conditional_t< __move_if_noexcept_cond< _Tp> ::value, const _Tp &, _Tp &&>  
# 125
move_if_noexcept(_Tp &__x) noexcept 
# 126
{ return std::move(__x); } 
# 142 "/usr/include/c++/12/bits/move.h" 3
template< class _Tp> 
# 143
[[__nodiscard__]] constexpr _Tp *
# 145
addressof(_Tp &__r) noexcept 
# 146
{ return std::__addressof(__r); } 
# 150
template < typename _Tp >
    const _Tp * addressof ( const _Tp && ) = delete;
# 154
template< class _Tp, class _Up = _Tp> inline _Tp 
# 157
__exchange(_Tp &__obj, _Up &&__new_val) 
# 158
{ 
# 159
_Tp __old_val = std::move(__obj); 
# 160
__obj = std::forward< _Up> (__new_val); 
# 161
return __old_val; 
# 162
} 
# 186 "/usr/include/c++/12/bits/move.h" 3
template< class _Tp> inline typename enable_if< __and_< __not_< __is_tuple_like< _Tp> > , is_move_constructible< _Tp> , is_move_assignable< _Tp> > ::value> ::type 
# 196
swap(_Tp &__a, _Tp &__b) noexcept(__and_< is_nothrow_move_constructible< _Tp> , is_nothrow_move_assignable< _Tp> > ::value) 
# 199
{ 
# 204
_Tp __tmp = std::move(__a); 
# 205
__a = std::move(__b); 
# 206
__b = std::move(__tmp); 
# 207
} 
# 212
template< class _Tp, size_t _Nm> inline typename enable_if< __is_swappable< _Tp> ::value> ::type 
# 220
swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm]) noexcept(__is_nothrow_swappable< _Tp> ::value) 
# 222
{ 
# 223
for (size_t __n = (0); __n < _Nm; ++__n) { 
# 224
swap(__a[__n], __b[__n]); }  
# 225
} 
# 229
}
# 43 "/usr/include/c++/12/bits/utility.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 48
template< class _Tp> struct tuple_size; 
# 55
template< class _Tp, class 
# 56
_Up = typename remove_cv< _Tp> ::type, class 
# 57
 = typename enable_if< is_same< _Tp, _Up> ::value> ::type, size_t 
# 58
 = tuple_size< _Tp> ::value> using __enable_if_has_tuple_size = _Tp; 
# 61
template< class _Tp> 
# 62
struct tuple_size< const __enable_if_has_tuple_size< _Tp> >  : public std::tuple_size< _Tp>  { 
# 63
}; 
# 65
template< class _Tp> 
# 66
struct tuple_size< volatile __enable_if_has_tuple_size< _Tp> >  : public std::tuple_size< _Tp>  { 
# 67
}; 
# 69
template< class _Tp> 
# 70
struct tuple_size< const volatile __enable_if_has_tuple_size< _Tp> >  : public std::tuple_size< _Tp>  { 
# 71
}; 
# 74
template< class _Tp> constexpr size_t 
# 75
tuple_size_v = (tuple_size< _Tp> ::value); 
# 79
template< size_t __i, class _Tp> struct tuple_element; 
# 83
template< size_t __i, class _Tp> using __tuple_element_t = typename tuple_element< __i, _Tp> ::type; 
# 86
template< size_t __i, class _Tp> 
# 87
struct tuple_element< __i, const _Tp>  { 
# 89
typedef typename add_const< __tuple_element_t< __i, _Tp> > ::type type; 
# 90
}; 
# 92
template< size_t __i, class _Tp> 
# 93
struct tuple_element< __i, volatile _Tp>  { 
# 95
typedef typename add_volatile< __tuple_element_t< __i, _Tp> > ::type type; 
# 96
}; 
# 98
template< size_t __i, class _Tp> 
# 99
struct tuple_element< __i, const volatile _Tp>  { 
# 101
typedef typename add_cv< __tuple_element_t< __i, _Tp> > ::type type; 
# 102
}; 
# 108
template< class _Tp, class ..._Types> constexpr size_t 
# 110
__find_uniq_type_in_pack() 
# 111
{ 
# 112
constexpr size_t __sz = sizeof...(_Types); 
# 113
constexpr bool __found[__sz] = {__is_same(_Tp, _Types)...}; 
# 114
size_t __n = __sz; 
# 115
for (size_t __i = (0); __i < __sz; ++__i) 
# 116
{ 
# 117
if (__found[__i]) 
# 118
{ 
# 119
if (__n < __sz) { 
# 120
return __sz; }  
# 121
__n = __i; 
# 122
}  
# 123
}  
# 124
return __n; 
# 125
} 
# 134 "/usr/include/c++/12/bits/utility.h" 3
template< size_t __i, class _Tp> using tuple_element_t = typename tuple_element< __i, _Tp> ::type; 
# 140
template< size_t ..._Indexes> struct _Index_tuple { }; 
# 143
template< size_t _Num> 
# 144
struct _Build_index_tuple { 
# 154 "/usr/include/c++/12/bits/utility.h" 3
using __type = _Index_tuple< __integer_pack(_Num)...> ; 
# 156
}; 
# 163
template< class _Tp, _Tp ..._Idx> 
# 164
struct integer_sequence { 
# 169
typedef _Tp value_type; 
# 170
static constexpr size_t size() noexcept { return sizeof...(_Idx); } 
# 171
}; 
# 174
template< class _Tp, _Tp _Num> using make_integer_sequence = integer_sequence< _Tp, __integer_pack((_Tp)_Num)...> ; 
# 183
template< size_t ..._Idx> using index_sequence = integer_sequence< unsigned long, _Idx...> ; 
# 187
template< size_t _Num> using make_index_sequence = make_integer_sequence< unsigned long, _Num> ; 
# 191
template< class ..._Types> using index_sequence_for = make_index_sequence< sizeof...(_Types)> ; 
# 196
struct in_place_t { 
# 197
explicit in_place_t() = default;
# 198
}; 
# 200
constexpr inline in_place_t in_place{}; 
# 202
template< class _Tp> struct in_place_type_t { 
# 204
explicit in_place_type_t() = default;
# 205
}; 
# 207
template< class _Tp> constexpr in_place_type_t< _Tp>  
# 208
in_place_type{}; 
# 210
template< size_t _Idx> struct in_place_index_t { 
# 212
explicit in_place_index_t() = default;
# 213
}; 
# 215
template< size_t _Idx> constexpr in_place_index_t< _Idx>  
# 216
in_place_index{}; 
# 218
template< class > constexpr bool 
# 219
__is_in_place_type_v = false; 
# 221
template< class _Tp> constexpr bool 
# 222
__is_in_place_type_v< in_place_type_t< _Tp> >  = true; 
# 224
template< class _Tp> using __is_in_place_type = bool_constant< __is_in_place_type_v< _Tp> > ; 
# 230
template< size_t _Np, class ..._Types> 
# 231
struct _Nth_type { 
# 232
}; 
# 234
template< class _Tp0, class ..._Rest> 
# 235
struct _Nth_type< 0, _Tp0, _Rest...>  { 
# 236
using type = _Tp0; }; 
# 238
template< class _Tp0, class _Tp1, class ..._Rest> 
# 239
struct _Nth_type< 1, _Tp0, _Tp1, _Rest...>  { 
# 240
using type = _Tp1; }; 
# 242
template< class _Tp0, class _Tp1, class _Tp2, class ..._Rest> 
# 243
struct _Nth_type< 2, _Tp0, _Tp1, _Tp2, _Rest...>  { 
# 244
using type = _Tp2; }; 
# 246
template< size_t _Np, class _Tp0, class _Tp1, class _Tp2, class ...
# 247
_Rest> 
# 251
struct _Nth_type< _Np, _Tp0, _Tp1, _Tp2, _Rest...>  : public std::_Nth_type< _Np - (3), _Rest...>  { 
# 253
}; 
# 256
template< class _Tp0, class _Tp1, class ..._Rest> 
# 257
struct _Nth_type< 0, _Tp0, _Tp1, _Rest...>  { 
# 258
using type = _Tp0; }; 
# 260
template< class _Tp0, class _Tp1, class _Tp2, class ..._Rest> 
# 261
struct _Nth_type< 0, _Tp0, _Tp1, _Tp2, _Rest...>  { 
# 262
using type = _Tp0; }; 
# 264
template< class _Tp0, class _Tp1, class _Tp2, class ..._Rest> 
# 265
struct _Nth_type< 1, _Tp0, _Tp1, _Tp2, _Rest...>  { 
# 266
using type = _Tp1; }; 
# 270
}
# 69 "/usr/include/c++/12/bits/stl_pair.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 80 "/usr/include/c++/12/bits/stl_pair.h" 3
struct piecewise_construct_t { explicit piecewise_construct_t() = default;}; 
# 83
constexpr inline piecewise_construct_t piecewise_construct = piecewise_construct_t(); 
# 89
template< class ...> class tuple; 
# 92
template< size_t ...> struct _Index_tuple; 
# 101
template< bool , class _T1, class _T2> 
# 102
struct _PCC { 
# 104
template< class _U1, class _U2> static constexpr bool 
# 105
_ConstructiblePair() 
# 106
{ 
# 107
return __and_< is_constructible< _T1, const _U1 &> , is_constructible< _T2, const _U2 &> > ::value; 
# 109
} 
# 111
template< class _U1, class _U2> static constexpr bool 
# 112
_ImplicitlyConvertiblePair() 
# 113
{ 
# 114
return __and_< is_convertible< const _U1 &, _T1> , is_convertible< const _U2 &, _T2> > ::value; 
# 116
} 
# 118
template< class _U1, class _U2> static constexpr bool 
# 119
_MoveConstructiblePair() 
# 120
{ 
# 121
return __and_< is_constructible< _T1, _U1 &&> , is_constructible< _T2, _U2 &&> > ::value; 
# 123
} 
# 125
template< class _U1, class _U2> static constexpr bool 
# 126
_ImplicitlyMoveConvertiblePair() 
# 127
{ 
# 128
return __and_< is_convertible< _U1 &&, _T1> , is_convertible< _U2 &&, _T2> > ::value; 
# 130
} 
# 131
}; 
# 133
template< class _T1, class _T2> 
# 134
struct _PCC< false, _T1, _T2>  { 
# 136
template< class _U1, class _U2> static constexpr bool 
# 137
_ConstructiblePair() 
# 138
{ 
# 139
return false; 
# 140
} 
# 142
template< class _U1, class _U2> static constexpr bool 
# 143
_ImplicitlyConvertiblePair() 
# 144
{ 
# 145
return false; 
# 146
} 
# 148
template< class _U1, class _U2> static constexpr bool 
# 149
_MoveConstructiblePair() 
# 150
{ 
# 151
return false; 
# 152
} 
# 154
template< class _U1, class _U2> static constexpr bool 
# 155
_ImplicitlyMoveConvertiblePair() 
# 156
{ 
# 157
return false; 
# 158
} 
# 159
}; 
# 163
template< class _U1, class _U2> class __pair_base { 
# 166
template< class _T1, class _T2> friend struct pair; 
# 167
__pair_base() = default;
# 168
~__pair_base() = default;
# 169
__pair_base(const __pair_base &) = default;
# 170
__pair_base &operator=(const __pair_base &) = delete;
# 172
}; 
# 186 "/usr/include/c++/12/bits/stl_pair.h" 3
template< class _T1, class _T2> 
# 187
struct pair : public __pair_base< _T1, _T2>  { 
# 190
typedef _T1 first_type; 
# 191
typedef _T2 second_type; 
# 193
_T1 first; 
# 194
_T2 second; 
# 197
constexpr pair(const pair &) = default;
# 198
constexpr pair(pair &&) = default;
# 200
template< class ..._Args1, class ..._Args2> pair(std::piecewise_construct_t, tuple< _Args1...> , tuple< _Args2...> ); 
# 206
void swap(pair &__p) noexcept(__and_< __is_nothrow_swappable< _T1> , __is_nothrow_swappable< _T2> > ::value) 
# 209
{ 
# 210
using std::swap;
# 211
swap(first, __p.first); 
# 212
swap(second, __p.second); 
# 213
} 
# 216
private: template< class ..._Args1, std::size_t ..._Indexes1, class ...
# 217
_Args2, std::size_t ..._Indexes2> 
# 216
pair(tuple< _Args1...>  &, tuple< _Args2...>  &, _Index_tuple< _Indexes1...> , _Index_tuple< _Indexes2...> ); 
# 386 "/usr/include/c++/12/bits/stl_pair.h" 3
public: 
# 380
template< class _U1 = _T1, class 
# 381
_U2 = _T2, typename enable_if< __and_< __is_implicitly_default_constructible< _U1> , __is_implicitly_default_constructible< _U2> > ::value, bool> ::type 
# 385
 = true> constexpr 
# 386
pair() : first(), second() 
# 387
{ } 
# 389
template< class _U1 = _T1, class 
# 390
_U2 = _T2, typename enable_if< __and_< is_default_constructible< _U1> , is_default_constructible< _U2> , __not_< __and_< __is_implicitly_default_constructible< _U1> , __is_implicitly_default_constructible< _U2> > > > ::value, bool> ::type 
# 397
 = false> constexpr explicit 
# 398
pair() : first(), second() 
# 399
{ } 
# 403
using _PCCP = _PCC< true, _T1, _T2> ; 
# 407
template< class _U1 = _T1, class _U2 = _T2, typename enable_if< _PCC< true, _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && _PCC< true, _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> (), bool> ::type 
# 412
 = true> constexpr 
# 413
pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) 
# 414
{ } 
# 417
template< class _U1 = _T1, class _U2 = _T2, typename enable_if< _PCC< true, _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && (!_PCC< true, _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> ()), bool> ::type 
# 422
 = false> constexpr explicit 
# 423
pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) 
# 424
{ } 
# 428
template< class _U1, class _U2> using _PCCFP = _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ; 
# 434
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> (), bool> ::type 
# 439
 = true> constexpr 
# 440
pair(const std::pair< _U1, _U2>  &__p) : first((__p.first)), second((__p.second)) 
# 441
{ } 
# 443
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && (!_PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> ()), bool> ::type 
# 448
 = false> constexpr explicit 
# 449
pair(const std::pair< _U1, _U2>  &__p) : first((__p.first)), second((__p.second)) 
# 450
{ } 
# 466 "/usr/include/c++/12/bits/stl_pair.h" 3
private: struct __zero_as_null_pointer_constant { 
# 468
__zero_as_null_pointer_constant(int (__zero_as_null_pointer_constant::*)) 
# 469
{ } 
# 470
template < typename _Tp,
   typename = __enable_if_t < is_null_pointer < _Tp > :: value > >
 __zero_as_null_pointer_constant ( _Tp ) = delete;
# 473
}; 
# 489
public: 
# 480
template< class _U1, std::__enable_if_t< __and_< __not_< is_reference< _U1> > , is_pointer< _T2> , is_constructible< _T1, _U1> , __not_< is_constructible< _T1, const _U1 &> > , is_convertible< _U1, _T1> > ::value, bool>  
# 486
 = true> 
# 487
__attribute((__deprecated__("use \'nullptr\' instead of \'0\' to initialize std::pair of move-only type and pointer"))) constexpr 
# 489
pair(_U1 &&__x, __zero_as_null_pointer_constant, ...) : first(std::forward< _U1> (__x)), second(nullptr) 
# 490
{ } 
# 492
template< class _U1, std::__enable_if_t< __and_< __not_< is_reference< _U1> > , is_pointer< _T2> , is_constructible< _T1, _U1> , __not_< is_constructible< _T1, const _U1 &> > , __not_< is_convertible< _U1, _T1> > > ::value, bool>  
# 498
 = false> 
# 499
__attribute((__deprecated__("use \'nullptr\' instead of \'0\' to initialize std::pair of move-only type and pointer"))) constexpr explicit 
# 501
pair(_U1 &&__x, __zero_as_null_pointer_constant, ...) : first(std::forward< _U1> (__x)), second(nullptr) 
# 502
{ } 
# 504
template< class _U2, std::__enable_if_t< __and_< is_pointer< _T1> , __not_< is_reference< _U2> > , is_constructible< _T2, _U2> , __not_< is_constructible< _T2, const _U2 &> > , is_convertible< _U2, _T2> > ::value, bool>  
# 510
 = true> 
# 511
__attribute((__deprecated__("use \'nullptr\' instead of \'0\' to initialize std::pair of move-only type and pointer"))) constexpr 
# 513
pair(__zero_as_null_pointer_constant, _U2 &&__y, ...) : first(nullptr), second(std::forward< _U2> (__y)) 
# 514
{ } 
# 516
template< class _U2, std::__enable_if_t< __and_< is_pointer< _T1> , __not_< is_reference< _U2> > , is_constructible< _T2, _U2> , __not_< is_constructible< _T2, const _U2 &> > , __not_< is_convertible< _U2, _T2> > > ::value, bool>  
# 522
 = false> 
# 523
__attribute((__deprecated__("use \'nullptr\' instead of \'0\' to initialize std::pair of move-only type and pointer"))) constexpr explicit 
# 525
pair(__zero_as_null_pointer_constant, _U2 &&__y, ...) : first(nullptr), second(std::forward< _U2> (__y)) 
# 526
{ } 
# 530
template< class _U1, class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && _PCC< true, _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> (), bool> ::type 
# 535
 = true> constexpr 
# 536
pair(_U1 &&__x, _U2 &&__y) : first(std::forward< _U1> (__x)), second(std::forward< _U2> (__y)) 
# 537
{ } 
# 539
template< class _U1, class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && (!_PCC< true, _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> ()), bool> ::type 
# 544
 = false> constexpr explicit 
# 545
pair(_U1 &&__x, _U2 &&__y) : first(std::forward< _U1> (__x)), second(std::forward< _U2> (__y)) 
# 546
{ } 
# 549
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> (), bool> ::type 
# 554
 = true> constexpr 
# 555
pair(std::pair< _U1, _U2>  &&__p) : first(std::forward< _U1> ((__p.first))), second(std::forward< _U2> ((__p.second))) 
# 557
{ } 
# 559
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && (!_PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> ()), bool> ::type 
# 564
 = false> constexpr explicit 
# 565
pair(std::pair< _U1, _U2>  &&__p) : first(std::forward< _U1> ((__p.first))), second(std::forward< _U2> ((__p.second))) 
# 567
{ } 
# 570
pair &operator=(std::__conditional_t< __and_< is_copy_assignable< _T1> , is_copy_assignable< _T2> > ::value, const pair &, const std::__nonesuch &>  
# 572
__p) 
# 573
{ 
# 574
(first) = (__p.first); 
# 575
(second) = (__p.second); 
# 576
return *this; 
# 577
} 
# 580
pair &operator=(std::__conditional_t< __and_< is_move_assignable< _T1> , is_move_assignable< _T2> > ::value, pair &&, std::__nonesuch &&>  
# 582
__p) noexcept(__and_< is_nothrow_move_assignable< _T1> , is_nothrow_move_assignable< _T2> > ::value) 
# 585
{ 
# 586
(first) = std::forward< first_type> ((__p.first)); 
# 587
(second) = std::forward< second_type> ((__p.second)); 
# 588
return *this; 
# 589
} 
# 591
template< class _U1, class _U2> typename enable_if< __and_< is_assignable< _T1 &, const _U1 &> , is_assignable< _T2 &, const _U2 &> > ::value, pair &> ::type 
# 595
operator=(const std::pair< _U1, _U2>  &__p) 
# 596
{ 
# 597
(first) = (__p.first); 
# 598
(second) = (__p.second); 
# 599
return *this; 
# 600
} 
# 602
template< class _U1, class _U2> typename enable_if< __and_< is_assignable< _T1 &, _U1 &&> , is_assignable< _T2 &, _U2 &&> > ::value, pair &> ::type 
# 606
operator=(std::pair< _U1, _U2>  &&__p) 
# 607
{ 
# 608
(first) = std::forward< _U1> ((__p.first)); 
# 609
(second) = std::forward< _U2> ((__p.second)); 
# 610
return *this; 
# 611
} 
# 631 "/usr/include/c++/12/bits/stl_pair.h" 3
}; 
# 636
template< class _T1, class _T2> pair(_T1, _T2)->pair< _T1, _T2> ; 
# 640
template< class _T1, class _T2> constexpr bool 
# 642
operator==(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 643
{ return ((__x.first) == (__y.first)) && ((__x.second) == (__y.second)); } 
# 663 "/usr/include/c++/12/bits/stl_pair.h" 3
template< class _T1, class _T2> constexpr bool 
# 665
operator<(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 666
{ return ((__x.first) < (__y.first)) || ((!((__y.first) < (__x.first))) && ((__x.second) < (__y.second))); 
# 667
} 
# 670
template< class _T1, class _T2> constexpr bool 
# 672
operator!=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 673
{ return !(__x == __y); } 
# 676
template< class _T1, class _T2> constexpr bool 
# 678
operator>(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 679
{ return __y < __x; } 
# 682
template< class _T1, class _T2> constexpr bool 
# 684
operator<=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 685
{ return !(__y < __x); } 
# 688
template< class _T1, class _T2> constexpr bool 
# 690
operator>=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 691
{ return !(__x < __y); } 
# 700 "/usr/include/c++/12/bits/stl_pair.h" 3
template< class _T1, class _T2> inline typename enable_if< __and_< __is_swappable< _T1> , __is_swappable< _T2> > ::value> ::type 
# 709
swap(pair< _T1, _T2>  &__x, pair< _T1, _T2>  &__y) noexcept(noexcept(__x.swap(__y))) 
# 711
{ __x.swap(__y); } 
# 714
template < typename _T1, typename _T2 >
    typename enable_if < ! __and_ < __is_swappable < _T1 >,
          __is_swappable < _T2 > > :: value > :: type
    swap ( pair < _T1, _T2 > &, pair < _T1, _T2 > & ) = delete;
# 740 "/usr/include/c++/12/bits/stl_pair.h" 3
template< class _T1, class _T2> constexpr pair< typename __decay_and_strip< _T1> ::__type, typename __decay_and_strip< _T2> ::__type>  
# 743
make_pair(_T1 &&__x, _T2 &&__y) 
# 744
{ 
# 745
typedef typename __decay_and_strip< _T1> ::__type __ds_type1; 
# 746
typedef typename __decay_and_strip< _T2> ::__type __ds_type2; 
# 747
typedef pair< typename __decay_and_strip< _T1> ::__type, typename __decay_and_strip< _T2> ::__type>  __pair_type; 
# 748
return __pair_type(std::forward< _T1> (__x), std::forward< _T2> (__y)); 
# 749
} 
# 763 "/usr/include/c++/12/bits/stl_pair.h" 3
template< class _T1, class _T2> 
# 764
struct __is_tuple_like_impl< pair< _T1, _T2> >  : public true_type { 
# 765
}; 
# 769
template< class _Tp1, class _Tp2> 
# 770
struct tuple_size< pair< _Tp1, _Tp2> >  : public integral_constant< unsigned long, 2UL>  { 
# 771
}; 
# 774
template< class _Tp1, class _Tp2> 
# 775
struct tuple_element< 0, pair< _Tp1, _Tp2> >  { 
# 776
typedef _Tp1 type; }; 
# 779
template< class _Tp1, class _Tp2> 
# 780
struct tuple_element< 1, pair< _Tp1, _Tp2> >  { 
# 781
typedef _Tp2 type; }; 
# 784
template< class _Tp1, class _Tp2> constexpr size_t 
# 785
tuple_size_v< pair< _Tp1, _Tp2> >  = (2); 
# 787
template< class _Tp1, class _Tp2> constexpr size_t 
# 788
tuple_size_v< const pair< _Tp1, _Tp2> >  = (2); 
# 790
template< class _Tp> constexpr bool 
# 791
__is_pair = false; 
# 793
template< class _Tp, class _Up> constexpr bool 
# 794
__is_pair< pair< _Tp, _Up> >  = true; 
# 796
template< class _Tp, class _Up> constexpr bool 
# 797
__is_pair< const pair< _Tp, _Up> >  = true; 
# 801
template< size_t _Int> struct __pair_get; 
# 805
template<> struct __pair_get< 0UL>  { 
# 807
template< class _Tp1, class _Tp2> static constexpr _Tp1 &
# 809
__get(pair< _Tp1, _Tp2>  &__pair) noexcept 
# 810
{ return __pair.first; } 
# 812
template< class _Tp1, class _Tp2> static constexpr _Tp1 &&
# 814
__move_get(pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 815
{ return std::forward< _Tp1> ((__pair.first)); } 
# 817
template< class _Tp1, class _Tp2> static constexpr const _Tp1 &
# 819
__const_get(const pair< _Tp1, _Tp2>  &__pair) noexcept 
# 820
{ return __pair.first; } 
# 822
template< class _Tp1, class _Tp2> static constexpr const _Tp1 &&
# 824
__const_move_get(const pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 825
{ return std::forward< const _Tp1> ((__pair.first)); } 
# 826
}; 
# 829
template<> struct __pair_get< 1UL>  { 
# 831
template< class _Tp1, class _Tp2> static constexpr _Tp2 &
# 833
__get(pair< _Tp1, _Tp2>  &__pair) noexcept 
# 834
{ return __pair.second; } 
# 836
template< class _Tp1, class _Tp2> static constexpr _Tp2 &&
# 838
__move_get(pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 839
{ return std::forward< _Tp2> ((__pair.second)); } 
# 841
template< class _Tp1, class _Tp2> static constexpr const _Tp2 &
# 843
__const_get(const pair< _Tp1, _Tp2>  &__pair) noexcept 
# 844
{ return __pair.second; } 
# 846
template< class _Tp1, class _Tp2> static constexpr const _Tp2 &&
# 848
__const_move_get(const pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 849
{ return std::forward< const _Tp2> ((__pair.second)); } 
# 850
}; 
# 857
template< size_t _Int, class _Tp1, class _Tp2> constexpr typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &
# 859
get(pair< _Tp1, _Tp2>  &__in) noexcept 
# 860
{ return __pair_get< _Int> ::__get(__in); } 
# 862
template< size_t _Int, class _Tp1, class _Tp2> constexpr typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &&
# 864
get(pair< _Tp1, _Tp2>  &&__in) noexcept 
# 865
{ return __pair_get< _Int> ::__move_get(std::move(__in)); } 
# 867
template< size_t _Int, class _Tp1, class _Tp2> constexpr const typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &
# 869
get(const pair< _Tp1, _Tp2>  &__in) noexcept 
# 870
{ return __pair_get< _Int> ::__const_get(__in); } 
# 872
template< size_t _Int, class _Tp1, class _Tp2> constexpr const typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &&
# 874
get(const pair< _Tp1, _Tp2>  &&__in) noexcept 
# 875
{ return __pair_get< _Int> ::__const_move_get(std::move(__in)); } 
# 881
template< class _Tp, class _Up> constexpr _Tp &
# 883
get(pair< _Tp, _Up>  &__p) noexcept 
# 884
{ return __p.first; } 
# 886
template< class _Tp, class _Up> constexpr const _Tp &
# 888
get(const pair< _Tp, _Up>  &__p) noexcept 
# 889
{ return __p.first; } 
# 891
template< class _Tp, class _Up> constexpr _Tp &&
# 893
get(pair< _Tp, _Up>  &&__p) noexcept 
# 894
{ return std::move((__p.first)); } 
# 896
template< class _Tp, class _Up> constexpr const _Tp &&
# 898
get(const pair< _Tp, _Up>  &&__p) noexcept 
# 899
{ return std::move((__p.first)); } 
# 901
template< class _Tp, class _Up> constexpr _Tp &
# 903
get(pair< _Up, _Tp>  &__p) noexcept 
# 904
{ return __p.second; } 
# 906
template< class _Tp, class _Up> constexpr const _Tp &
# 908
get(const pair< _Up, _Tp>  &__p) noexcept 
# 909
{ return __p.second; } 
# 911
template< class _Tp, class _Up> constexpr _Tp &&
# 913
get(pair< _Up, _Tp>  &&__p) noexcept 
# 914
{ return std::move((__p.second)); } 
# 916
template< class _Tp, class _Up> constexpr const _Tp &&
# 918
get(const pair< _Up, _Tp>  &&__p) noexcept 
# 919
{ return std::move((__p.second)); } 
# 926
}
# 74 "/usr/include/c++/12/bits/stl_iterator_base_types.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 93 "/usr/include/c++/12/bits/stl_iterator_base_types.h" 3
struct input_iterator_tag { }; 
# 96
struct output_iterator_tag { }; 
# 99
struct forward_iterator_tag : public input_iterator_tag { }; 
# 103
struct bidirectional_iterator_tag : public forward_iterator_tag { }; 
# 107
struct random_access_iterator_tag : public bidirectional_iterator_tag { }; 
# 125 "/usr/include/c++/12/bits/stl_iterator_base_types.h" 3
template< class _Category, class _Tp, class _Distance = ptrdiff_t, class 
# 126
_Pointer = _Tp *, class _Reference = _Tp &> 
# 127
struct [[__deprecated__]] iterator { 
# 130
typedef _Category iterator_category; 
# 132
typedef _Tp value_type; 
# 134
typedef _Distance difference_type; 
# 136
typedef _Pointer pointer; 
# 138
typedef _Reference reference; 
# 139
}; 
# 149 "/usr/include/c++/12/bits/stl_iterator_base_types.h" 3
template< class _Iterator> struct iterator_traits; 
# 155
template< class _Iterator, class  = __void_t< > > 
# 156
struct __iterator_traits { }; 
# 160
template< class _Iterator> 
# 161
struct __iterator_traits< _Iterator, __void_t< typename _Iterator::iterator_category, typename _Iterator::value_type, typename _Iterator::difference_type, typename _Iterator::pointer, typename _Iterator::reference> >  { 
# 168
typedef typename _Iterator::iterator_category iterator_category; 
# 169
typedef typename _Iterator::value_type value_type; 
# 170
typedef typename _Iterator::difference_type difference_type; 
# 171
typedef typename _Iterator::pointer pointer; 
# 172
typedef typename _Iterator::reference reference; 
# 173
}; 
# 176
template< class _Iterator> 
# 177
struct iterator_traits : public __iterator_traits< _Iterator>  { 
# 178
}; 
# 209 "/usr/include/c++/12/bits/stl_iterator_base_types.h" 3
template< class _Tp> 
# 210
struct iterator_traits< _Tp *>  { 
# 212
typedef random_access_iterator_tag iterator_category; 
# 213
typedef _Tp value_type; 
# 214
typedef ptrdiff_t difference_type; 
# 215
typedef _Tp *pointer; 
# 216
typedef _Tp &reference; 
# 217
}; 
# 220
template< class _Tp> 
# 221
struct iterator_traits< const _Tp *>  { 
# 223
typedef random_access_iterator_tag iterator_category; 
# 224
typedef _Tp value_type; 
# 225
typedef ptrdiff_t difference_type; 
# 226
typedef const _Tp *pointer; 
# 227
typedef const _Tp &reference; 
# 228
}; 
# 235
template< class _Iter> constexpr typename iterator_traits< _Iter> ::iterator_category 
# 238
__iterator_category(const _Iter &) 
# 239
{ return typename iterator_traits< _Iter> ::iterator_category(); } 
# 244
template< class _Iter> using __iterator_category_t = typename iterator_traits< _Iter> ::iterator_category; 
# 248
template< class _InIter> using _RequireInputIter = __enable_if_t< is_convertible< __iterator_category_t< _InIter> , input_iterator_tag> ::value> ; 
# 253
template< class _It, class 
# 254
_Cat = __iterator_category_t< _It> > 
# 255
struct __is_random_access_iter : public is_base_of< random_access_iterator_tag, _Cat>  { 
# 258
typedef is_base_of< std::random_access_iterator_tag, _Cat>  _Base; 
# 259
enum { __value = is_base_of< std::random_access_iterator_tag, _Cat> ::value}; 
# 260
}; 
# 269
}
# 68 "/usr/include/c++/12/bits/stl_iterator_base_funcs.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 74
template< class > struct _List_iterator; 
# 75
template< class > struct _List_const_iterator; 
# 78
template< class _InputIterator> constexpr typename iterator_traits< _InputIterator> ::difference_type 
# 81
__distance(_InputIterator __first, _InputIterator __last, input_iterator_tag) 
# 83
{ 
# 87
typename iterator_traits< _InputIterator> ::difference_type __n = (0); 
# 88
while (__first != __last) 
# 89
{ 
# 90
++__first; 
# 91
++__n; 
# 92
}  
# 93
return __n; 
# 94
} 
# 96
template< class _RandomAccessIterator> constexpr typename iterator_traits< _RandomAccessIterator> ::difference_type 
# 99
__distance(_RandomAccessIterator __first, _RandomAccessIterator __last, random_access_iterator_tag) 
# 101
{ 
# 105
return __last - __first; 
# 106
} 
# 110
template< class _Tp> ptrdiff_t __distance(_List_iterator< _Tp> , _List_iterator< _Tp> , input_iterator_tag); 
# 116
template< class _Tp> ptrdiff_t __distance(_List_const_iterator< _Tp> , _List_const_iterator< _Tp> , input_iterator_tag); 
# 125
template < typename _OutputIterator >
    void
    __distance ( _OutputIterator, _OutputIterator, output_iterator_tag ) = delete;
# 143 "/usr/include/c++/12/bits/stl_iterator_base_funcs.h" 3
template< class _InputIterator> 
# 144
[[__nodiscard__]] constexpr typename iterator_traits< _InputIterator> ::difference_type 
# 147
distance(_InputIterator __first, _InputIterator __last) 
# 148
{ 
# 150
return std::__distance(__first, __last, std::__iterator_category(__first)); 
# 152
} 
# 154
template< class _InputIterator, class _Distance> constexpr void 
# 156
__advance(_InputIterator &__i, _Distance __n, input_iterator_tag) 
# 157
{ 
# 160
do { if (std::__is_constant_evaluated() && (!((bool)(__n >= 0)))) { __builtin_unreachable(); }  } while (false); 
# 161
while (__n--) { 
# 162
++__i; }  
# 163
} 
# 165
template< class _BidirectionalIterator, class _Distance> constexpr void 
# 167
__advance(_BidirectionalIterator &__i, _Distance __n, bidirectional_iterator_tag) 
# 169
{ 
# 173
if (__n > 0) { 
# 174
while (__n--) { 
# 175
++__i; }  } else { 
# 177
while (__n++) { 
# 178
--__i; }  }  
# 179
} 
# 181
template< class _RandomAccessIterator, class _Distance> constexpr void 
# 183
__advance(_RandomAccessIterator &__i, _Distance __n, random_access_iterator_tag) 
# 185
{ 
# 189
if (__builtin_constant_p(__n) && (__n == 1)) { 
# 190
++__i; } else { 
# 191
if (__builtin_constant_p(__n) && (__n == (-1))) { 
# 192
--__i; } else { 
# 194
__i += __n; }  }  
# 195
} 
# 199
template < typename _OutputIterator, typename _Distance >
    void
    __advance ( _OutputIterator &, _Distance, output_iterator_tag ) = delete;
# 216 "/usr/include/c++/12/bits/stl_iterator_base_funcs.h" 3
template< class _InputIterator, class _Distance> constexpr void 
# 218
advance(_InputIterator &__i, _Distance __n) 
# 219
{ 
# 221
typename iterator_traits< _InputIterator> ::difference_type __d = __n; 
# 222
std::__advance(__i, __d, std::__iterator_category(__i)); 
# 223
} 
# 227
template< class _InputIterator> 
# 228
[[__nodiscard__]] constexpr _InputIterator 
# 230
next(_InputIterator __x, typename iterator_traits< _InputIterator> ::difference_type 
# 231
__n = 1) 
# 232
{ 
# 235
std::advance(__x, __n); 
# 236
return __x; 
# 237
} 
# 239
template< class _BidirectionalIterator> 
# 240
[[__nodiscard__]] constexpr _BidirectionalIterator 
# 242
prev(_BidirectionalIterator __x, typename iterator_traits< _BidirectionalIterator> ::difference_type 
# 243
__n = 1) 
# 244
{ 
# 248
std::advance(__x, -__n); 
# 249
return __x; 
# 250
} 
# 255
}
# 46 "/usr/include/c++/12/bits/ptr_traits.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 52
class __undefined; 
# 56
template< class _Tp> 
# 57
struct __get_first_arg { 
# 58
using type = __undefined; }; 
# 60
template< template< class , class ...>  class _SomeTemplate, class _Tp, class ...
# 61
_Types> 
# 62
struct __get_first_arg< _SomeTemplate< _Tp, _Types...> >  { 
# 63
using type = _Tp; }; 
# 67
template< class _Tp, class _Up> 
# 68
struct __replace_first_arg { 
# 69
}; 
# 71
template< template< class , class ...>  class _SomeTemplate, class _Up, class 
# 72
_Tp, class ..._Types> 
# 73
struct __replace_first_arg< _SomeTemplate< _Tp, _Types...> , _Up>  { 
# 74
using type = _SomeTemplate< _Up, _Types...> ; }; 
# 77
template< class _Ptr, class  = void> 
# 78
struct __ptr_traits_elem : public __get_first_arg< _Ptr>  { 
# 79
}; 
# 87
template< class _Ptr> 
# 88
struct __ptr_traits_elem< _Ptr, __void_t< typename _Ptr::element_type> >  { 
# 89
using type = typename _Ptr::element_type; }; 
# 92
template< class _Ptr> using __ptr_traits_elem_t = typename __ptr_traits_elem< _Ptr> ::type; 
# 98
template< class _Ptr, class _Elt, bool  = is_void< _Elt> ::value> 
# 99
struct __ptr_traits_ptr_to { 
# 101
using pointer = _Ptr; 
# 102
using element_type = _Elt; 
# 111
static pointer pointer_to(element_type &__r) 
# 117
{ return pointer::pointer_to(__r); } 
# 118
}; 
# 121
template< class _Ptr, class _Elt> 
# 122
struct __ptr_traits_ptr_to< _Ptr, _Elt, true>  { 
# 123
}; 
# 126
template< class _Tp> 
# 127
struct __ptr_traits_ptr_to< _Tp *, _Tp, false>  { 
# 129
using pointer = _Tp *; 
# 130
using element_type = _Tp; 
# 138
static pointer pointer_to(element_type &__r) noexcept 
# 139
{ return std::addressof(__r); } 
# 140
}; 
# 142
template< class _Ptr, class _Elt> 
# 143
struct __ptr_traits_impl : public __ptr_traits_ptr_to< _Ptr, _Elt>  { 
# 147
private: 
# 146
template< class _Tp> using __diff_t = typename _Tp::difference_type; 
# 149
template< class _Tp, class _Up> using __rebind = __type_identity< typename _Tp::template rebind< _Up> > ; 
# 154
public: using pointer = _Ptr; 
# 157
using element_type = _Elt; 
# 160
using difference_type = std::__detected_or_t< std::ptrdiff_t, __diff_t, _Ptr> ; 
# 163
template< class _Up> using rebind = typename std::__detected_or_t< __replace_first_arg< _Ptr, _Up> , __rebind, _Ptr, _Up> ::type; 
# 166
}; 
# 170
template< class _Ptr> 
# 171
struct __ptr_traits_impl< _Ptr, __undefined>  { 
# 172
}; 
# 180
template< class _Ptr> 
# 181
struct pointer_traits : public __ptr_traits_impl< _Ptr, __ptr_traits_elem_t< _Ptr> >  { 
# 182
}; 
# 190
template< class _Tp> 
# 191
struct pointer_traits< _Tp *>  : public __ptr_traits_ptr_to< _Tp *, _Tp>  { 
# 194
typedef _Tp *pointer; 
# 196
typedef _Tp element_type; 
# 198
typedef std::ptrdiff_t difference_type; 
# 200
template< class _Up> using rebind = _Up *; 
# 201
}; 
# 204
template< class _Ptr, class _Tp> using __ptr_rebind = typename pointer_traits< _Ptr> ::template rebind< _Tp> ; 
# 207
template< class _Tp> constexpr _Tp *
# 209
__to_address(_Tp *__ptr) noexcept 
# 210
{ 
# 211
static_assert((!std::template is_function< _Tp> ::value), "not a function pointer");
# 212
return __ptr; 
# 213
} 
# 216
template< class _Ptr> constexpr typename pointer_traits< _Ptr> ::element_type *
# 218
__to_address(const _Ptr &__ptr) 
# 219
{ return std::__to_address(__ptr.operator->()); } 
# 264 "/usr/include/c++/12/bits/ptr_traits.h" 3
}
# 88 "/usr/include/c++/12/bits/stl_iterator.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 109 "/usr/include/c++/12/bits/stl_iterator.h" 3
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
# 131 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Iterator> 
# 132
class reverse_iterator : public iterator< typename iterator_traits< _Iterator> ::iterator_category, typename iterator_traits< _Iterator> ::value_type, typename iterator_traits< _Iterator> ::difference_type, typename iterator_traits< _Iterator> ::pointer, typename iterator_traits< _Iterator> ::reference>  { 
# 139
template< class _Iter> friend class reverse_iterator; 
# 151 "/usr/include/c++/12/bits/stl_iterator.h" 3
protected: _Iterator current; 
# 153
typedef iterator_traits< _Iterator>  __traits_type; 
# 156
public: typedef _Iterator iterator_type; 
# 157
typedef typename iterator_traits< _Iterator> ::pointer pointer; 
# 159
typedef typename iterator_traits< _Iterator> ::difference_type difference_type; 
# 160
typedef typename iterator_traits< _Iterator> ::reference reference; 
# 182 "/usr/include/c++/12/bits/stl_iterator.h" 3
constexpr reverse_iterator() noexcept(noexcept((_Iterator()))) : current() 
# 185
{ } 
# 191
constexpr explicit reverse_iterator(iterator_type __x) noexcept(noexcept(((_Iterator)__x))) : current(__x) 
# 194
{ } 
# 200
constexpr reverse_iterator(const reverse_iterator &__x) noexcept(noexcept(((_Iterator)(__x.current)))) : current(__x.current) 
# 203
{ } 
# 206
reverse_iterator &operator=(const reverse_iterator &) = default;
# 213
template< class _Iter> constexpr 
# 218
reverse_iterator(const reverse_iterator< _Iter>  &__x) noexcept(noexcept(((_Iterator)(__x.current)))) : current((__x.current)) 
# 221
{ } 
# 224
template< class _Iter> constexpr reverse_iterator &
# 231
operator=(const reverse_iterator< _Iter>  &__x) noexcept(noexcept(((current) = (__x.current)))) 
# 233
{ 
# 234
(current) = (__x.current); 
# 235
return *this; 
# 236
} 
# 242
[[__nodiscard__]] constexpr iterator_type 
# 244
base() const noexcept(noexcept(((_Iterator)(current)))) 
# 246
{ return current; } 
# 258 "/usr/include/c++/12/bits/stl_iterator.h" 3
[[__nodiscard__]] constexpr reference 
# 260
operator*() const 
# 261
{ 
# 262
_Iterator __tmp = current; 
# 263
return *(--__tmp); 
# 264
} 
# 271
[[__nodiscard__]] constexpr pointer 
# 273
operator->() const 
# 278
{ 
# 281
_Iterator __tmp = current; 
# 282
--__tmp; 
# 283
return _S_to_pointer(__tmp); 
# 284
} 
# 292
constexpr reverse_iterator &operator++() 
# 293
{ 
# 294
--(current); 
# 295
return *this; 
# 296
} 
# 304
constexpr reverse_iterator operator++(int) 
# 305
{ 
# 306
reverse_iterator __tmp = *this; 
# 307
--(current); 
# 308
return __tmp; 
# 309
} 
# 317
constexpr reverse_iterator &operator--() 
# 318
{ 
# 319
++(current); 
# 320
return *this; 
# 321
} 
# 329
constexpr reverse_iterator operator--(int) 
# 330
{ 
# 331
reverse_iterator __tmp = *this; 
# 332
++(current); 
# 333
return __tmp; 
# 334
} 
# 341
[[__nodiscard__]] constexpr reverse_iterator 
# 343
operator+(difference_type __n) const 
# 344
{ return ((reverse_iterator)((current) - __n)); } 
# 353
constexpr reverse_iterator &operator+=(difference_type __n) 
# 354
{ 
# 355
(current) -= __n; 
# 356
return *this; 
# 357
} 
# 364
[[__nodiscard__]] constexpr reverse_iterator 
# 366
operator-(difference_type __n) const 
# 367
{ return ((reverse_iterator)((current) + __n)); } 
# 376
constexpr reverse_iterator &operator-=(difference_type __n) 
# 377
{ 
# 378
(current) += __n; 
# 379
return *this; 
# 380
} 
# 387
[[__nodiscard__]] constexpr reference 
# 389
operator[](difference_type __n) const 
# 390
{ return *((*this) + __n); } 
# 421 "/usr/include/c++/12/bits/stl_iterator.h" 3
private: 
# 419
template< class _Tp> static constexpr _Tp *
# 421
_S_to_pointer(_Tp *__p) 
# 422
{ return __p; } 
# 424
template< class _Tp> static constexpr pointer 
# 426
_S_to_pointer(_Tp __t) 
# 427
{ return __t.operator->(); } 
# 428
}; 
# 441 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Iterator> 
# 442
[[__nodiscard__]] constexpr bool 
# 444
operator==(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 445
__y) 
# 446
{ return __x.base() == __y.base(); } 
# 448
template< class _Iterator> 
# 449
[[__nodiscard__]] constexpr bool 
# 451
operator<(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 452
__y) 
# 453
{ return __y.base() < __x.base(); } 
# 455
template< class _Iterator> 
# 456
[[__nodiscard__]] constexpr bool 
# 458
operator!=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 459
__y) 
# 460
{ return !(__x == __y); } 
# 462
template< class _Iterator> 
# 463
[[__nodiscard__]] constexpr bool 
# 465
operator>(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 466
__y) 
# 467
{ return __y < __x; } 
# 469
template< class _Iterator> 
# 470
[[__nodiscard__]] constexpr bool 
# 472
operator<=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 473
__y) 
# 474
{ return !(__y < __x); } 
# 476
template< class _Iterator> 
# 477
[[__nodiscard__]] constexpr bool 
# 479
operator>=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 480
__y) 
# 481
{ return !(__x < __y); } 
# 486
template< class _IteratorL, class _IteratorR> 
# 487
[[__nodiscard__]] constexpr bool 
# 489
operator==(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 490
__y) 
# 491
{ return __x.base() == __y.base(); } 
# 493
template< class _IteratorL, class _IteratorR> 
# 494
[[__nodiscard__]] constexpr bool 
# 496
operator<(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 497
__y) 
# 498
{ return __x.base() > __y.base(); } 
# 500
template< class _IteratorL, class _IteratorR> 
# 501
[[__nodiscard__]] constexpr bool 
# 503
operator!=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 504
__y) 
# 505
{ return __x.base() != __y.base(); } 
# 507
template< class _IteratorL, class _IteratorR> 
# 508
[[__nodiscard__]] constexpr bool 
# 510
operator>(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 511
__y) 
# 512
{ return __x.base() < __y.base(); } 
# 514
template< class _IteratorL, class _IteratorR> constexpr bool 
# 516
operator<=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 517
__y) 
# 518
{ return __x.base() >= __y.base(); } 
# 520
template< class _IteratorL, class _IteratorR> 
# 521
[[__nodiscard__]] constexpr bool 
# 523
operator>=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 524
__y) 
# 525
{ return __x.base() <= __y.base(); } 
# 618 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _IteratorL, class _IteratorR> 
# 619
[[__nodiscard__]] constexpr auto 
# 621
operator-(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 622
__y)->__decltype((__y.base() - __x.base())) 
# 624
{ return __y.base() - __x.base(); } 
# 627
template< class _Iterator> 
# 628
[[__nodiscard__]] constexpr reverse_iterator< _Iterator>  
# 630
operator+(typename reverse_iterator< _Iterator> ::difference_type __n, const reverse_iterator< _Iterator>  &
# 631
__x) 
# 632
{ return ((reverse_iterator< _Iterator> )(__x.base() - __n)); } 
# 636
template< class _Iterator> constexpr reverse_iterator< _Iterator>  
# 638
__make_reverse_iterator(_Iterator __i) 
# 639
{ return ((reverse_iterator< _Iterator> )(__i)); } 
# 647
template< class _Iterator> 
# 648
[[__nodiscard__]] constexpr reverse_iterator< _Iterator>  
# 650
make_reverse_iterator(_Iterator __i) 
# 651
{ return ((reverse_iterator< _Iterator> )(__i)); } 
# 662 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Iterator> auto 
# 665
__niter_base(reverse_iterator< _Iterator>  __it)->__decltype((__make_reverse_iterator(__niter_base(__it.base())))) 
# 667
{ return __make_reverse_iterator(__niter_base(__it.base())); } 
# 669
template< class _Iterator> 
# 670
struct __is_move_iterator< reverse_iterator< _Iterator> >  : public std::__is_move_iterator< _Iterator>  { 
# 672
}; 
# 674
template< class _Iterator> auto 
# 677
__miter_base(reverse_iterator< _Iterator>  __it)->__decltype((__make_reverse_iterator(__miter_base(__it.base())))) 
# 679
{ return __make_reverse_iterator(__miter_base(__it.base())); } 
# 693 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Container> 
# 694
class back_insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 698
protected: _Container *container; 
# 702
public: typedef _Container container_type; 
# 709
explicit back_insert_iterator(_Container &__x) : container(std::__addressof(__x)) 
# 710
{ } 
# 733 "/usr/include/c++/12/bits/stl_iterator.h" 3
back_insert_iterator &operator=(const typename _Container::value_type &__value) 
# 734
{ 
# 735
(container)->push_back(__value); 
# 736
return *this; 
# 737
} 
# 741
back_insert_iterator &operator=(typename _Container::value_type &&__value) 
# 742
{ 
# 743
(container)->push_back(std::move(__value)); 
# 744
return *this; 
# 745
} 
# 749
[[__nodiscard__]] back_insert_iterator &
# 751
operator*() 
# 752
{ return *this; } 
# 757
back_insert_iterator &operator++() 
# 758
{ return *this; } 
# 763
back_insert_iterator operator++(int) 
# 764
{ return *this; } 
# 765
}; 
# 778 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Container> 
# 779
[[__nodiscard__]] inline back_insert_iterator< _Container>  
# 781
back_inserter(_Container &__x) 
# 782
{ return ((back_insert_iterator< _Container> )(__x)); } 
# 794 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Container> 
# 795
class front_insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 799
protected: _Container *container; 
# 803
public: typedef _Container container_type; 
# 810
explicit front_insert_iterator(_Container &__x) : container(std::__addressof(__x)) 
# 811
{ } 
# 834 "/usr/include/c++/12/bits/stl_iterator.h" 3
front_insert_iterator &operator=(const typename _Container::value_type &__value) 
# 835
{ 
# 836
(container)->push_front(__value); 
# 837
return *this; 
# 838
} 
# 842
front_insert_iterator &operator=(typename _Container::value_type &&__value) 
# 843
{ 
# 844
(container)->push_front(std::move(__value)); 
# 845
return *this; 
# 846
} 
# 850
[[__nodiscard__]] front_insert_iterator &
# 852
operator*() 
# 853
{ return *this; } 
# 858
front_insert_iterator &operator++() 
# 859
{ return *this; } 
# 864
front_insert_iterator operator++(int) 
# 865
{ return *this; } 
# 866
}; 
# 879 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Container> 
# 880
[[__nodiscard__]] inline front_insert_iterator< _Container>  
# 882
front_inserter(_Container &__x) 
# 883
{ return ((front_insert_iterator< _Container> )(__x)); } 
# 899 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Container> 
# 900
class insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 906
typedef typename _Container::iterator _Iter; 
# 909
protected: _Container *container; 
# 910
_Iter iter; 
# 914
public: typedef _Container container_type; 
# 925 "/usr/include/c++/12/bits/stl_iterator.h" 3
insert_iterator(_Container &__x, _Iter __i) : container(std::__addressof(__x)), iter(__i) 
# 926
{ } 
# 962 "/usr/include/c++/12/bits/stl_iterator.h" 3
insert_iterator &operator=(const typename _Container::value_type &__value) 
# 963
{ 
# 964
(iter) = (container)->insert(iter, __value); 
# 965
++(iter); 
# 966
return *this; 
# 967
} 
# 971
insert_iterator &operator=(typename _Container::value_type &&__value) 
# 972
{ 
# 973
(iter) = (container)->insert(iter, std::move(__value)); 
# 974
++(iter); 
# 975
return *this; 
# 976
} 
# 980
[[__nodiscard__]] insert_iterator &
# 982
operator*() 
# 983
{ return *this; } 
# 988
insert_iterator &operator++() 
# 989
{ return *this; } 
# 994
insert_iterator &operator++(int) 
# 995
{ return *this; } 
# 996
}; 
# 998
#pragma GCC diagnostic pop
# 1019 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Container> 
# 1020
[[__nodiscard__]] inline insert_iterator< _Container>  
# 1022
inserter(_Container &__x, typename _Container::iterator __i) 
# 1023
{ return insert_iterator< _Container> (__x, __i); } 
# 1029
}
# 1031
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 1042 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Iterator, class _Container> 
# 1043
class __normal_iterator { 
# 1046
protected: _Iterator _M_current; 
# 1048
typedef std::iterator_traits< _Iterator>  __traits_type; 
# 1051
template< class _Iter> using __convertible_from = std::__enable_if_t< std::is_convertible< _Iter, _Iterator> ::value> ; 
# 1057
public: typedef _Iterator iterator_type; 
# 1058
typedef typename std::iterator_traits< _Iterator> ::iterator_category iterator_category; 
# 1059
typedef typename std::iterator_traits< _Iterator> ::value_type value_type; 
# 1060
typedef typename std::iterator_traits< _Iterator> ::difference_type difference_type; 
# 1061
typedef typename std::iterator_traits< _Iterator> ::reference reference; 
# 1062
typedef typename std::iterator_traits< _Iterator> ::pointer pointer; 
# 1068
constexpr __normal_iterator() noexcept : _M_current(_Iterator()) 
# 1069
{ } 
# 1072
explicit __normal_iterator(const _Iterator &__i) noexcept : _M_current(__i) 
# 1073
{ } 
# 1077
template< class _Iter, class  = __convertible_from< _Iter> > 
# 1079
__normal_iterator(const __normal_iterator< _Iter, _Container>  &__i) noexcept : _M_current(__i.base()) 
# 1090 "/usr/include/c++/12/bits/stl_iterator.h" 3
{ } 
# 1095
reference operator*() const noexcept 
# 1096
{ return *(_M_current); } 
# 1100
pointer operator->() const noexcept 
# 1101
{ return _M_current; } 
# 1105
__normal_iterator &operator++() noexcept 
# 1106
{ 
# 1107
++(_M_current); 
# 1108
return *this; 
# 1109
} 
# 1113
__normal_iterator operator++(int) noexcept 
# 1114
{ return ((__normal_iterator)((_M_current)++)); } 
# 1119
__normal_iterator &operator--() noexcept 
# 1120
{ 
# 1121
--(_M_current); 
# 1122
return *this; 
# 1123
} 
# 1127
__normal_iterator operator--(int) noexcept 
# 1128
{ return ((__normal_iterator)((_M_current)--)); } 
# 1133
reference operator[](difference_type __n) const noexcept 
# 1134
{ return (_M_current)[__n]; } 
# 1138
__normal_iterator &operator+=(difference_type __n) noexcept 
# 1139
{ (_M_current) += __n; return *this; } 
# 1143
__normal_iterator operator+(difference_type __n) const noexcept 
# 1144
{ return ((__normal_iterator)((_M_current) + __n)); } 
# 1148
__normal_iterator &operator-=(difference_type __n) noexcept 
# 1149
{ (_M_current) -= __n; return *this; } 
# 1153
__normal_iterator operator-(difference_type __n) const noexcept 
# 1154
{ return ((__normal_iterator)((_M_current) - __n)); } 
# 1158
const _Iterator &base() const noexcept 
# 1159
{ return _M_current; } 
# 1160
}; 
# 1210 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _IteratorL, class _IteratorR, class _Container> 
# 1211
[[__nodiscard__]] inline bool 
# 1213
operator==(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1214
__rhs) noexcept 
# 1216
{ return __lhs.base() == __rhs.base(); } 
# 1218
template< class _Iterator, class _Container> 
# 1219
[[__nodiscard__]] inline bool 
# 1221
operator==(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1222
__rhs) noexcept 
# 1224
{ return __lhs.base() == __rhs.base(); } 
# 1226
template< class _IteratorL, class _IteratorR, class _Container> 
# 1227
[[__nodiscard__]] inline bool 
# 1229
operator!=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1230
__rhs) noexcept 
# 1232
{ return __lhs.base() != __rhs.base(); } 
# 1234
template< class _Iterator, class _Container> 
# 1235
[[__nodiscard__]] inline bool 
# 1237
operator!=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1238
__rhs) noexcept 
# 1240
{ return __lhs.base() != __rhs.base(); } 
# 1243
template< class _IteratorL, class _IteratorR, class _Container> 
# 1244
[[__nodiscard__]] inline bool 
# 1246
operator<(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1247
__rhs) noexcept 
# 1249
{ return __lhs.base() < __rhs.base(); } 
# 1251
template< class _Iterator, class _Container> 
# 1252
[[__nodiscard__]] inline bool 
# 1254
operator<(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1255
__rhs) noexcept 
# 1257
{ return __lhs.base() < __rhs.base(); } 
# 1259
template< class _IteratorL, class _IteratorR, class _Container> 
# 1260
[[__nodiscard__]] inline bool 
# 1262
operator>(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1263
__rhs) noexcept 
# 1265
{ return __lhs.base() > __rhs.base(); } 
# 1267
template< class _Iterator, class _Container> 
# 1268
[[__nodiscard__]] inline bool 
# 1270
operator>(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1271
__rhs) noexcept 
# 1273
{ return __lhs.base() > __rhs.base(); } 
# 1275
template< class _IteratorL, class _IteratorR, class _Container> 
# 1276
[[__nodiscard__]] inline bool 
# 1278
operator<=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1279
__rhs) noexcept 
# 1281
{ return __lhs.base() <= __rhs.base(); } 
# 1283
template< class _Iterator, class _Container> 
# 1284
[[__nodiscard__]] inline bool 
# 1286
operator<=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1287
__rhs) noexcept 
# 1289
{ return __lhs.base() <= __rhs.base(); } 
# 1291
template< class _IteratorL, class _IteratorR, class _Container> 
# 1292
[[__nodiscard__]] inline bool 
# 1294
operator>=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1295
__rhs) noexcept 
# 1297
{ return __lhs.base() >= __rhs.base(); } 
# 1299
template< class _Iterator, class _Container> 
# 1300
[[__nodiscard__]] inline bool 
# 1302
operator>=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1303
__rhs) noexcept 
# 1305
{ return __lhs.base() >= __rhs.base(); } 
# 1312
template< class _IteratorL, class _IteratorR, class _Container> 
# 1315
[[__nodiscard__]] inline auto 
# 1317
operator-(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1318
__rhs) noexcept->__decltype((__lhs.base() - __rhs.base())) 
# 1325
{ return __lhs.base() - __rhs.base(); } 
# 1327
template< class _Iterator, class _Container> 
# 1328
[[__nodiscard__]] inline typename __normal_iterator< _Iterator, _Container> ::difference_type 
# 1330
operator-(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1331
__rhs) noexcept 
# 1333
{ return __lhs.base() - __rhs.base(); } 
# 1335
template< class _Iterator, class _Container> 
# 1336
[[__nodiscard__]] inline __normal_iterator< _Iterator, _Container>  
# 1338
operator+(typename __normal_iterator< _Iterator, _Container> ::difference_type 
# 1339
__n, const __normal_iterator< _Iterator, _Container>  &__i) noexcept 
# 1341
{ return ((__normal_iterator< _Iterator, _Container> )(__i.base() + __n)); } 
# 1344
}
# 1346
namespace std __attribute((__visibility__("default"))) { 
# 1350
template< class _Iterator, class _Container> _Iterator 
# 1353
__niter_base(__gnu_cxx::__normal_iterator< _Iterator, _Container>  __it) noexcept(std::template is_nothrow_copy_constructible< _Iterator> ::value) 
# 1355
{ return __it.base(); } 
# 1362
template< class _Iterator, class _Container> constexpr auto 
# 1364
__to_address(const __gnu_cxx::__normal_iterator< _Iterator, _Container>  &
# 1365
__it) noexcept->__decltype((std::__to_address(__it.base()))) 
# 1367
{ return std::__to_address(__it.base()); } 
# 1417 "/usr/include/c++/12/bits/stl_iterator.h" 3
namespace __detail { 
# 1433 "/usr/include/c++/12/bits/stl_iterator.h" 3
}
# 1444 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Iterator> 
# 1445
class move_iterator { 
# 1450
_Iterator _M_current; 
# 1452
using __traits_type = iterator_traits< _Iterator> ; 
# 1454
using __base_ref = typename iterator_traits< _Iterator> ::reference; 
# 1457
template< class _Iter2> friend class move_iterator; 
# 1484 "/usr/include/c++/12/bits/stl_iterator.h" 3
public: using iterator_type = _Iterator; 
# 1497 "/usr/include/c++/12/bits/stl_iterator.h" 3
typedef typename iterator_traits< _Iterator> ::iterator_category iterator_category; 
# 1498
typedef typename iterator_traits< _Iterator> ::value_type value_type; 
# 1499
typedef typename iterator_traits< _Iterator> ::difference_type difference_type; 
# 1501
typedef _Iterator pointer; 
# 1504
using reference = __conditional_t< is_reference< __base_ref> ::value, typename remove_reference< __base_ref> ::type &&, __base_ref> ; 
# 1511
constexpr move_iterator() : _M_current() 
# 1512
{ } 
# 1515
constexpr explicit move_iterator(iterator_type __i) : _M_current(std::move(__i)) 
# 1516
{ } 
# 1518
template< class _Iter> constexpr 
# 1523
move_iterator(const move_iterator< _Iter>  &__i) : _M_current((__i._M_current)) 
# 1524
{ } 
# 1526
template< class _Iter> constexpr move_iterator &
# 1532
operator=(const move_iterator< _Iter>  &__i) 
# 1533
{ 
# 1534
(_M_current) = (__i._M_current); 
# 1535
return *this; 
# 1536
} 
# 1539
[[__nodiscard__]] constexpr iterator_type 
# 1541
base() const 
# 1542
{ return _M_current; } 
# 1555 "/usr/include/c++/12/bits/stl_iterator.h" 3
[[__nodiscard__]] constexpr reference 
# 1557
operator*() const 
# 1561
{ return static_cast< reference>(*(_M_current)); } 
# 1564
[[__nodiscard__]] constexpr pointer 
# 1566
operator->() const 
# 1567
{ return _M_current; } 
# 1570
constexpr move_iterator &operator++() 
# 1571
{ 
# 1572
++(_M_current); 
# 1573
return *this; 
# 1574
} 
# 1577
constexpr move_iterator operator++(int) 
# 1578
{ 
# 1579
move_iterator __tmp = *this; 
# 1580
++(_M_current); 
# 1581
return __tmp; 
# 1582
} 
# 1591
constexpr move_iterator &operator--() 
# 1592
{ 
# 1593
--(_M_current); 
# 1594
return *this; 
# 1595
} 
# 1598
constexpr move_iterator operator--(int) 
# 1599
{ 
# 1600
move_iterator __tmp = *this; 
# 1601
--(_M_current); 
# 1602
return __tmp; 
# 1603
} 
# 1605
[[__nodiscard__]] constexpr move_iterator 
# 1607
operator+(difference_type __n) const 
# 1608
{ return ((move_iterator)((_M_current) + __n)); } 
# 1611
constexpr move_iterator &operator+=(difference_type __n) 
# 1612
{ 
# 1613
(_M_current) += __n; 
# 1614
return *this; 
# 1615
} 
# 1617
[[__nodiscard__]] constexpr move_iterator 
# 1619
operator-(difference_type __n) const 
# 1620
{ return ((move_iterator)((_M_current) - __n)); } 
# 1623
constexpr move_iterator &operator-=(difference_type __n) 
# 1624
{ 
# 1625
(_M_current) -= __n; 
# 1626
return *this; 
# 1627
} 
# 1629
[[__nodiscard__]] constexpr reference 
# 1631
operator[](difference_type __n) const 
# 1635
{ return std::move((_M_current)[__n]); } 
# 1669 "/usr/include/c++/12/bits/stl_iterator.h" 3
}; 
# 1671
template< class _IteratorL, class _IteratorR> 
# 1672
[[__nodiscard__]] constexpr bool 
# 1674
operator==(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1675
__y) 
# 1679
{ return __x.base() == __y.base(); } 
# 1690 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _IteratorL, class _IteratorR> 
# 1691
[[__nodiscard__]] constexpr bool 
# 1693
operator!=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1694
__y) 
# 1695
{ return !(__x == __y); } 
# 1698
template< class _IteratorL, class _IteratorR> 
# 1699
[[__nodiscard__]] constexpr bool 
# 1701
operator<(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1702
__y) 
# 1706
{ return __x.base() < __y.base(); } 
# 1708
template< class _IteratorL, class _IteratorR> 
# 1709
[[__nodiscard__]] constexpr bool 
# 1711
operator<=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1712
__y) 
# 1716
{ return !(__y < __x); } 
# 1718
template< class _IteratorL, class _IteratorR> 
# 1719
[[__nodiscard__]] constexpr bool 
# 1721
operator>(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1722
__y) 
# 1726
{ return __y < __x; } 
# 1728
template< class _IteratorL, class _IteratorR> 
# 1729
[[__nodiscard__]] constexpr bool 
# 1731
operator>=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1732
__y) 
# 1736
{ return !(__x < __y); } 
# 1741
template< class _Iterator> 
# 1742
[[__nodiscard__]] constexpr bool 
# 1744
operator==(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1745
__y) 
# 1746
{ return __x.base() == __y.base(); } 
# 1756 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Iterator> 
# 1757
[[__nodiscard__]] constexpr bool 
# 1759
operator!=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1760
__y) 
# 1761
{ return !(__x == __y); } 
# 1763
template< class _Iterator> 
# 1764
[[__nodiscard__]] constexpr bool 
# 1766
operator<(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1767
__y) 
# 1768
{ return __x.base() < __y.base(); } 
# 1770
template< class _Iterator> 
# 1771
[[__nodiscard__]] constexpr bool 
# 1773
operator<=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1774
__y) 
# 1775
{ return !(__y < __x); } 
# 1777
template< class _Iterator> 
# 1778
[[__nodiscard__]] constexpr bool 
# 1780
operator>(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1781
__y) 
# 1782
{ return __y < __x; } 
# 1784
template< class _Iterator> 
# 1785
[[__nodiscard__]] constexpr bool 
# 1787
operator>=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1788
__y) 
# 1789
{ return !(__x < __y); } 
# 1793
template< class _IteratorL, class _IteratorR> 
# 1794
[[__nodiscard__]] constexpr auto 
# 1796
operator-(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1797
__y)->__decltype((__x.base() - __y.base())) 
# 1799
{ return __x.base() - __y.base(); } 
# 1801
template< class _Iterator> 
# 1802
[[__nodiscard__]] constexpr move_iterator< _Iterator>  
# 1804
operator+(typename move_iterator< _Iterator> ::difference_type __n, const move_iterator< _Iterator>  &
# 1805
__x) 
# 1806
{ return __x + __n; } 
# 1808
template< class _Iterator> 
# 1809
[[__nodiscard__]] constexpr move_iterator< _Iterator>  
# 1811
make_move_iterator(_Iterator __i) 
# 1812
{ return ((move_iterator< _Iterator> )(std::move(__i))); } 
# 1814
template< class _Iterator, class _ReturnType = __conditional_t< __move_if_noexcept_cond< typename iterator_traits< _Iterator> ::value_type> ::value, _Iterator, move_iterator< _Iterator> > > constexpr _ReturnType 
# 1819
__make_move_if_noexcept_iterator(_Iterator __i) 
# 1820
{ return (_ReturnType)__i; } 
# 1824
template< class _Tp, class _ReturnType = __conditional_t< __move_if_noexcept_cond< _Tp> ::value, const _Tp *, move_iterator< _Tp *> > > constexpr _ReturnType 
# 1828
__make_move_if_noexcept_iterator(_Tp *__i) 
# 1829
{ return (_ReturnType)__i; } 
# 2570 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _Iterator> auto 
# 2573
__niter_base(move_iterator< _Iterator>  __it)->__decltype((make_move_iterator(__niter_base(__it.base())))) 
# 2575
{ return make_move_iterator(__niter_base(__it.base())); } 
# 2577
template< class _Iterator> 
# 2578
struct __is_move_iterator< move_iterator< _Iterator> >  { 
# 2580
enum { __value = 1}; 
# 2581
typedef __true_type __type; 
# 2582
}; 
# 2584
template< class _Iterator> auto 
# 2587
__miter_base(move_iterator< _Iterator>  __it)->__decltype((__miter_base(__it.base()))) 
# 2589
{ return __miter_base(__it.base()); } 
# 2602 "/usr/include/c++/12/bits/stl_iterator.h" 3
template< class _InputIterator> using __iter_key_t = remove_const_t< typename iterator_traits< _InputIterator> ::value_type::first_type> ; 
# 2606
template< class _InputIterator> using __iter_val_t = typename iterator_traits< _InputIterator> ::value_type::second_type; 
# 2610
template< class _T1, class _T2> struct pair; 
# 2613
template< class _InputIterator> using __iter_to_alloc_t = pair< add_const_t< __iter_key_t< _InputIterator> > , __iter_val_t< _InputIterator> > ; 
# 2620
}
# 48 "/usr/include/c++/12/debug/debug.h" 3
namespace std { 
# 50
namespace __debug { }
# 51
}
# 56
namespace __gnu_debug { 
# 58
using namespace std::__debug;
# 60
template< class _Ite, class _Seq, class _Cat> struct _Safe_iterator; 
# 62
}
# 35 "/usr/include/c++/12/bits/predefined_ops.h" 3
namespace __gnu_cxx { 
# 37
namespace __ops { 
# 39
struct _Iter_less_iter { 
# 41
template< class _Iterator1, class _Iterator2> constexpr bool 
# 44
operator()(_Iterator1 __it1, _Iterator2 __it2) const 
# 45
{ return (*__it1) < (*__it2); } 
# 46
}; 
# 50
constexpr _Iter_less_iter __iter_less_iter() 
# 51
{ return _Iter_less_iter(); } 
# 53
struct _Iter_less_val { 
# 56
constexpr _Iter_less_val() = default;
# 63
explicit _Iter_less_val(_Iter_less_iter) { } 
# 65
template< class _Iterator, class _Value> bool 
# 68
operator()(_Iterator __it, _Value &__val) const 
# 69
{ return (*__it) < __val; } 
# 70
}; 
# 74
inline _Iter_less_val __iter_less_val() 
# 75
{ return _Iter_less_val(); } 
# 79
inline _Iter_less_val __iter_comp_val(_Iter_less_iter) 
# 80
{ return _Iter_less_val(); } 
# 82
struct _Val_less_iter { 
# 85
constexpr _Val_less_iter() = default;
# 92
explicit _Val_less_iter(_Iter_less_iter) { } 
# 94
template< class _Value, class _Iterator> bool 
# 97
operator()(_Value &__val, _Iterator __it) const 
# 98
{ return __val < (*__it); } 
# 99
}; 
# 103
inline _Val_less_iter __val_less_iter() 
# 104
{ return _Val_less_iter(); } 
# 108
inline _Val_less_iter __val_comp_iter(_Iter_less_iter) 
# 109
{ return _Val_less_iter(); } 
# 111
struct _Iter_equal_to_iter { 
# 113
template< class _Iterator1, class _Iterator2> bool 
# 116
operator()(_Iterator1 __it1, _Iterator2 __it2) const 
# 117
{ return (*__it1) == (*__it2); } 
# 118
}; 
# 122
inline _Iter_equal_to_iter __iter_equal_to_iter() 
# 123
{ return _Iter_equal_to_iter(); } 
# 125
struct _Iter_equal_to_val { 
# 127
template< class _Iterator, class _Value> bool 
# 130
operator()(_Iterator __it, _Value &__val) const 
# 131
{ return (*__it) == __val; } 
# 132
}; 
# 136
inline _Iter_equal_to_val __iter_equal_to_val() 
# 137
{ return _Iter_equal_to_val(); } 
# 141
inline _Iter_equal_to_val __iter_comp_val(_Iter_equal_to_iter) 
# 142
{ return _Iter_equal_to_val(); } 
# 144
template< class _Compare> 
# 145
struct _Iter_comp_iter { 
# 147
_Compare _M_comp; 
# 150
constexpr explicit _Iter_comp_iter(_Compare __comp) : _M_comp(std::move(__comp)) 
# 152
{ } 
# 154
template< class _Iterator1, class _Iterator2> constexpr bool 
# 157
operator()(_Iterator1 __it1, _Iterator2 __it2) 
# 158
{ return (bool)(_M_comp)(*__it1, *__it2); } 
# 159
}; 
# 161
template< class _Compare> constexpr _Iter_comp_iter< _Compare>  
# 164
__iter_comp_iter(_Compare __comp) 
# 165
{ return ((_Iter_comp_iter< _Compare> )(std::move(__comp))); } 
# 167
template< class _Compare> 
# 168
struct _Iter_comp_val { 
# 170
_Compare _M_comp; 
# 174
explicit _Iter_comp_val(_Compare __comp) : _M_comp(std::move(__comp)) 
# 176
{ } 
# 180
explicit _Iter_comp_val(const _Iter_comp_iter< _Compare>  &__comp) : _M_comp((__comp._M_comp)) 
# 182
{ } 
# 187
explicit _Iter_comp_val(_Iter_comp_iter< _Compare>  &&__comp) : _M_comp(std::move((__comp._M_comp))) 
# 189
{ } 
# 192
template< class _Iterator, class _Value> bool 
# 195
operator()(_Iterator __it, _Value &__val) 
# 196
{ return (bool)(_M_comp)(*__it, __val); } 
# 197
}; 
# 199
template< class _Compare> inline _Iter_comp_val< _Compare>  
# 202
__iter_comp_val(_Compare __comp) 
# 203
{ return ((_Iter_comp_val< _Compare> )(std::move(__comp))); } 
# 205
template< class _Compare> inline _Iter_comp_val< _Compare>  
# 208
__iter_comp_val(_Iter_comp_iter< _Compare>  __comp) 
# 209
{ return ((_Iter_comp_val< _Compare> )(std::move(__comp))); } 
# 211
template< class _Compare> 
# 212
struct _Val_comp_iter { 
# 214
_Compare _M_comp; 
# 218
explicit _Val_comp_iter(_Compare __comp) : _M_comp(std::move(__comp)) 
# 220
{ } 
# 224
explicit _Val_comp_iter(const _Iter_comp_iter< _Compare>  &__comp) : _M_comp((__comp._M_comp)) 
# 226
{ } 
# 231
explicit _Val_comp_iter(_Iter_comp_iter< _Compare>  &&__comp) : _M_comp(std::move((__comp._M_comp))) 
# 233
{ } 
# 236
template< class _Value, class _Iterator> bool 
# 239
operator()(_Value &__val, _Iterator __it) 
# 240
{ return (bool)(_M_comp)(__val, *__it); } 
# 241
}; 
# 243
template< class _Compare> inline _Val_comp_iter< _Compare>  
# 246
__val_comp_iter(_Compare __comp) 
# 247
{ return ((_Val_comp_iter< _Compare> )(std::move(__comp))); } 
# 249
template< class _Compare> inline _Val_comp_iter< _Compare>  
# 252
__val_comp_iter(_Iter_comp_iter< _Compare>  __comp) 
# 253
{ return ((_Val_comp_iter< _Compare> )(std::move(__comp))); } 
# 255
template< class _Value> 
# 256
struct _Iter_equals_val { 
# 258
_Value &_M_value; 
# 262
explicit _Iter_equals_val(_Value &__value) : _M_value(__value) 
# 264
{ } 
# 266
template< class _Iterator> bool 
# 269
operator()(_Iterator __it) 
# 270
{ return (*__it) == (_M_value); } 
# 271
}; 
# 273
template< class _Value> inline _Iter_equals_val< _Value>  
# 276
__iter_equals_val(_Value &__val) 
# 277
{ return ((_Iter_equals_val< _Value> )(__val)); } 
# 279
template< class _Iterator1> 
# 280
struct _Iter_equals_iter { 
# 282
_Iterator1 _M_it1; 
# 286
explicit _Iter_equals_iter(_Iterator1 __it1) : _M_it1(__it1) 
# 288
{ } 
# 290
template< class _Iterator2> bool 
# 293
operator()(_Iterator2 __it2) 
# 294
{ return (*__it2) == (*(_M_it1)); } 
# 295
}; 
# 297
template< class _Iterator> inline _Iter_equals_iter< _Iterator>  
# 300
__iter_comp_iter(_Iter_equal_to_iter, _Iterator __it) 
# 301
{ return ((_Iter_equals_iter< _Iterator> )(__it)); } 
# 303
template< class _Predicate> 
# 304
struct _Iter_pred { 
# 306
_Predicate _M_pred; 
# 310
explicit _Iter_pred(_Predicate __pred) : _M_pred(std::move(__pred)) 
# 312
{ } 
# 314
template< class _Iterator> bool 
# 317
operator()(_Iterator __it) 
# 318
{ return (bool)(_M_pred)(*__it); } 
# 319
}; 
# 321
template< class _Predicate> inline _Iter_pred< _Predicate>  
# 324
__pred_iter(_Predicate __pred) 
# 325
{ return ((_Iter_pred< _Predicate> )(std::move(__pred))); } 
# 327
template< class _Compare, class _Value> 
# 328
struct _Iter_comp_to_val { 
# 330
_Compare _M_comp; 
# 331
_Value &_M_value; 
# 334
_Iter_comp_to_val(_Compare __comp, _Value &__value) : _M_comp(std::move(__comp)), _M_value(__value) 
# 336
{ } 
# 338
template< class _Iterator> bool 
# 341
operator()(_Iterator __it) 
# 342
{ return (bool)(_M_comp)(*__it, _M_value); } 
# 343
}; 
# 345
template< class _Compare, class _Value> _Iter_comp_to_val< _Compare, _Value>  
# 348
__iter_comp_val(_Compare __comp, _Value &__val) 
# 349
{ 
# 350
return _Iter_comp_to_val< _Compare, _Value> (std::move(__comp), __val); 
# 351
} 
# 353
template< class _Compare, class _Iterator1> 
# 354
struct _Iter_comp_to_iter { 
# 356
_Compare _M_comp; 
# 357
_Iterator1 _M_it1; 
# 360
_Iter_comp_to_iter(_Compare __comp, _Iterator1 __it1) : _M_comp(std::move(__comp)), _M_it1(__it1) 
# 362
{ } 
# 364
template< class _Iterator2> bool 
# 367
operator()(_Iterator2 __it2) 
# 368
{ return (bool)(_M_comp)(*__it2, *(_M_it1)); } 
# 369
}; 
# 371
template< class _Compare, class _Iterator> inline _Iter_comp_to_iter< _Compare, _Iterator>  
# 374
__iter_comp_iter(_Iter_comp_iter< _Compare>  __comp, _Iterator __it) 
# 375
{ 
# 376
return _Iter_comp_to_iter< _Compare, _Iterator> (std::move((__comp._M_comp)), __it); 
# 378
} 
# 380
template< class _Predicate> 
# 381
struct _Iter_negate { 
# 383
_Predicate _M_pred; 
# 387
explicit _Iter_negate(_Predicate __pred) : _M_pred(std::move(__pred)) 
# 389
{ } 
# 391
template< class _Iterator> bool 
# 394
operator()(_Iterator __it) 
# 395
{ return !((bool)(_M_pred)(*__it)); } 
# 396
}; 
# 398
template< class _Predicate> inline _Iter_negate< _Predicate>  
# 401
__negate(_Iter_pred< _Predicate>  __pred) 
# 402
{ return ((_Iter_negate< _Predicate> )(std::move((__pred._M_pred)))); } 
# 404
}
# 405
}
# 79 "/usr/include/c++/12/bits/stl_algobase.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 87
template< class _Tp, class _Up> constexpr int 
# 90
__memcmp(const _Tp *__first1, const _Up *__first2, size_t __num) 
# 91
{ 
# 93
static_assert((sizeof(_Tp) == sizeof(_Up)), "can be compared with memcmp");
# 105 "/usr/include/c++/12/bits/stl_algobase.h" 3
return __builtin_memcmp(__first1, __first2, sizeof(_Tp) * __num); 
# 106
} 
# 149 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _ForwardIterator1, class _ForwardIterator2> inline void 
# 152
iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b) 
# 153
{ 
# 182 "/usr/include/c++/12/bits/stl_algobase.h" 3
swap(*__a, *__b); 
# 184
} 
# 198 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _ForwardIterator1, class _ForwardIterator2> _ForwardIterator2 
# 201
swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 
# 202
__first2) 
# 203
{ 
# 209
; 
# 211
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 212
std::iter_swap(__first1, __first2); }  
# 213
return __first2; 
# 214
} 
# 227 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _Tp> constexpr const _Tp &
# 230
min(const _Tp &__a, const _Tp &__b) 
# 231
{ 
# 235
if (__b < __a) { 
# 236
return __b; }  
# 237
return __a; 
# 238
} 
# 251 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _Tp> constexpr const _Tp &
# 254
max(const _Tp &__a, const _Tp &__b) 
# 255
{ 
# 259
if (__a < __b) { 
# 260
return __b; }  
# 261
return __a; 
# 262
} 
# 275 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _Tp, class _Compare> constexpr const _Tp &
# 278
min(const _Tp &__a, const _Tp &__b, _Compare __comp) 
# 279
{ 
# 281
if (__comp(__b, __a)) { 
# 282
return __b; }  
# 283
return __a; 
# 284
} 
# 297 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _Tp, class _Compare> constexpr const _Tp &
# 300
max(const _Tp &__a, const _Tp &__b, _Compare __comp) 
# 301
{ 
# 303
if (__comp(__a, __b)) { 
# 304
return __b; }  
# 305
return __a; 
# 306
} 
# 310
template< class _Iterator> inline _Iterator 
# 313
__niter_base(_Iterator __it) noexcept(std::template is_nothrow_copy_constructible< _Iterator> ::value) 
# 315
{ return __it; } 
# 317
template< class _Ite, class _Seq> _Ite __niter_base(const __gnu_debug::_Safe_iterator< _Ite, _Seq, random_access_iterator_tag>  &); 
# 325
template< class _From, class _To> inline _From 
# 328
__niter_wrap(_From __from, _To __res) 
# 329
{ return __from + (__res - std::__niter_base(__from)); } 
# 332
template< class _Iterator> inline _Iterator 
# 335
__niter_wrap(const _Iterator &, _Iterator __res) 
# 336
{ return __res; } 
# 344
template< bool _IsMove, bool _IsSimple, class _Category> 
# 345
struct __copy_move { 
# 347
template< class _II, class _OI> static _OI 
# 350
__copy_m(_II __first, _II __last, _OI __result) 
# 351
{ 
# 352
for (; __first != __last; (++__result), ((void)(++__first))) { 
# 353
(*__result) = (*__first); }  
# 354
return __result; 
# 355
} 
# 356
}; 
# 359
template< class _Category> 
# 360
struct __copy_move< true, false, _Category>  { 
# 362
template< class _II, class _OI> static _OI 
# 365
__copy_m(_II __first, _II __last, _OI __result) 
# 366
{ 
# 367
for (; __first != __last; (++__result), ((void)(++__first))) { 
# 368
(*__result) = std::move(*__first); }  
# 369
return __result; 
# 370
} 
# 371
}; 
# 375
template<> struct __copy_move< false, false, random_access_iterator_tag>  { 
# 377
template< class _II, class _OI> static _OI 
# 380
__copy_m(_II __first, _II __last, _OI __result) 
# 381
{ 
# 382
typedef typename iterator_traits< _II> ::difference_type _Distance; 
# 383
for (_Distance __n = __last - __first; __n > 0; --__n) 
# 384
{ 
# 385
(*__result) = (*__first); 
# 386
++__first; 
# 387
++__result; 
# 388
}  
# 389
return __result; 
# 390
} 
# 392
template< class _Tp, class _Up> static void 
# 394
__assign_one(_Tp *__to, _Up *__from) 
# 395
{ (*__to) = (*__from); } 
# 396
}; 
# 400
template<> struct __copy_move< true, false, random_access_iterator_tag>  { 
# 402
template< class _II, class _OI> static _OI 
# 405
__copy_m(_II __first, _II __last, _OI __result) 
# 406
{ 
# 407
typedef typename iterator_traits< _II> ::difference_type _Distance; 
# 408
for (_Distance __n = __last - __first; __n > 0; --__n) 
# 409
{ 
# 410
(*__result) = std::move(*__first); 
# 411
++__first; 
# 412
++__result; 
# 413
}  
# 414
return __result; 
# 415
} 
# 417
template< class _Tp, class _Up> static void 
# 419
__assign_one(_Tp *__to, _Up *__from) 
# 420
{ (*__to) = std::move(*__from); } 
# 421
}; 
# 424
template< bool _IsMove> 
# 425
struct __copy_move< _IsMove, true, random_access_iterator_tag>  { 
# 427
template< class _Tp, class _Up> static _Up *
# 430
__copy_m(_Tp *__first, _Tp *__last, _Up *__result) 
# 431
{ 
# 432
const ptrdiff_t _Num = __last - __first; 
# 433
if (__builtin_expect(_Num > (1), true)) { 
# 434
__builtin_memmove(__result, __first, sizeof(_Tp) * _Num); } else { 
# 435
if (_Num == (1)) { 
# 436
std::template __copy_move< _IsMove, false, random_access_iterator_tag> ::__assign_one(__result, __first); }  }  
# 438
return __result + _Num; 
# 439
} 
# 440
}; 
# 444
template< class _Tp, class _Ref, class _Ptr> struct _Deque_iterator; 
# 447
struct _Bit_iterator; 
# 453
template< class _CharT> struct char_traits; 
# 456
template< class _CharT, class _Traits> class istreambuf_iterator; 
# 459
template< class _CharT, class _Traits> class ostreambuf_iterator; 
# 462
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, ostreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type __copy_move_a2(_CharT *, _CharT *, ostreambuf_iterator< _CharT, char_traits< _CharT> > ); 
# 468
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, ostreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type __copy_move_a2(const _CharT *, const _CharT *, ostreambuf_iterator< _CharT, char_traits< _CharT> > ); 
# 474
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _CharT *> ::__type __copy_move_a2(istreambuf_iterator< _CharT, char_traits< _CharT> > , istreambuf_iterator< _CharT, char_traits< _CharT> > , _CharT *); 
# 480
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _Deque_iterator< _CharT, _CharT &, _CharT *> > ::__type __copy_move_a2(istreambuf_iterator< _CharT, char_traits< _CharT> > , istreambuf_iterator< _CharT, char_traits< _CharT> > , _Deque_iterator< _CharT, _CharT &, _CharT *> ); 
# 489
template< bool _IsMove, class _II, class _OI> inline _OI 
# 492
__copy_move_a2(_II __first, _II __last, _OI __result) 
# 493
{ 
# 494
typedef typename iterator_traits< _II> ::iterator_category _Category; 
# 500
return std::template __copy_move< _IsMove, __memcpyable< _OI, _II> ::__value, typename iterator_traits< _II> ::iterator_category> ::__copy_m(__first, __last, __result); 
# 502
} 
# 504
template< bool _IsMove, class 
# 505
_Tp, class _Ref, class _Ptr, class _OI> _OI 
# 504
__copy_move_a1(_Deque_iterator< _Tp, _Ref, _Ptr> , _Deque_iterator< _Tp, _Ref, _Ptr> , _OI); 
# 511
template< bool _IsMove, class 
# 512
_ITp, class _IRef, class _IPtr, class _OTp> _Deque_iterator< _OTp, _OTp &, _OTp *>  
# 511
__copy_move_a1(_Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _OTp, _OTp &, _OTp *> ); 
# 518
template< bool _IsMove, class _II, class _Tp> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, _Deque_iterator< _Tp, _Tp &, _Tp *> > ::__type __copy_move_a1(_II, _II, _Deque_iterator< _Tp, _Tp &, _Tp *> ); 
# 524
template< bool _IsMove, class _II, class _OI> inline _OI 
# 527
__copy_move_a1(_II __first, _II __last, _OI __result) 
# 528
{ return std::__copy_move_a2< _IsMove> (__first, __last, __result); } 
# 530
template< bool _IsMove, class _II, class _OI> inline _OI 
# 533
__copy_move_a(_II __first, _II __last, _OI __result) 
# 534
{ 
# 535
return std::__niter_wrap(__result, std::__copy_move_a1< _IsMove> (std::__niter_base(__first), std::__niter_base(__last), std::__niter_base(__result))); 
# 539
} 
# 541
template< bool _IsMove, class 
# 542
_Ite, class _Seq, class _Cat, class _OI> _OI 
# 541
__copy_move_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, _OI); 
# 548
template< bool _IsMove, class 
# 549
_II, class _Ite, class _Seq, class _Cat> __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  
# 548
__copy_move_a(_II, _II, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &); 
# 554
template< bool _IsMove, class 
# 555
_IIte, class _ISeq, class _ICat, class 
# 556
_OIte, class _OSeq, class _OCat> __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  
# 554
__copy_move_a(const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  &); 
# 562
template< class _InputIterator, class _Size, class _OutputIterator> _OutputIterator 
# 565
__copy_n_a(_InputIterator __first, _Size __n, _OutputIterator __result, bool) 
# 567
{ 
# 568
if (__n > 0) 
# 569
{ 
# 570
while (true) 
# 571
{ 
# 572
(*__result) = (*__first); 
# 573
++__result; 
# 574
if ((--__n) > 0) { 
# 575
++__first; } else { 
# 577
break; }  
# 578
}  
# 579
}  
# 580
return __result; 
# 581
} 
# 583
template< class _CharT, class _Size> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _CharT *> ::__type __copy_n_a(istreambuf_iterator< _CharT, char_traits< _CharT> > , _Size, _CharT *, bool); 
# 589
template< class _CharT, class _Size> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _Deque_iterator< _CharT, _CharT &, _CharT *> > ::__type __copy_n_a(istreambuf_iterator< _CharT, char_traits< _CharT> > , _Size, _Deque_iterator< _CharT, _CharT &, _CharT *> , bool); 
# 614 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _II, class _OI> inline _OI 
# 617
copy(_II __first, _II __last, _OI __result) 
# 618
{ 
# 623
; 
# 625
return std::__copy_move_a< __is_move_iterator< _II> ::__value> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 627
} 
# 647 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _II, class _OI> inline _OI 
# 650
move(_II __first, _II __last, _OI __result) 
# 651
{ 
# 656
; 
# 658
return std::__copy_move_a< true> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 660
} 
# 667
template< bool _IsMove, bool _IsSimple, class _Category> 
# 668
struct __copy_move_backward { 
# 670
template< class _BI1, class _BI2> static _BI2 
# 673
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 674
{ 
# 675
while (__first != __last) { 
# 676
(*(--__result)) = (*(--__last)); }  
# 677
return __result; 
# 678
} 
# 679
}; 
# 682
template< class _Category> 
# 683
struct __copy_move_backward< true, false, _Category>  { 
# 685
template< class _BI1, class _BI2> static _BI2 
# 688
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 689
{ 
# 690
while (__first != __last) { 
# 691
(*(--__result)) = std::move(*(--__last)); }  
# 692
return __result; 
# 693
} 
# 694
}; 
# 698
template<> struct __copy_move_backward< false, false, random_access_iterator_tag>  { 
# 700
template< class _BI1, class _BI2> static _BI2 
# 703
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 704
{ 
# 706
typename iterator_traits< _BI1> ::difference_type __n = __last - __first; 
# 707
for (; __n > 0; --__n) { 
# 708
(*(--__result)) = (*(--__last)); }  
# 709
return __result; 
# 710
} 
# 711
}; 
# 715
template<> struct __copy_move_backward< true, false, random_access_iterator_tag>  { 
# 717
template< class _BI1, class _BI2> static _BI2 
# 720
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 721
{ 
# 723
typename iterator_traits< _BI1> ::difference_type __n = __last - __first; 
# 724
for (; __n > 0; --__n) { 
# 725
(*(--__result)) = std::move(*(--__last)); }  
# 726
return __result; 
# 727
} 
# 728
}; 
# 731
template< bool _IsMove> 
# 732
struct __copy_move_backward< _IsMove, true, random_access_iterator_tag>  { 
# 734
template< class _Tp, class _Up> static _Up *
# 737
__copy_move_b(_Tp *__first, _Tp *__last, _Up *__result) 
# 738
{ 
# 739
const ptrdiff_t _Num = __last - __first; 
# 740
if (__builtin_expect(_Num > (1), true)) { 
# 741
__builtin_memmove(__result - _Num, __first, sizeof(_Tp) * _Num); } else { 
# 742
if (_Num == (1)) { 
# 743
std::template __copy_move< _IsMove, false, random_access_iterator_tag> ::__assign_one(__result - 1, __first); }  }  
# 745
return __result - _Num; 
# 746
} 
# 747
}; 
# 749
template< bool _IsMove, class _BI1, class _BI2> inline _BI2 
# 752
__copy_move_backward_a2(_BI1 __first, _BI1 __last, _BI2 __result) 
# 753
{ 
# 754
typedef typename iterator_traits< _BI1> ::iterator_category _Category; 
# 760
return std::template __copy_move_backward< _IsMove, __memcpyable< _BI2, _BI1> ::__value, typename iterator_traits< _BI1> ::iterator_category> ::__copy_move_b(__first, __last, __result); 
# 765
} 
# 767
template< bool _IsMove, class _BI1, class _BI2> inline _BI2 
# 770
__copy_move_backward_a1(_BI1 __first, _BI1 __last, _BI2 __result) 
# 771
{ return std::__copy_move_backward_a2< _IsMove> (__first, __last, __result); } 
# 773
template< bool _IsMove, class 
# 774
_Tp, class _Ref, class _Ptr, class _OI> _OI 
# 773
__copy_move_backward_a1(_Deque_iterator< _Tp, _Ref, _Ptr> , _Deque_iterator< _Tp, _Ref, _Ptr> , _OI); 
# 780
template< bool _IsMove, class 
# 781
_ITp, class _IRef, class _IPtr, class _OTp> _Deque_iterator< _OTp, _OTp &, _OTp *>  
# 780
__copy_move_backward_a1(_Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _OTp, _OTp &, _OTp *> ); 
# 788
template< bool _IsMove, class _II, class _Tp> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, _Deque_iterator< _Tp, _Tp &, _Tp *> > ::__type __copy_move_backward_a1(_II, _II, _Deque_iterator< _Tp, _Tp &, _Tp *> ); 
# 795
template< bool _IsMove, class _II, class _OI> inline _OI 
# 798
__copy_move_backward_a(_II __first, _II __last, _OI __result) 
# 799
{ 
# 800
return std::__niter_wrap(__result, std::__copy_move_backward_a1< _IsMove> (std::__niter_base(__first), std::__niter_base(__last), std::__niter_base(__result))); 
# 804
} 
# 806
template< bool _IsMove, class 
# 807
_Ite, class _Seq, class _Cat, class _OI> _OI 
# 806
__copy_move_backward_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, _OI); 
# 814
template< bool _IsMove, class 
# 815
_II, class _Ite, class _Seq, class _Cat> __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  
# 814
__copy_move_backward_a(_II, _II, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &); 
# 820
template< bool _IsMove, class 
# 821
_IIte, class _ISeq, class _ICat, class 
# 822
_OIte, class _OSeq, class _OCat> __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  
# 820
__copy_move_backward_a(const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  &); 
# 847 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _BI1, class _BI2> inline _BI2 
# 850
copy_backward(_BI1 __first, _BI1 __last, _BI2 __result) 
# 851
{ 
# 857
; 
# 859
return std::__copy_move_backward_a< __is_move_iterator< _BI1> ::__value> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 861
} 
# 882 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _BI1, class _BI2> inline _BI2 
# 885
move_backward(_BI1 __first, _BI1 __last, _BI2 __result) 
# 886
{ 
# 892
; 
# 894
return std::__copy_move_backward_a< true> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 897
} 
# 904
template< class _ForwardIterator, class _Tp> inline typename __gnu_cxx::__enable_if< !__is_scalar< _Tp> ::__value, void> ::__type 
# 908
__fill_a1(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 909
__value) 
# 910
{ 
# 911
for (; __first != __last; ++__first) { 
# 912
(*__first) = __value; }  
# 913
} 
# 915
template< class _ForwardIterator, class _Tp> inline typename __gnu_cxx::__enable_if< __is_scalar< _Tp> ::__value, void> ::__type 
# 919
__fill_a1(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 920
__value) 
# 921
{ 
# 922
const _Tp __tmp = __value; 
# 923
for (; __first != __last; ++__first) { 
# 924
(*__first) = __tmp; }  
# 925
} 
# 928
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_byte< _Tp> ::__value, void> ::__type 
# 932
__fill_a1(_Tp *__first, _Tp *__last, const _Tp &__c) 
# 933
{ 
# 934
const _Tp __tmp = __c; 
# 943 "/usr/include/c++/12/bits/stl_algobase.h" 3
if (const size_t __len = __last - __first) { 
# 944
__builtin_memset(__first, static_cast< unsigned char>(__tmp), __len); }  
# 945
} 
# 947
template< class _Ite, class _Cont, class _Tp> inline void 
# 950
__fill_a1(__gnu_cxx::__normal_iterator< _Ite, _Cont>  __first, __gnu_cxx::__normal_iterator< _Ite, _Cont>  
# 951
__last, const _Tp &
# 952
__value) 
# 953
{ std::__fill_a1(__first.base(), __last.base(), __value); } 
# 955
template< class _Tp, class _VTp> void __fill_a1(const _Deque_iterator< _Tp, _Tp &, _Tp *>  &, const _Deque_iterator< _Tp, _Tp &, _Tp *>  &, const _VTp &); 
# 963
void __fill_a1(_Bit_iterator, _Bit_iterator, const bool &); 
# 966
template< class _FIte, class _Tp> inline void 
# 969
__fill_a(_FIte __first, _FIte __last, const _Tp &__value) 
# 970
{ std::__fill_a1(__first, __last, __value); } 
# 972
template< class _Ite, class _Seq, class _Cat, class _Tp> void __fill_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const _Tp &); 
# 990 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _ForwardIterator, class _Tp> inline void 
# 993
fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp &__value) 
# 994
{ 
# 998
; 
# 1000
std::__fill_a(__first, __last, __value); 
# 1001
} 
# 1005
constexpr int __size_to_integer(int __n) { return __n; } 
# 1007
constexpr unsigned __size_to_integer(unsigned __n) { return __n; } 
# 1009
constexpr long __size_to_integer(long __n) { return __n; } 
# 1011
constexpr unsigned long __size_to_integer(unsigned long __n) { return __n; } 
# 1013
constexpr long long __size_to_integer(long long __n) { return __n; } 
# 1015
constexpr unsigned long long __size_to_integer(unsigned long long __n) { return __n; } 
# 1019
__extension__ constexpr __int128 __size_to_integer(__int128 __n) { return __n; } 
# 1021
__extension__ constexpr unsigned __int128 __size_to_integer(unsigned __int128 __n) { return __n; } 
# 1043 "/usr/include/c++/12/bits/stl_algobase.h" 3
constexpr long long __size_to_integer(float __n) { return (long long)__n; } 
# 1045
constexpr long long __size_to_integer(double __n) { return (long long)__n; } 
# 1047
constexpr long long __size_to_integer(long double __n) { return (long long)__n; } 
# 1053
template< class _OutputIterator, class _Size, class _Tp> inline typename __gnu_cxx::__enable_if< !__is_scalar< _Tp> ::__value, _OutputIterator> ::__type 
# 1057
__fill_n_a1(_OutputIterator __first, _Size __n, const _Tp &__value) 
# 1058
{ 
# 1059
for (; __n > 0; (--__n), ((void)(++__first))) { 
# 1060
(*__first) = __value; }  
# 1061
return __first; 
# 1062
} 
# 1064
template< class _OutputIterator, class _Size, class _Tp> inline typename __gnu_cxx::__enable_if< __is_scalar< _Tp> ::__value, _OutputIterator> ::__type 
# 1068
__fill_n_a1(_OutputIterator __first, _Size __n, const _Tp &__value) 
# 1069
{ 
# 1070
const _Tp __tmp = __value; 
# 1071
for (; __n > 0; (--__n), ((void)(++__first))) { 
# 1072
(*__first) = __tmp; }  
# 1073
return __first; 
# 1074
} 
# 1076
template< class _Ite, class _Seq, class _Cat, class _Size, class 
# 1077
_Tp> __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  
# 1076
__fill_n_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  & __first, _Size __n, const _Tp & __value, input_iterator_tag); 
# 1083
template< class _OutputIterator, class _Size, class _Tp> inline _OutputIterator 
# 1086
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value, output_iterator_tag) 
# 1088
{ 
# 1090
static_assert((is_integral< _Size> {}), "fill_n must pass integral size");
# 1092
return __fill_n_a1(__first, __n, __value); 
# 1093
} 
# 1095
template< class _OutputIterator, class _Size, class _Tp> inline _OutputIterator 
# 1098
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value, input_iterator_tag) 
# 1100
{ 
# 1102
static_assert((is_integral< _Size> {}), "fill_n must pass integral size");
# 1104
return __fill_n_a1(__first, __n, __value); 
# 1105
} 
# 1107
template< class _OutputIterator, class _Size, class _Tp> inline _OutputIterator 
# 1110
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value, random_access_iterator_tag) 
# 1112
{ 
# 1114
static_assert((is_integral< _Size> {}), "fill_n must pass integral size");
# 1116
if (__n <= 0) { 
# 1117
return __first; }  
# 1119
; 
# 1121
std::__fill_a(__first, __first + __n, __value); 
# 1122
return __first + __n; 
# 1123
} 
# 1142 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _OI, class _Size, class _Tp> inline _OI 
# 1145
fill_n(_OI __first, _Size __n, const _Tp &__value) 
# 1146
{ 
# 1150
return std::__fill_n_a(__first, std::__size_to_integer(__n), __value, std::__iterator_category(__first)); 
# 1152
} 
# 1154
template< bool _BoolType> 
# 1155
struct __equal { 
# 1157
template< class _II1, class _II2> static bool 
# 1160
equal(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1161
{ 
# 1162
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 1163
if (!((*__first1) == (*__first2))) { 
# 1164
return false; }  }  
# 1165
return true; 
# 1166
} 
# 1167
}; 
# 1170
template<> struct __equal< true>  { 
# 1172
template< class _Tp> static bool 
# 1175
equal(const _Tp *__first1, const _Tp *__last1, const _Tp *__first2) 
# 1176
{ 
# 1177
if (const size_t __len = __last1 - __first1) { 
# 1178
return !std::__memcmp(__first1, __first2, __len); }  
# 1179
return true; 
# 1180
} 
# 1181
}; 
# 1183
template< class _Tp, class _Ref, class _Ptr, class _II> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, bool> ::__type __equal_aux1(_Deque_iterator< _Tp, _Ref, _Ptr> , _Deque_iterator< _Tp, _Ref, _Ptr> , _II); 
# 1190
template< class _Tp1, class _Ref1, class _Ptr1, class 
# 1191
_Tp2, class _Ref2, class _Ptr2> bool 
# 1190
__equal_aux1(_Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> ); 
# 1197
template< class _II, class _Tp, class _Ref, class _Ptr> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, bool> ::__type __equal_aux1(_II, _II, _Deque_iterator< _Tp, _Ref, _Ptr> ); 
# 1203
template< class _II1, class _II2> inline bool 
# 1206
__equal_aux1(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1207
{ 
# 1208
typedef typename iterator_traits< _II1> ::value_type _ValueType1; 
# 1209
const bool __simple = ((__is_integer< typename iterator_traits< _II1> ::value_type> ::__value || __is_pointer< typename iterator_traits< _II1> ::value_type> ::__value) && __memcmpable< _II1, _II2> ::__value); 
# 1212
return std::template __equal< __simple> ::equal(__first1, __last1, __first2); 
# 1213
} 
# 1215
template< class _II1, class _II2> inline bool 
# 1218
__equal_aux(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1219
{ 
# 1220
return std::__equal_aux1(std::__niter_base(__first1), std::__niter_base(__last1), std::__niter_base(__first2)); 
# 1223
} 
# 1225
template< class _II1, class _Seq1, class _Cat1, class _II2> bool __equal_aux(const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, _II2); 
# 1231
template< class _II1, class _II2, class _Seq2, class _Cat2> bool __equal_aux(_II1, _II1, const __gnu_debug::_Safe_iterator< _II2, _Seq2, _Cat2>  &); 
# 1236
template< class _II1, class _Seq1, class _Cat1, class 
# 1237
_II2, class _Seq2, class _Cat2> bool 
# 1236
__equal_aux(const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _II2, _Seq2, _Cat2>  &); 
# 1243
template< class , class > 
# 1244
struct __lc_rai { 
# 1246
template< class _II1, class _II2> static _II1 
# 1249
__newlast1(_II1, _II1 __last1, _II2, _II2) 
# 1250
{ return __last1; } 
# 1252
template< class _II> static bool 
# 1255
__cnd2(_II __first, _II __last) 
# 1256
{ return __first != __last; } 
# 1257
}; 
# 1260
template<> struct __lc_rai< random_access_iterator_tag, random_access_iterator_tag>  { 
# 1262
template< class _RAI1, class _RAI2> static _RAI1 
# 1265
__newlast1(_RAI1 __first1, _RAI1 __last1, _RAI2 
# 1266
__first2, _RAI2 __last2) 
# 1267
{ 
# 1269
const typename iterator_traits< _RAI1> ::difference_type __diff1 = __last1 - __first1; 
# 1271
const typename iterator_traits< _RAI2> ::difference_type __diff2 = __last2 - __first2; 
# 1272
return (__diff2 < __diff1) ? __first1 + __diff2 : __last1; 
# 1273
} 
# 1275
template< class _RAI> static bool 
# 1277
__cnd2(_RAI, _RAI) 
# 1278
{ return true; } 
# 1279
}; 
# 1281
template< class _II1, class _II2, class _Compare> bool 
# 1284
__lexicographical_compare_impl(_II1 __first1, _II1 __last1, _II2 
# 1285
__first2, _II2 __last2, _Compare 
# 1286
__comp) 
# 1287
{ 
# 1288
typedef typename iterator_traits< _II1> ::iterator_category _Category1; 
# 1289
typedef typename iterator_traits< _II2> ::iterator_category _Category2; 
# 1290
typedef __lc_rai< typename iterator_traits< _II1> ::iterator_category, typename iterator_traits< _II2> ::iterator_category>  __rai_type; 
# 1292
__last1 = __rai_type::__newlast1(__first1, __last1, __first2, __last2); 
# 1293
for (; (__first1 != __last1) && __rai_type::__cnd2(__first2, __last2); (++__first1), ((void)(++__first2))) 
# 1295
{ 
# 1296
if (__comp(__first1, __first2)) { 
# 1297
return true; }  
# 1298
if (__comp(__first2, __first1)) { 
# 1299
return false; }  
# 1300
}  
# 1301
return (__first1 == __last1) && (__first2 != __last2); 
# 1302
} 
# 1304
template< bool _BoolType> 
# 1305
struct __lexicographical_compare { 
# 1307
template< class _II1, class _II2> static bool 
# 1310
__lc(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1311
{ 
# 1312
using __gnu_cxx::__ops::__iter_less_iter;
# 1313
return std::__lexicographical_compare_impl(__first1, __last1, __first2, __last2, __iter_less_iter()); 
# 1316
} 
# 1318
template< class _II1, class _II2> static int 
# 1321
__3way(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1322
{ 
# 1323
while (__first1 != __last1) 
# 1324
{ 
# 1325
if (__first2 == __last2) { 
# 1326
return +1; }  
# 1327
if ((*__first1) < (*__first2)) { 
# 1328
return -1; }  
# 1329
if ((*__first2) < (*__first1)) { 
# 1330
return +1; }  
# 1331
++__first1; 
# 1332
++__first2; 
# 1333
}  
# 1334
return ((int)(__first2 == __last2)) - 1; 
# 1335
} 
# 1336
}; 
# 1339
template<> struct __lexicographical_compare< true>  { 
# 1341
template< class _Tp, class _Up> static bool 
# 1344
__lc(const _Tp *__first1, const _Tp *__last1, const _Up *
# 1345
__first2, const _Up *__last2) 
# 1346
{ return __3way(__first1, __last1, __first2, __last2) < 0; } 
# 1348
template< class _Tp, class _Up> static ptrdiff_t 
# 1351
__3way(const _Tp *__first1, const _Tp *__last1, const _Up *
# 1352
__first2, const _Up *__last2) 
# 1353
{ 
# 1354
const size_t __len1 = __last1 - __first1; 
# 1355
const size_t __len2 = __last2 - __first2; 
# 1356
if (const size_t __len = std::min(__len1, __len2)) { 
# 1357
if (int __result = std::__memcmp(__first1, __first2, __len)) { 
# 1358
return __result; }  }  
# 1359
return (ptrdiff_t)(__len1 - __len2); 
# 1360
} 
# 1361
}; 
# 1363
template< class _II1, class _II2> inline bool 
# 1366
__lexicographical_compare_aux1(_II1 __first1, _II1 __last1, _II2 
# 1367
__first2, _II2 __last2) 
# 1368
{ 
# 1369
typedef typename iterator_traits< _II1> ::value_type _ValueType1; 
# 1370
typedef typename iterator_traits< _II2> ::value_type _ValueType2; 
# 1371
const bool __simple = (__is_memcmp_ordered_with< typename iterator_traits< _II1> ::value_type, typename iterator_traits< _II2> ::value_type> ::__value && __is_pointer< _II1> ::__value && __is_pointer< _II2> ::__value); 
# 1384
return std::template __lexicographical_compare< __simple> ::__lc(__first1, __last1, __first2, __last2); 
# 1386
} 
# 1388
template< class _Tp1, class _Ref1, class _Ptr1, class 
# 1389
_Tp2> bool 
# 1388
__lexicographical_compare_aux1(_Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Tp2 *, _Tp2 *); 
# 1396
template< class _Tp1, class 
# 1397
_Tp2, class _Ref2, class _Ptr2> bool 
# 1396
__lexicographical_compare_aux1(_Tp1 *, _Tp1 *, _Deque_iterator< _Tp2, _Ref2, _Ptr2> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> ); 
# 1403
template< class _Tp1, class _Ref1, class _Ptr1, class 
# 1404
_Tp2, class _Ref2, class _Ptr2> bool 
# 1403
__lexicographical_compare_aux1(_Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> ); 
# 1412
template< class _II1, class _II2> inline bool 
# 1415
__lexicographical_compare_aux(_II1 __first1, _II1 __last1, _II2 
# 1416
__first2, _II2 __last2) 
# 1417
{ 
# 1418
return std::__lexicographical_compare_aux1(std::__niter_base(__first1), std::__niter_base(__last1), std::__niter_base(__first2), std::__niter_base(__last2)); 
# 1422
} 
# 1424
template< class _Iter1, class _Seq1, class _Cat1, class 
# 1425
_II2> bool 
# 1424
__lexicographical_compare_aux(const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, _II2, _II2); 
# 1432
template< class _II1, class 
# 1433
_Iter2, class _Seq2, class _Cat2> bool 
# 1432
__lexicographical_compare_aux(_II1, _II1, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &); 
# 1440
template< class _Iter1, class _Seq1, class _Cat1, class 
# 1441
_Iter2, class _Seq2, class _Cat2> bool 
# 1440
__lexicographical_compare_aux(const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &); 
# 1449
template< class _ForwardIterator, class _Tp, class _Compare> _ForwardIterator 
# 1452
__lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 1453
__val, _Compare __comp) 
# 1454
{ 
# 1456
typedef typename iterator_traits< _ForwardIterator> ::difference_type _DistanceType; 
# 1458
_DistanceType __len = std::distance(__first, __last); 
# 1460
while (__len > 0) 
# 1461
{ 
# 1462
_DistanceType __half = __len >> 1; 
# 1463
_ForwardIterator __middle = __first; 
# 1464
std::advance(__middle, __half); 
# 1465
if (__comp(__middle, __val)) 
# 1466
{ 
# 1467
__first = __middle; 
# 1468
++__first; 
# 1469
__len = ((__len - __half) - 1); 
# 1470
} else { 
# 1472
__len = __half; }  
# 1473
}  
# 1474
return __first; 
# 1475
} 
# 1488 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _ForwardIterator, class _Tp> inline _ForwardIterator 
# 1491
lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 1492
__val) 
# 1493
{ 
# 1498
; 
# 1500
return std::__lower_bound(__first, __last, __val, __gnu_cxx::__ops::__iter_less_val()); 
# 1502
} 
# 1507
constexpr int __lg(int __n) 
# 1508
{ return ((((int)sizeof(int)) * 8) - 1) - __builtin_clz(__n); } 
# 1511
constexpr unsigned __lg(unsigned __n) 
# 1512
{ return ((((int)sizeof(int)) * 8) - 1) - __builtin_clz(__n); } 
# 1515
constexpr long __lg(long __n) 
# 1516
{ return ((((int)sizeof(long)) * 8) - 1) - __builtin_clzl(__n); } 
# 1519
constexpr unsigned long __lg(unsigned long __n) 
# 1520
{ return ((((int)sizeof(long)) * 8) - 1) - __builtin_clzl(__n); } 
# 1523
constexpr long long __lg(long long __n) 
# 1524
{ return ((((int)sizeof(long long)) * 8) - 1) - __builtin_clzll(__n); } 
# 1527
constexpr unsigned long long __lg(unsigned long long __n) 
# 1528
{ return ((((int)sizeof(long long)) * 8) - 1) - __builtin_clzll(__n); } 
# 1544 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _II1, class _II2> inline bool 
# 1547
equal(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1548
{ 
# 1555
; 
# 1557
return std::__equal_aux(__first1, __last1, __first2); 
# 1558
} 
# 1575 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _IIter1, class _IIter2, class _BinaryPredicate> inline bool 
# 1578
equal(_IIter1 __first1, _IIter1 __last1, _IIter2 
# 1579
__first2, _BinaryPredicate __binary_pred) 
# 1580
{ 
# 1584
; 
# 1586
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 1587
if (!((bool)__binary_pred(*__first1, *__first2))) { 
# 1588
return false; }  }  
# 1589
return true; 
# 1590
} 
# 1594
template< class _II1, class _II2> inline bool 
# 1597
__equal4(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1598
{ 
# 1599
using _RATag = random_access_iterator_tag; 
# 1600
using _Cat1 = typename iterator_traits< _II1> ::iterator_category; 
# 1601
using _Cat2 = typename iterator_traits< _II2> ::iterator_category; 
# 1602
using _RAIters = __and_< is_same< typename iterator_traits< _II1> ::iterator_category, random_access_iterator_tag> , is_same< typename iterator_traits< _II2> ::iterator_category, random_access_iterator_tag> > ; 
# 1603
if (_RAIters()) 
# 1604
{ 
# 1605
auto __d1 = std::distance(__first1, __last1); 
# 1606
auto __d2 = std::distance(__first2, __last2); 
# 1607
if (__d1 != __d2) { 
# 1608
return false; }  
# 1609
return std::equal(__first1, __last1, __first2); 
# 1610
}  
# 1612
for (; (__first1 != __last1) && (__first2 != __last2); (++__first1), ((void)(++__first2))) { 
# 1614
if (!((*__first1) == (*__first2))) { 
# 1615
return false; }  }  
# 1616
return (__first1 == __last1) && (__first2 == __last2); 
# 1617
} 
# 1620
template< class _II1, class _II2, class _BinaryPredicate> inline bool 
# 1623
__equal4(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2, _BinaryPredicate 
# 1624
__binary_pred) 
# 1625
{ 
# 1626
using _RATag = random_access_iterator_tag; 
# 1627
using _Cat1 = typename iterator_traits< _II1> ::iterator_category; 
# 1628
using _Cat2 = typename iterator_traits< _II2> ::iterator_category; 
# 1629
using _RAIters = __and_< is_same< typename iterator_traits< _II1> ::iterator_category, random_access_iterator_tag> , is_same< typename iterator_traits< _II2> ::iterator_category, random_access_iterator_tag> > ; 
# 1630
if (_RAIters()) 
# 1631
{ 
# 1632
auto __d1 = std::distance(__first1, __last1); 
# 1633
auto __d2 = std::distance(__first2, __last2); 
# 1634
if (__d1 != __d2) { 
# 1635
return false; }  
# 1636
return std::equal(__first1, __last1, __first2, __binary_pred); 
# 1638
}  
# 1640
for (; (__first1 != __last1) && (__first2 != __last2); (++__first1), ((void)(++__first2))) { 
# 1642
if (!((bool)__binary_pred(*__first1, *__first2))) { 
# 1643
return false; }  }  
# 1644
return (__first1 == __last1) && (__first2 == __last2); 
# 1645
} 
# 1665 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _II1, class _II2> inline bool 
# 1668
equal(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1669
{ 
# 1676
; 
# 1677
; 
# 1679
return std::__equal4(__first1, __last1, __first2, __last2); 
# 1680
} 
# 1698 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _IIter1, class _IIter2, class _BinaryPredicate> inline bool 
# 1701
equal(_IIter1 __first1, _IIter1 __last1, _IIter2 
# 1702
__first2, _IIter2 __last2, _BinaryPredicate __binary_pred) 
# 1703
{ 
# 1707
; 
# 1708
; 
# 1710
return std::__equal4(__first1, __last1, __first2, __last2, __binary_pred); 
# 1712
} 
# 1730 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _II1, class _II2> inline bool 
# 1733
lexicographical_compare(_II1 __first1, _II1 __last1, _II2 
# 1734
__first2, _II2 __last2) 
# 1735
{ 
# 1745
; 
# 1746
; 
# 1748
return std::__lexicographical_compare_aux(__first1, __last1, __first2, __last2); 
# 1750
} 
# 1765 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _II1, class _II2, class _Compare> inline bool 
# 1768
lexicographical_compare(_II1 __first1, _II1 __last1, _II2 
# 1769
__first2, _II2 __last2, _Compare __comp) 
# 1770
{ 
# 1774
; 
# 1775
; 
# 1777
return std::__lexicographical_compare_impl(__first1, __last1, __first2, __last2, __gnu_cxx::__ops::__iter_comp_iter(__comp)); 
# 1780
} 
# 1880 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _InputIterator1, class _InputIterator2, class 
# 1881
_BinaryPredicate> pair< _InputIterator1, _InputIterator2>  
# 1884
__mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1885
__first2, _BinaryPredicate __binary_pred) 
# 1886
{ 
# 1887
while ((__first1 != __last1) && __binary_pred(__first1, __first2)) 
# 1888
{ 
# 1889
++__first1; 
# 1890
++__first2; 
# 1891
}  
# 1892
return pair< _InputIterator1, _InputIterator2> (__first1, __first2); 
# 1893
} 
# 1908 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _InputIterator1, class _InputIterator2> inline pair< _InputIterator1, _InputIterator2>  
# 1911
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1912
__first2) 
# 1913
{ 
# 1920
; 
# 1922
return std::__mismatch(__first1, __last1, __first2, __gnu_cxx::__ops::__iter_equal_to_iter()); 
# 1924
} 
# 1942 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _InputIterator1, class _InputIterator2, class 
# 1943
_BinaryPredicate> inline pair< _InputIterator1, _InputIterator2>  
# 1946
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1947
__first2, _BinaryPredicate __binary_pred) 
# 1948
{ 
# 1952
; 
# 1954
return std::__mismatch(__first1, __last1, __first2, __gnu_cxx::__ops::__iter_comp_iter(__binary_pred)); 
# 1956
} 
# 1960
template< class _InputIterator1, class _InputIterator2, class 
# 1961
_BinaryPredicate> pair< _InputIterator1, _InputIterator2>  
# 1964
__mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1965
__first2, _InputIterator2 __last2, _BinaryPredicate 
# 1966
__binary_pred) 
# 1967
{ 
# 1968
while ((__first1 != __last1) && (__first2 != __last2) && __binary_pred(__first1, __first2)) 
# 1970
{ 
# 1971
++__first1; 
# 1972
++__first2; 
# 1973
}  
# 1974
return pair< _InputIterator1, _InputIterator2> (__first1, __first2); 
# 1975
} 
# 1991 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _InputIterator1, class _InputIterator2> inline pair< _InputIterator1, _InputIterator2>  
# 1994
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1995
__first2, _InputIterator2 __last2) 
# 1996
{ 
# 2003
; 
# 2004
; 
# 2006
return std::__mismatch(__first1, __last1, __first2, __last2, __gnu_cxx::__ops::__iter_equal_to_iter()); 
# 2008
} 
# 2027 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _InputIterator1, class _InputIterator2, class 
# 2028
_BinaryPredicate> inline pair< _InputIterator1, _InputIterator2>  
# 2031
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 2032
__first2, _InputIterator2 __last2, _BinaryPredicate 
# 2033
__binary_pred) 
# 2034
{ 
# 2038
; 
# 2039
; 
# 2041
return std::__mismatch(__first1, __last1, __first2, __last2, __gnu_cxx::__ops::__iter_comp_iter(__binary_pred)); 
# 2043
} 
# 2049
template< class _InputIterator, class _Predicate> inline _InputIterator 
# 2052
__find_if(_InputIterator __first, _InputIterator __last, _Predicate 
# 2053
__pred, input_iterator_tag) 
# 2054
{ 
# 2055
while ((__first != __last) && (!__pred(__first))) { 
# 2056
++__first; }  
# 2057
return __first; 
# 2058
} 
# 2061
template< class _RandomAccessIterator, class _Predicate> _RandomAccessIterator 
# 2064
__find_if(_RandomAccessIterator __first, _RandomAccessIterator __last, _Predicate 
# 2065
__pred, random_access_iterator_tag) 
# 2066
{ 
# 2068
typename iterator_traits< _RandomAccessIterator> ::difference_type __trip_count = (__last - __first) >> 2; 
# 2070
for (; __trip_count > 0; --__trip_count) 
# 2071
{ 
# 2072
if (__pred(__first)) { 
# 2073
return __first; }  
# 2074
++__first; 
# 2076
if (__pred(__first)) { 
# 2077
return __first; }  
# 2078
++__first; 
# 2080
if (__pred(__first)) { 
# 2081
return __first; }  
# 2082
++__first; 
# 2084
if (__pred(__first)) { 
# 2085
return __first; }  
# 2086
++__first; 
# 2087
}  
# 2089
switch (__last - __first) 
# 2090
{ 
# 2091
case 3:  
# 2092
if (__pred(__first)) { 
# 2093
return __first; }  
# 2094
++__first; 
# 2096
case 2:  
# 2097
if (__pred(__first)) { 
# 2098
return __first; }  
# 2099
++__first; 
# 2101
case 1:  
# 2102
if (__pred(__first)) { 
# 2103
return __first; }  
# 2104
++__first; 
# 2106
case 0:  
# 2107
default:  
# 2108
return __last; 
# 2109
}  
# 2110
} 
# 2112
template< class _Iterator, class _Predicate> inline _Iterator 
# 2115
__find_if(_Iterator __first, _Iterator __last, _Predicate __pred) 
# 2116
{ 
# 2117
return __find_if(__first, __last, __pred, std::__iterator_category(__first)); 
# 2119
} 
# 2121
template< class _InputIterator, class _Predicate> typename iterator_traits< _InputIterator> ::difference_type 
# 2124
__count_if(_InputIterator __first, _InputIterator __last, _Predicate __pred) 
# 2125
{ 
# 2126
typename iterator_traits< _InputIterator> ::difference_type __n = (0); 
# 2127
for (; __first != __last; ++__first) { 
# 2128
if (__pred(__first)) { 
# 2129
++__n; }  }  
# 2130
return __n; 
# 2131
} 
# 2133
template< class _ForwardIterator, class _Predicate> _ForwardIterator 
# 2136
__remove_if(_ForwardIterator __first, _ForwardIterator __last, _Predicate 
# 2137
__pred) 
# 2138
{ 
# 2139
__first = std::__find_if(__first, __last, __pred); 
# 2140
if (__first == __last) { 
# 2141
return __first; }  
# 2142
_ForwardIterator __result = __first; 
# 2143
++__first; 
# 2144
for (; __first != __last; ++__first) { 
# 2145
if (!__pred(__first)) 
# 2146
{ 
# 2147
(*__result) = std::move(*__first); 
# 2148
++__result; 
# 2149
}  }  
# 2150
return __result; 
# 2151
} 
# 2154
template< class _ForwardIterator1, class _ForwardIterator2, class 
# 2155
_BinaryPredicate> bool 
# 2158
__is_permutation(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 
# 2159
__first2, _BinaryPredicate __pred) 
# 2160
{ 
# 2163
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 2164
if (!__pred(__first1, __first2)) { 
# 2165
break; }  }  
# 2167
if (__first1 == __last1) { 
# 2168
return true; }  
# 2172
_ForwardIterator2 __last2 = __first2; 
# 2173
std::advance(__last2, std::distance(__first1, __last1)); 
# 2174
for (_ForwardIterator1 __scan = __first1; __scan != __last1; ++__scan) 
# 2175
{ 
# 2176
if (__scan != std::__find_if(__first1, __scan, __gnu_cxx::__ops::__iter_comp_iter(__pred, __scan))) { 
# 2178
continue; }  
# 2180
auto __matches = std::__count_if(__first2, __last2, __gnu_cxx::__ops::__iter_comp_iter(__pred, __scan)); 
# 2183
if ((0 == __matches) || (std::__count_if(__scan, __last1, __gnu_cxx::__ops::__iter_comp_iter(__pred, __scan)) != __matches)) { 
# 2187
return false; }  
# 2188
}   
# 2189
return true; 
# 2190
} 
# 2204 "/usr/include/c++/12/bits/stl_algobase.h" 3
template< class _ForwardIterator1, class _ForwardIterator2> inline bool 
# 2207
is_permutation(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 
# 2208
__first2) 
# 2209
{ 
# 2216
; 
# 2218
return std::__is_permutation(__first1, __last1, __first2, __gnu_cxx::__ops::__iter_equal_to_iter()); 
# 2220
} 
# 2224
}
# 158 "/usr/include/c++/12/limits" 3
namespace std __attribute((__visibility__("default"))) { 
# 167
enum float_round_style { 
# 169
round_indeterminate = (-1), 
# 170
round_toward_zero = 0, 
# 171
round_to_nearest, 
# 172
round_toward_infinity, 
# 173
round_toward_neg_infinity
# 174
}; 
# 182
enum float_denorm_style { 
# 185
denorm_indeterminate = (-1), 
# 187
denorm_absent = 0, 
# 189
denorm_present
# 190
}; 
# 202 "/usr/include/c++/12/limits" 3
struct __numeric_limits_base { 
# 206
static constexpr inline bool is_specialized = false; 
# 211
static constexpr inline int digits = 0; 
# 214
static constexpr inline int digits10 = 0; 
# 219
static constexpr inline int max_digits10 = 0; 
# 223
static constexpr inline bool is_signed = false; 
# 226
static constexpr inline bool is_integer = false; 
# 231
static constexpr inline bool is_exact = false; 
# 235
static constexpr inline int radix = 0; 
# 239
static constexpr inline int min_exponent = 0; 
# 243
static constexpr inline int min_exponent10 = 0; 
# 248
static constexpr inline int max_exponent = 0; 
# 252
static constexpr inline int max_exponent10 = 0; 
# 255
static constexpr inline bool has_infinity = false; 
# 259
static constexpr inline bool has_quiet_NaN = false; 
# 263
static constexpr inline bool has_signaling_NaN = false; 
# 266
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 270
static constexpr inline bool has_denorm_loss = false; 
# 274
static constexpr inline bool is_iec559 = false; 
# 279
static constexpr inline bool is_bounded = false; 
# 288 "/usr/include/c++/12/limits" 3
static constexpr inline bool is_modulo = false; 
# 291
static constexpr inline bool traps = false; 
# 294
static constexpr inline bool tinyness_before = false; 
# 299
static constexpr inline float_round_style round_style = round_toward_zero; 
# 301
}; 
# 311 "/usr/include/c++/12/limits" 3
template< class _Tp> 
# 312
struct numeric_limits : public __numeric_limits_base { 
# 317
static constexpr _Tp min() noexcept { return _Tp(); } 
# 321
static constexpr _Tp max() noexcept { return _Tp(); } 
# 327
static constexpr _Tp lowest() noexcept { return _Tp(); } 
# 333
static constexpr _Tp epsilon() noexcept { return _Tp(); } 
# 337
static constexpr _Tp round_error() noexcept { return _Tp(); } 
# 341
static constexpr _Tp infinity() noexcept { return _Tp(); } 
# 346
static constexpr _Tp quiet_NaN() noexcept { return _Tp(); } 
# 351
static constexpr _Tp signaling_NaN() noexcept { return _Tp(); } 
# 357
static constexpr _Tp denorm_min() noexcept { return _Tp(); } 
# 358
}; 
# 363
template< class _Tp> 
# 364
struct numeric_limits< const _Tp>  : public std::numeric_limits< _Tp>  { 
# 365
}; 
# 367
template< class _Tp> 
# 368
struct numeric_limits< volatile _Tp>  : public std::numeric_limits< _Tp>  { 
# 369
}; 
# 371
template< class _Tp> 
# 372
struct numeric_limits< const volatile _Tp>  : public std::numeric_limits< _Tp>  { 
# 373
}; 
# 384 "/usr/include/c++/12/limits" 3
template<> struct numeric_limits< bool>  { 
# 386
static constexpr inline bool is_specialized = true; 
# 389
static constexpr bool min() noexcept { return false; } 
# 392
static constexpr bool max() noexcept { return true; } 
# 396
static constexpr bool lowest() noexcept { return min(); } 
# 398
static constexpr inline int digits = 1; 
# 399
static constexpr inline int digits10 = 0; 
# 401
static constexpr inline int max_digits10 = 0; 
# 403
static constexpr inline bool is_signed = false; 
# 404
static constexpr inline bool is_integer = true; 
# 405
static constexpr inline bool is_exact = true; 
# 406
static constexpr inline int radix = 2; 
# 409
static constexpr bool epsilon() noexcept { return false; } 
# 412
static constexpr bool round_error() noexcept { return false; } 
# 414
static constexpr inline int min_exponent = 0; 
# 415
static constexpr inline int min_exponent10 = 0; 
# 416
static constexpr inline int max_exponent = 0; 
# 417
static constexpr inline int max_exponent10 = 0; 
# 419
static constexpr inline bool has_infinity = false; 
# 420
static constexpr inline bool has_quiet_NaN = false; 
# 421
static constexpr inline bool has_signaling_NaN = false; 
# 422
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 424
static constexpr inline bool has_denorm_loss = false; 
# 427
static constexpr bool infinity() noexcept { return false; } 
# 430
static constexpr bool quiet_NaN() noexcept { return false; } 
# 433
static constexpr bool signaling_NaN() noexcept { return false; } 
# 436
static constexpr bool denorm_min() noexcept { return false; } 
# 438
static constexpr inline bool is_iec559 = false; 
# 439
static constexpr inline bool is_bounded = true; 
# 440
static constexpr inline bool is_modulo = false; 
# 445
static constexpr inline bool traps = true; 
# 446
static constexpr inline bool tinyness_before = false; 
# 447
static constexpr inline float_round_style round_style = round_toward_zero; 
# 449
}; 
# 453
template<> struct numeric_limits< char>  { 
# 455
static constexpr inline bool is_specialized = true; 
# 458
static constexpr char min() noexcept { return ((((char)(-1)) < 0) ? (-((((char)(-1)) < 0) ? (((((char)1) << (((sizeof(char) * (8)) - (((char)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char)0)))) - 1 : ((char)0)); } 
# 461
static constexpr char max() noexcept { return ((((char)(-1)) < 0) ? (((((char)1) << (((sizeof(char) * (8)) - (((char)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char)0))); } 
# 465
static constexpr char lowest() noexcept { return min(); } 
# 468
static constexpr inline int digits = ((sizeof(char) * (8)) - (((char)(-1)) < 0)); 
# 469
static constexpr inline int digits10 = ((((sizeof(char) * (8)) - (((char)(-1)) < 0)) * (643L)) / (2136)); 
# 471
static constexpr inline int max_digits10 = 0; 
# 473
static constexpr inline bool is_signed = (((char)(-1)) < 0); 
# 474
static constexpr inline bool is_integer = true; 
# 475
static constexpr inline bool is_exact = true; 
# 476
static constexpr inline int radix = 2; 
# 479
static constexpr char epsilon() noexcept { return 0; } 
# 482
static constexpr char round_error() noexcept { return 0; } 
# 484
static constexpr inline int min_exponent = 0; 
# 485
static constexpr inline int min_exponent10 = 0; 
# 486
static constexpr inline int max_exponent = 0; 
# 487
static constexpr inline int max_exponent10 = 0; 
# 489
static constexpr inline bool has_infinity = false; 
# 490
static constexpr inline bool has_quiet_NaN = false; 
# 491
static constexpr inline bool has_signaling_NaN = false; 
# 492
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 494
static constexpr inline bool has_denorm_loss = false; 
# 497
static constexpr char infinity() noexcept { return ((char)0); } 
# 500
static constexpr char quiet_NaN() noexcept { return ((char)0); } 
# 503
static constexpr char signaling_NaN() noexcept { return ((char)0); } 
# 506
static constexpr char denorm_min() noexcept { return static_cast< char>(0); } 
# 508
static constexpr inline bool is_iec559 = false; 
# 509
static constexpr inline bool is_bounded = true; 
# 510
static constexpr inline bool is_modulo = (!is_signed); 
# 512
static constexpr inline bool traps = true; 
# 513
static constexpr inline bool tinyness_before = false; 
# 514
static constexpr inline float_round_style round_style = round_toward_zero; 
# 516
}; 
# 520
template<> struct numeric_limits< signed char>  { 
# 522
static constexpr inline bool is_specialized = true; 
# 525
static constexpr signed char min() noexcept { return (-127) - 1; } 
# 528
static constexpr signed char max() noexcept { return 127; } 
# 532
static constexpr signed char lowest() noexcept { return min(); } 
# 535
static constexpr inline int digits = ((sizeof(signed char) * (8)) - (((signed char)(-1)) < 0)); 
# 536
static constexpr inline int digits10 = ((((sizeof(signed char) * (8)) - (((signed char)(-1)) < 0)) * (643L)) / (2136)); 
# 539
static constexpr inline int max_digits10 = 0; 
# 541
static constexpr inline bool is_signed = true; 
# 542
static constexpr inline bool is_integer = true; 
# 543
static constexpr inline bool is_exact = true; 
# 544
static constexpr inline int radix = 2; 
# 547
static constexpr signed char epsilon() noexcept { return 0; } 
# 550
static constexpr signed char round_error() noexcept { return 0; } 
# 552
static constexpr inline int min_exponent = 0; 
# 553
static constexpr inline int min_exponent10 = 0; 
# 554
static constexpr inline int max_exponent = 0; 
# 555
static constexpr inline int max_exponent10 = 0; 
# 557
static constexpr inline bool has_infinity = false; 
# 558
static constexpr inline bool has_quiet_NaN = false; 
# 559
static constexpr inline bool has_signaling_NaN = false; 
# 560
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 562
static constexpr inline bool has_denorm_loss = false; 
# 565
static constexpr signed char infinity() noexcept { return static_cast< signed char>(0); } 
# 568
static constexpr signed char quiet_NaN() noexcept { return static_cast< signed char>(0); } 
# 571
static constexpr signed char signaling_NaN() noexcept 
# 572
{ return static_cast< signed char>(0); } 
# 575
static constexpr signed char denorm_min() noexcept 
# 576
{ return static_cast< signed char>(0); } 
# 578
static constexpr inline bool is_iec559 = false; 
# 579
static constexpr inline bool is_bounded = true; 
# 580
static constexpr inline bool is_modulo = false; 
# 582
static constexpr inline bool traps = true; 
# 583
static constexpr inline bool tinyness_before = false; 
# 584
static constexpr inline float_round_style round_style = round_toward_zero; 
# 586
}; 
# 590
template<> struct numeric_limits< unsigned char>  { 
# 592
static constexpr inline bool is_specialized = true; 
# 595
static constexpr unsigned char min() noexcept { return 0; } 
# 598
static constexpr unsigned char max() noexcept { return ((127) * 2U) + (1); } 
# 602
static constexpr unsigned char lowest() noexcept { return min(); } 
# 605
static constexpr inline int digits = ((sizeof(unsigned char) * (8)) - (((unsigned char)(-1)) < 0)); 
# 607
static constexpr inline int digits10 = ((((sizeof(unsigned char) * (8)) - (((unsigned char)(-1)) < 0)) * (643L)) / (2136)); 
# 610
static constexpr inline int max_digits10 = 0; 
# 612
static constexpr inline bool is_signed = false; 
# 613
static constexpr inline bool is_integer = true; 
# 614
static constexpr inline bool is_exact = true; 
# 615
static constexpr inline int radix = 2; 
# 618
static constexpr unsigned char epsilon() noexcept { return 0; } 
# 621
static constexpr unsigned char round_error() noexcept { return 0; } 
# 623
static constexpr inline int min_exponent = 0; 
# 624
static constexpr inline int min_exponent10 = 0; 
# 625
static constexpr inline int max_exponent = 0; 
# 626
static constexpr inline int max_exponent10 = 0; 
# 628
static constexpr inline bool has_infinity = false; 
# 629
static constexpr inline bool has_quiet_NaN = false; 
# 630
static constexpr inline bool has_signaling_NaN = false; 
# 631
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 633
static constexpr inline bool has_denorm_loss = false; 
# 636
static constexpr unsigned char infinity() noexcept 
# 637
{ return static_cast< unsigned char>(0); } 
# 640
static constexpr unsigned char quiet_NaN() noexcept 
# 641
{ return static_cast< unsigned char>(0); } 
# 644
static constexpr unsigned char signaling_NaN() noexcept 
# 645
{ return static_cast< unsigned char>(0); } 
# 648
static constexpr unsigned char denorm_min() noexcept 
# 649
{ return static_cast< unsigned char>(0); } 
# 651
static constexpr inline bool is_iec559 = false; 
# 652
static constexpr inline bool is_bounded = true; 
# 653
static constexpr inline bool is_modulo = true; 
# 655
static constexpr inline bool traps = true; 
# 656
static constexpr inline bool tinyness_before = false; 
# 657
static constexpr inline float_round_style round_style = round_toward_zero; 
# 659
}; 
# 663
template<> struct numeric_limits< wchar_t>  { 
# 665
static constexpr inline bool is_specialized = true; 
# 668
static constexpr wchar_t min() noexcept { return ((((wchar_t)(-1)) < 0) ? (-((((wchar_t)(-1)) < 0) ? (((((wchar_t)1) << (((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((wchar_t)0)))) - 1 : ((wchar_t)0)); } 
# 671
static constexpr wchar_t max() noexcept { return ((((wchar_t)(-1)) < 0) ? (((((wchar_t)1) << (((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((wchar_t)0))); } 
# 675
static constexpr wchar_t lowest() noexcept { return min(); } 
# 678
static constexpr inline int digits = ((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)); 
# 679
static constexpr inline int digits10 = ((((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)) * (643L)) / (2136)); 
# 682
static constexpr inline int max_digits10 = 0; 
# 684
static constexpr inline bool is_signed = (((wchar_t)(-1)) < 0); 
# 685
static constexpr inline bool is_integer = true; 
# 686
static constexpr inline bool is_exact = true; 
# 687
static constexpr inline int radix = 2; 
# 690
static constexpr wchar_t epsilon() noexcept { return 0; } 
# 693
static constexpr wchar_t round_error() noexcept { return 0; } 
# 695
static constexpr inline int min_exponent = 0; 
# 696
static constexpr inline int min_exponent10 = 0; 
# 697
static constexpr inline int max_exponent = 0; 
# 698
static constexpr inline int max_exponent10 = 0; 
# 700
static constexpr inline bool has_infinity = false; 
# 701
static constexpr inline bool has_quiet_NaN = false; 
# 702
static constexpr inline bool has_signaling_NaN = false; 
# 703
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 705
static constexpr inline bool has_denorm_loss = false; 
# 708
static constexpr wchar_t infinity() noexcept { return ((wchar_t)0); } 
# 711
static constexpr wchar_t quiet_NaN() noexcept { return ((wchar_t)0); } 
# 714
static constexpr wchar_t signaling_NaN() noexcept { return ((wchar_t)0); } 
# 717
static constexpr wchar_t denorm_min() noexcept { return ((wchar_t)0); } 
# 719
static constexpr inline bool is_iec559 = false; 
# 720
static constexpr inline bool is_bounded = true; 
# 721
static constexpr inline bool is_modulo = (!is_signed); 
# 723
static constexpr inline bool traps = true; 
# 724
static constexpr inline bool tinyness_before = false; 
# 725
static constexpr inline float_round_style round_style = round_toward_zero; 
# 727
}; 
# 797 "/usr/include/c++/12/limits" 3
template<> struct numeric_limits< char16_t>  { 
# 799
static constexpr inline bool is_specialized = true; 
# 802
static constexpr char16_t min() noexcept { return ((((char16_t)(-1)) < 0) ? (-((((char16_t)(-1)) < 0) ? (((((char16_t)1) << (((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char16_t)0)))) - 1 : ((char16_t)0)); } 
# 805
static constexpr char16_t max() noexcept { return ((((char16_t)(-1)) < 0) ? (((((char16_t)1) << (((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char16_t)0))); } 
# 808
static constexpr char16_t lowest() noexcept { return min(); } 
# 810
static constexpr inline int digits = ((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)); 
# 811
static constexpr inline int digits10 = ((((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)) * (643L)) / (2136)); 
# 812
static constexpr inline int max_digits10 = 0; 
# 813
static constexpr inline bool is_signed = (((char16_t)(-1)) < 0); 
# 814
static constexpr inline bool is_integer = true; 
# 815
static constexpr inline bool is_exact = true; 
# 816
static constexpr inline int radix = 2; 
# 819
static constexpr char16_t epsilon() noexcept { return 0; } 
# 822
static constexpr char16_t round_error() noexcept { return 0; } 
# 824
static constexpr inline int min_exponent = 0; 
# 825
static constexpr inline int min_exponent10 = 0; 
# 826
static constexpr inline int max_exponent = 0; 
# 827
static constexpr inline int max_exponent10 = 0; 
# 829
static constexpr inline bool has_infinity = false; 
# 830
static constexpr inline bool has_quiet_NaN = false; 
# 831
static constexpr inline bool has_signaling_NaN = false; 
# 832
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 833
static constexpr inline bool has_denorm_loss = false; 
# 836
static constexpr char16_t infinity() noexcept { return ((char16_t)0); } 
# 839
static constexpr char16_t quiet_NaN() noexcept { return ((char16_t)0); } 
# 842
static constexpr char16_t signaling_NaN() noexcept { return ((char16_t)0); } 
# 845
static constexpr char16_t denorm_min() noexcept { return ((char16_t)0); } 
# 847
static constexpr inline bool is_iec559 = false; 
# 848
static constexpr inline bool is_bounded = true; 
# 849
static constexpr inline bool is_modulo = (!is_signed); 
# 851
static constexpr inline bool traps = true; 
# 852
static constexpr inline bool tinyness_before = false; 
# 853
static constexpr inline float_round_style round_style = round_toward_zero; 
# 854
}; 
# 858
template<> struct numeric_limits< char32_t>  { 
# 860
static constexpr inline bool is_specialized = true; 
# 863
static constexpr char32_t min() noexcept { return ((((char32_t)(-1)) < (0)) ? (-((((char32_t)(-1)) < (0)) ? (((((char32_t)1) << (((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))) - (1))) - (1)) << 1) + (1) : (~((char32_t)0)))) - (1) : ((char32_t)0)); } 
# 866
static constexpr char32_t max() noexcept { return ((((char32_t)(-1)) < (0)) ? (((((char32_t)1) << (((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))) - (1))) - (1)) << 1) + (1) : (~((char32_t)0))); } 
# 869
static constexpr char32_t lowest() noexcept { return min(); } 
# 871
static constexpr inline int digits = ((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))); 
# 872
static constexpr inline int digits10 = ((((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))) * (643L)) / (2136)); 
# 873
static constexpr inline int max_digits10 = 0; 
# 874
static constexpr inline bool is_signed = (((char32_t)(-1)) < (0)); 
# 875
static constexpr inline bool is_integer = true; 
# 876
static constexpr inline bool is_exact = true; 
# 877
static constexpr inline int radix = 2; 
# 880
static constexpr char32_t epsilon() noexcept { return 0; } 
# 883
static constexpr char32_t round_error() noexcept { return 0; } 
# 885
static constexpr inline int min_exponent = 0; 
# 886
static constexpr inline int min_exponent10 = 0; 
# 887
static constexpr inline int max_exponent = 0; 
# 888
static constexpr inline int max_exponent10 = 0; 
# 890
static constexpr inline bool has_infinity = false; 
# 891
static constexpr inline bool has_quiet_NaN = false; 
# 892
static constexpr inline bool has_signaling_NaN = false; 
# 893
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 894
static constexpr inline bool has_denorm_loss = false; 
# 897
static constexpr char32_t infinity() noexcept { return ((char32_t)0); } 
# 900
static constexpr char32_t quiet_NaN() noexcept { return ((char32_t)0); } 
# 903
static constexpr char32_t signaling_NaN() noexcept { return ((char32_t)0); } 
# 906
static constexpr char32_t denorm_min() noexcept { return ((char32_t)0); } 
# 908
static constexpr inline bool is_iec559 = false; 
# 909
static constexpr inline bool is_bounded = true; 
# 910
static constexpr inline bool is_modulo = (!is_signed); 
# 912
static constexpr inline bool traps = true; 
# 913
static constexpr inline bool tinyness_before = false; 
# 914
static constexpr inline float_round_style round_style = round_toward_zero; 
# 915
}; 
# 920
template<> struct numeric_limits< short>  { 
# 922
static constexpr inline bool is_specialized = true; 
# 925
static constexpr short min() noexcept { return (-32767) - 1; } 
# 928
static constexpr short max() noexcept { return 32767; } 
# 932
static constexpr short lowest() noexcept { return min(); } 
# 935
static constexpr inline int digits = ((sizeof(short) * (8)) - (((short)(-1)) < 0)); 
# 936
static constexpr inline int digits10 = ((((sizeof(short) * (8)) - (((short)(-1)) < 0)) * (643L)) / (2136)); 
# 938
static constexpr inline int max_digits10 = 0; 
# 940
static constexpr inline bool is_signed = true; 
# 941
static constexpr inline bool is_integer = true; 
# 942
static constexpr inline bool is_exact = true; 
# 943
static constexpr inline int radix = 2; 
# 946
static constexpr short epsilon() noexcept { return 0; } 
# 949
static constexpr short round_error() noexcept { return 0; } 
# 951
static constexpr inline int min_exponent = 0; 
# 952
static constexpr inline int min_exponent10 = 0; 
# 953
static constexpr inline int max_exponent = 0; 
# 954
static constexpr inline int max_exponent10 = 0; 
# 956
static constexpr inline bool has_infinity = false; 
# 957
static constexpr inline bool has_quiet_NaN = false; 
# 958
static constexpr inline bool has_signaling_NaN = false; 
# 959
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 961
static constexpr inline bool has_denorm_loss = false; 
# 964
static constexpr short infinity() noexcept { return ((short)0); } 
# 967
static constexpr short quiet_NaN() noexcept { return ((short)0); } 
# 970
static constexpr short signaling_NaN() noexcept { return ((short)0); } 
# 973
static constexpr short denorm_min() noexcept { return ((short)0); } 
# 975
static constexpr inline bool is_iec559 = false; 
# 976
static constexpr inline bool is_bounded = true; 
# 977
static constexpr inline bool is_modulo = false; 
# 979
static constexpr inline bool traps = true; 
# 980
static constexpr inline bool tinyness_before = false; 
# 981
static constexpr inline float_round_style round_style = round_toward_zero; 
# 983
}; 
# 987
template<> struct numeric_limits< unsigned short>  { 
# 989
static constexpr inline bool is_specialized = true; 
# 992
static constexpr unsigned short min() noexcept { return 0; } 
# 995
static constexpr unsigned short max() noexcept { return ((32767) * 2U) + (1); } 
# 999
static constexpr unsigned short lowest() noexcept { return min(); } 
# 1002
static constexpr inline int digits = ((sizeof(unsigned short) * (8)) - (((unsigned short)(-1)) < 0)); 
# 1004
static constexpr inline int digits10 = ((((sizeof(unsigned short) * (8)) - (((unsigned short)(-1)) < 0)) * (643L)) / (2136)); 
# 1007
static constexpr inline int max_digits10 = 0; 
# 1009
static constexpr inline bool is_signed = false; 
# 1010
static constexpr inline bool is_integer = true; 
# 1011
static constexpr inline bool is_exact = true; 
# 1012
static constexpr inline int radix = 2; 
# 1015
static constexpr unsigned short epsilon() noexcept { return 0; } 
# 1018
static constexpr unsigned short round_error() noexcept { return 0; } 
# 1020
static constexpr inline int min_exponent = 0; 
# 1021
static constexpr inline int min_exponent10 = 0; 
# 1022
static constexpr inline int max_exponent = 0; 
# 1023
static constexpr inline int max_exponent10 = 0; 
# 1025
static constexpr inline bool has_infinity = false; 
# 1026
static constexpr inline bool has_quiet_NaN = false; 
# 1027
static constexpr inline bool has_signaling_NaN = false; 
# 1028
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1030
static constexpr inline bool has_denorm_loss = false; 
# 1033
static constexpr unsigned short infinity() noexcept 
# 1034
{ return static_cast< unsigned short>(0); } 
# 1037
static constexpr unsigned short quiet_NaN() noexcept 
# 1038
{ return static_cast< unsigned short>(0); } 
# 1041
static constexpr unsigned short signaling_NaN() noexcept 
# 1042
{ return static_cast< unsigned short>(0); } 
# 1045
static constexpr unsigned short denorm_min() noexcept 
# 1046
{ return static_cast< unsigned short>(0); } 
# 1048
static constexpr inline bool is_iec559 = false; 
# 1049
static constexpr inline bool is_bounded = true; 
# 1050
static constexpr inline bool is_modulo = true; 
# 1052
static constexpr inline bool traps = true; 
# 1053
static constexpr inline bool tinyness_before = false; 
# 1054
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1056
}; 
# 1060
template<> struct numeric_limits< int>  { 
# 1062
static constexpr inline bool is_specialized = true; 
# 1065
static constexpr int min() noexcept { return (-2147483647) - 1; } 
# 1068
static constexpr int max() noexcept { return 2147483647; } 
# 1072
static constexpr int lowest() noexcept { return min(); } 
# 1075
static constexpr inline int digits = ((sizeof(int) * (8)) - (((int)(-1)) < 0)); 
# 1076
static constexpr inline int digits10 = ((((sizeof(int) * (8)) - (((int)(-1)) < 0)) * (643L)) / (2136)); 
# 1078
static constexpr inline int max_digits10 = 0; 
# 1080
static constexpr inline bool is_signed = true; 
# 1081
static constexpr inline bool is_integer = true; 
# 1082
static constexpr inline bool is_exact = true; 
# 1083
static constexpr inline int radix = 2; 
# 1086
static constexpr int epsilon() noexcept { return 0; } 
# 1089
static constexpr int round_error() noexcept { return 0; } 
# 1091
static constexpr inline int min_exponent = 0; 
# 1092
static constexpr inline int min_exponent10 = 0; 
# 1093
static constexpr inline int max_exponent = 0; 
# 1094
static constexpr inline int max_exponent10 = 0; 
# 1096
static constexpr inline bool has_infinity = false; 
# 1097
static constexpr inline bool has_quiet_NaN = false; 
# 1098
static constexpr inline bool has_signaling_NaN = false; 
# 1099
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1101
static constexpr inline bool has_denorm_loss = false; 
# 1104
static constexpr int infinity() noexcept { return static_cast< int>(0); } 
# 1107
static constexpr int quiet_NaN() noexcept { return static_cast< int>(0); } 
# 1110
static constexpr int signaling_NaN() noexcept { return static_cast< int>(0); } 
# 1113
static constexpr int denorm_min() noexcept { return static_cast< int>(0); } 
# 1115
static constexpr inline bool is_iec559 = false; 
# 1116
static constexpr inline bool is_bounded = true; 
# 1117
static constexpr inline bool is_modulo = false; 
# 1119
static constexpr inline bool traps = true; 
# 1120
static constexpr inline bool tinyness_before = false; 
# 1121
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1123
}; 
# 1127
template<> struct numeric_limits< unsigned>  { 
# 1129
static constexpr inline bool is_specialized = true; 
# 1132
static constexpr unsigned min() noexcept { return 0; } 
# 1135
static constexpr unsigned max() noexcept { return ((2147483647) * 2U) + (1); } 
# 1139
static constexpr unsigned lowest() noexcept { return min(); } 
# 1142
static constexpr inline int digits = ((sizeof(unsigned) * (8)) - (((unsigned)(-1)) < (0))); 
# 1144
static constexpr inline int digits10 = ((((sizeof(unsigned) * (8)) - (((unsigned)(-1)) < (0))) * (643L)) / (2136)); 
# 1147
static constexpr inline int max_digits10 = 0; 
# 1149
static constexpr inline bool is_signed = false; 
# 1150
static constexpr inline bool is_integer = true; 
# 1151
static constexpr inline bool is_exact = true; 
# 1152
static constexpr inline int radix = 2; 
# 1155
static constexpr unsigned epsilon() noexcept { return 0; } 
# 1158
static constexpr unsigned round_error() noexcept { return 0; } 
# 1160
static constexpr inline int min_exponent = 0; 
# 1161
static constexpr inline int min_exponent10 = 0; 
# 1162
static constexpr inline int max_exponent = 0; 
# 1163
static constexpr inline int max_exponent10 = 0; 
# 1165
static constexpr inline bool has_infinity = false; 
# 1166
static constexpr inline bool has_quiet_NaN = false; 
# 1167
static constexpr inline bool has_signaling_NaN = false; 
# 1168
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1170
static constexpr inline bool has_denorm_loss = false; 
# 1173
static constexpr unsigned infinity() noexcept { return static_cast< unsigned>(0); } 
# 1176
static constexpr unsigned quiet_NaN() noexcept 
# 1177
{ return static_cast< unsigned>(0); } 
# 1180
static constexpr unsigned signaling_NaN() noexcept 
# 1181
{ return static_cast< unsigned>(0); } 
# 1184
static constexpr unsigned denorm_min() noexcept 
# 1185
{ return static_cast< unsigned>(0); } 
# 1187
static constexpr inline bool is_iec559 = false; 
# 1188
static constexpr inline bool is_bounded = true; 
# 1189
static constexpr inline bool is_modulo = true; 
# 1191
static constexpr inline bool traps = true; 
# 1192
static constexpr inline bool tinyness_before = false; 
# 1193
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1195
}; 
# 1199
template<> struct numeric_limits< long>  { 
# 1201
static constexpr inline bool is_specialized = true; 
# 1204
static constexpr long min() noexcept { return (-9223372036854775807L) - (1); } 
# 1207
static constexpr long max() noexcept { return 9223372036854775807L; } 
# 1211
static constexpr long lowest() noexcept { return min(); } 
# 1214
static constexpr inline int digits = ((sizeof(long) * (8)) - (((long)(-1)) < (0))); 
# 1215
static constexpr inline int digits10 = ((((sizeof(long) * (8)) - (((long)(-1)) < (0))) * (643L)) / (2136)); 
# 1217
static constexpr inline int max_digits10 = 0; 
# 1219
static constexpr inline bool is_signed = true; 
# 1220
static constexpr inline bool is_integer = true; 
# 1221
static constexpr inline bool is_exact = true; 
# 1222
static constexpr inline int radix = 2; 
# 1225
static constexpr long epsilon() noexcept { return 0; } 
# 1228
static constexpr long round_error() noexcept { return 0; } 
# 1230
static constexpr inline int min_exponent = 0; 
# 1231
static constexpr inline int min_exponent10 = 0; 
# 1232
static constexpr inline int max_exponent = 0; 
# 1233
static constexpr inline int max_exponent10 = 0; 
# 1235
static constexpr inline bool has_infinity = false; 
# 1236
static constexpr inline bool has_quiet_NaN = false; 
# 1237
static constexpr inline bool has_signaling_NaN = false; 
# 1238
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1240
static constexpr inline bool has_denorm_loss = false; 
# 1243
static constexpr long infinity() noexcept { return static_cast< long>(0); } 
# 1246
static constexpr long quiet_NaN() noexcept { return static_cast< long>(0); } 
# 1249
static constexpr long signaling_NaN() noexcept { return static_cast< long>(0); } 
# 1252
static constexpr long denorm_min() noexcept { return static_cast< long>(0); } 
# 1254
static constexpr inline bool is_iec559 = false; 
# 1255
static constexpr inline bool is_bounded = true; 
# 1256
static constexpr inline bool is_modulo = false; 
# 1258
static constexpr inline bool traps = true; 
# 1259
static constexpr inline bool tinyness_before = false; 
# 1260
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1262
}; 
# 1266
template<> struct numeric_limits< unsigned long>  { 
# 1268
static constexpr inline bool is_specialized = true; 
# 1271
static constexpr unsigned long min() noexcept { return 0; } 
# 1274
static constexpr unsigned long max() noexcept { return ((9223372036854775807L) * 2UL) + (1); } 
# 1278
static constexpr unsigned long lowest() noexcept { return min(); } 
# 1281
static constexpr inline int digits = ((sizeof(unsigned long) * (8)) - (((unsigned long)(-1)) < (0))); 
# 1283
static constexpr inline int digits10 = ((((sizeof(unsigned long) * (8)) - (((unsigned long)(-1)) < (0))) * (643L)) / (2136)); 
# 1286
static constexpr inline int max_digits10 = 0; 
# 1288
static constexpr inline bool is_signed = false; 
# 1289
static constexpr inline bool is_integer = true; 
# 1290
static constexpr inline bool is_exact = true; 
# 1291
static constexpr inline int radix = 2; 
# 1294
static constexpr unsigned long epsilon() noexcept { return 0; } 
# 1297
static constexpr unsigned long round_error() noexcept { return 0; } 
# 1299
static constexpr inline int min_exponent = 0; 
# 1300
static constexpr inline int min_exponent10 = 0; 
# 1301
static constexpr inline int max_exponent = 0; 
# 1302
static constexpr inline int max_exponent10 = 0; 
# 1304
static constexpr inline bool has_infinity = false; 
# 1305
static constexpr inline bool has_quiet_NaN = false; 
# 1306
static constexpr inline bool has_signaling_NaN = false; 
# 1307
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1309
static constexpr inline bool has_denorm_loss = false; 
# 1312
static constexpr unsigned long infinity() noexcept 
# 1313
{ return static_cast< unsigned long>(0); } 
# 1316
static constexpr unsigned long quiet_NaN() noexcept 
# 1317
{ return static_cast< unsigned long>(0); } 
# 1320
static constexpr unsigned long signaling_NaN() noexcept 
# 1321
{ return static_cast< unsigned long>(0); } 
# 1324
static constexpr unsigned long denorm_min() noexcept 
# 1325
{ return static_cast< unsigned long>(0); } 
# 1327
static constexpr inline bool is_iec559 = false; 
# 1328
static constexpr inline bool is_bounded = true; 
# 1329
static constexpr inline bool is_modulo = true; 
# 1331
static constexpr inline bool traps = true; 
# 1332
static constexpr inline bool tinyness_before = false; 
# 1333
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1335
}; 
# 1339
template<> struct numeric_limits< long long>  { 
# 1341
static constexpr inline bool is_specialized = true; 
# 1344
static constexpr long long min() noexcept { return (-9223372036854775807LL) - (1); } 
# 1347
static constexpr long long max() noexcept { return 9223372036854775807LL; } 
# 1351
static constexpr long long lowest() noexcept { return min(); } 
# 1354
static constexpr inline int digits = ((sizeof(long long) * (8)) - (((long long)(-1)) < (0))); 
# 1356
static constexpr inline int digits10 = ((((sizeof(long long) * (8)) - (((long long)(-1)) < (0))) * (643L)) / (2136)); 
# 1359
static constexpr inline int max_digits10 = 0; 
# 1361
static constexpr inline bool is_signed = true; 
# 1362
static constexpr inline bool is_integer = true; 
# 1363
static constexpr inline bool is_exact = true; 
# 1364
static constexpr inline int radix = 2; 
# 1367
static constexpr long long epsilon() noexcept { return 0; } 
# 1370
static constexpr long long round_error() noexcept { return 0; } 
# 1372
static constexpr inline int min_exponent = 0; 
# 1373
static constexpr inline int min_exponent10 = 0; 
# 1374
static constexpr inline int max_exponent = 0; 
# 1375
static constexpr inline int max_exponent10 = 0; 
# 1377
static constexpr inline bool has_infinity = false; 
# 1378
static constexpr inline bool has_quiet_NaN = false; 
# 1379
static constexpr inline bool has_signaling_NaN = false; 
# 1380
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1382
static constexpr inline bool has_denorm_loss = false; 
# 1385
static constexpr long long infinity() noexcept { return static_cast< long long>(0); } 
# 1388
static constexpr long long quiet_NaN() noexcept { return static_cast< long long>(0); } 
# 1391
static constexpr long long signaling_NaN() noexcept 
# 1392
{ return static_cast< long long>(0); } 
# 1395
static constexpr long long denorm_min() noexcept { return static_cast< long long>(0); } 
# 1397
static constexpr inline bool is_iec559 = false; 
# 1398
static constexpr inline bool is_bounded = true; 
# 1399
static constexpr inline bool is_modulo = false; 
# 1401
static constexpr inline bool traps = true; 
# 1402
static constexpr inline bool tinyness_before = false; 
# 1403
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1405
}; 
# 1409
template<> struct numeric_limits< unsigned long long>  { 
# 1411
static constexpr inline bool is_specialized = true; 
# 1414
static constexpr unsigned long long min() noexcept { return 0; } 
# 1417
static constexpr unsigned long long max() noexcept { return ((9223372036854775807LL) * 2ULL) + (1); } 
# 1421
static constexpr unsigned long long lowest() noexcept { return min(); } 
# 1424
static constexpr inline int digits = ((sizeof(unsigned long long) * (8)) - (((unsigned long long)(-1)) < (0))); 
# 1426
static constexpr inline int digits10 = ((((sizeof(unsigned long long) * (8)) - (((unsigned long long)(-1)) < (0))) * (643L)) / (2136)); 
# 1429
static constexpr inline int max_digits10 = 0; 
# 1431
static constexpr inline bool is_signed = false; 
# 1432
static constexpr inline bool is_integer = true; 
# 1433
static constexpr inline bool is_exact = true; 
# 1434
static constexpr inline int radix = 2; 
# 1437
static constexpr unsigned long long epsilon() noexcept { return 0; } 
# 1440
static constexpr unsigned long long round_error() noexcept { return 0; } 
# 1442
static constexpr inline int min_exponent = 0; 
# 1443
static constexpr inline int min_exponent10 = 0; 
# 1444
static constexpr inline int max_exponent = 0; 
# 1445
static constexpr inline int max_exponent10 = 0; 
# 1447
static constexpr inline bool has_infinity = false; 
# 1448
static constexpr inline bool has_quiet_NaN = false; 
# 1449
static constexpr inline bool has_signaling_NaN = false; 
# 1450
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1452
static constexpr inline bool has_denorm_loss = false; 
# 1455
static constexpr unsigned long long infinity() noexcept 
# 1456
{ return static_cast< unsigned long long>(0); } 
# 1459
static constexpr unsigned long long quiet_NaN() noexcept 
# 1460
{ return static_cast< unsigned long long>(0); } 
# 1463
static constexpr unsigned long long signaling_NaN() noexcept 
# 1464
{ return static_cast< unsigned long long>(0); } 
# 1467
static constexpr unsigned long long denorm_min() noexcept 
# 1468
{ return static_cast< unsigned long long>(0); } 
# 1470
static constexpr inline bool is_iec559 = false; 
# 1471
static constexpr inline bool is_bounded = true; 
# 1472
static constexpr inline bool is_modulo = true; 
# 1474
static constexpr inline bool traps = true; 
# 1475
static constexpr inline bool tinyness_before = false; 
# 1476
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1478
}; 
# 1637 "/usr/include/c++/12/limits" 3
template<> struct numeric_limits< __int128>  { static constexpr inline bool is_specialized = true; static constexpr __int128 min() noexcept { return ((((__int128)(-1)) < (0)) ? (-((((__int128)(-1)) < (0)) ? (((((__int128)1) << ((128 - (((__int128)(-1)) < (0))) - 1)) - (1)) << 1) + (1) : (~((__int128)0)))) - (1) : ((__int128)0)); } static constexpr __int128 max() noexcept { return ((((__int128)(-1)) < (0)) ? (((((__int128)1) << ((128 - (((__int128)(-1)) < (0))) - 1)) - (1)) << 1) + (1) : (~((__int128)0))); } static constexpr inline int digits = (128 - 1); static constexpr inline int digits10 = (((128 - 1) * 643L) / (2136)); static constexpr inline bool is_signed = true; static constexpr inline bool is_integer = true; static constexpr inline bool is_exact = true; static constexpr inline int radix = 2; static constexpr __int128 epsilon() noexcept { return 0; } static constexpr __int128 round_error() noexcept { return 0; } static constexpr __int128 lowest() noexcept { return min(); } static constexpr inline int max_digits10 = 0; static constexpr inline int min_exponent = 0; static constexpr inline int min_exponent10 = 0; static constexpr inline int max_exponent = 0; static constexpr inline int max_exponent10 = 0; static constexpr inline bool has_infinity = false; static constexpr inline bool has_quiet_NaN = false; static constexpr inline bool has_signaling_NaN = false; static constexpr inline float_denorm_style has_denorm = denorm_absent; static constexpr inline bool has_denorm_loss = false; static constexpr __int128 infinity() noexcept { return static_cast< __int128>(0); } static constexpr __int128 quiet_NaN() noexcept { return static_cast< __int128>(0); } static constexpr __int128 signaling_NaN() noexcept { return static_cast< __int128>(0); } static constexpr __int128 denorm_min() noexcept { return static_cast< __int128>(0); } static constexpr inline bool is_iec559 = false; static constexpr inline bool is_bounded = true; static constexpr inline bool is_modulo = false; static constexpr inline bool traps = true; static constexpr inline bool tinyness_before = false; static constexpr inline float_round_style round_style = round_toward_zero; }; template<> struct numeric_limits< unsigned __int128>  { static constexpr inline bool is_specialized = true; static constexpr unsigned __int128 min() noexcept { return 0; } static constexpr unsigned __int128 max() noexcept { return ((((unsigned __int128)(-1)) < (0)) ? (((((unsigned __int128)1) << ((128 - (((unsigned __int128)(-1)) < (0))) - 1)) - (1)) << 1) + (1) : (~((unsigned __int128)0))); } static constexpr unsigned __int128 lowest() noexcept { return min(); } static constexpr inline int max_digits10 = 0; static constexpr inline int digits = 128; static constexpr inline int digits10 = (((128) * 643L) / (2136)); static constexpr inline bool is_signed = false; static constexpr inline bool is_integer = true; static constexpr inline bool is_exact = true; static constexpr inline int radix = 2; static constexpr unsigned __int128 epsilon() noexcept { return 0; } static constexpr unsigned __int128 round_error() noexcept { return 0; } static constexpr inline int min_exponent = 0; static constexpr inline int min_exponent10 = 0; static constexpr inline int max_exponent = 0; static constexpr inline int max_exponent10 = 0; static constexpr inline bool has_infinity = false; static constexpr inline bool has_quiet_NaN = false; static constexpr inline bool has_signaling_NaN = false; static constexpr inline float_denorm_style has_denorm = denorm_absent; static constexpr inline bool has_denorm_loss = false; static constexpr unsigned __int128 infinity() noexcept { return static_cast< unsigned __int128>(0); } static constexpr unsigned __int128 quiet_NaN() noexcept { return static_cast< unsigned __int128>(0); } static constexpr unsigned __int128 signaling_NaN() noexcept { return static_cast< unsigned __int128>(0); } static constexpr unsigned __int128 denorm_min() noexcept { return static_cast< unsigned __int128>(0); } static constexpr inline bool is_iec559 = false; static constexpr inline bool is_bounded = true; static constexpr inline bool is_modulo = true; static constexpr inline bool traps = true; static constexpr inline bool tinyness_before = false; static constexpr inline float_round_style round_style = round_toward_zero; }; 
# 1670 "/usr/include/c++/12/limits" 3
template<> struct numeric_limits< float>  { 
# 1672
static constexpr inline bool is_specialized = true; 
# 1675
static constexpr float min() noexcept { return (1.1754944E-38F); } 
# 1678
static constexpr float max() noexcept { return (3.4028235E38F); } 
# 1682
static constexpr float lowest() noexcept { return -(3.4028235E38F); } 
# 1685
static constexpr inline int digits = 24; 
# 1686
static constexpr inline int digits10 = 6; 
# 1688
static constexpr inline int max_digits10 = ((2) + (((24) * 643L) / (2136))); 
# 1691
static constexpr inline bool is_signed = true; 
# 1692
static constexpr inline bool is_integer = false; 
# 1693
static constexpr inline bool is_exact = false; 
# 1694
static constexpr inline int radix = 2; 
# 1697
static constexpr float epsilon() noexcept { return (1.1920929E-7F); } 
# 1700
static constexpr float round_error() noexcept { return (0.5F); } 
# 1702
static constexpr inline int min_exponent = (-125); 
# 1703
static constexpr inline int min_exponent10 = (-37); 
# 1704
static constexpr inline int max_exponent = 128; 
# 1705
static constexpr inline int max_exponent10 = 38; 
# 1707
static constexpr inline bool has_infinity = (1); 
# 1708
static constexpr inline bool has_quiet_NaN = (1); 
# 1709
static constexpr inline bool has_signaling_NaN = has_quiet_NaN; 
# 1710
static constexpr inline float_denorm_style has_denorm = (((bool)1) ? denorm_present : denorm_absent); 
# 1712
static constexpr inline bool has_denorm_loss = false; 
# 1716
static constexpr float infinity() noexcept { return __builtin_huge_valf(); } 
# 1719
static constexpr float quiet_NaN() noexcept { return __builtin_nanf(""); } 
# 1722
static constexpr float signaling_NaN() noexcept { return __builtin_nansf(""); } 
# 1725
static constexpr float denorm_min() noexcept { return (1.4E-45F); } 
# 1727
static constexpr inline bool is_iec559 = (has_infinity && has_quiet_NaN && (has_denorm == (denorm_present))); 
# 1729
static constexpr inline bool is_bounded = true; 
# 1730
static constexpr inline bool is_modulo = false; 
# 1732
static constexpr inline bool traps = false; 
# 1733
static constexpr inline bool tinyness_before = false; 
# 1735
static constexpr inline float_round_style round_style = round_to_nearest; 
# 1737
}; 
# 1745
template<> struct numeric_limits< double>  { 
# 1747
static constexpr inline bool is_specialized = true; 
# 1750
static constexpr double min() noexcept { return (double)(2.2250738585072013831E-308L); } 
# 1753
static constexpr double max() noexcept { return (double)(1.7976931348623157081E308L); } 
# 1757
static constexpr double lowest() noexcept { return -((double)(1.7976931348623157081E308L)); } 
# 1760
static constexpr inline int digits = 53; 
# 1761
static constexpr inline int digits10 = 15; 
# 1763
static constexpr inline int max_digits10 = ((2) + (((53) * 643L) / (2136))); 
# 1766
static constexpr inline bool is_signed = true; 
# 1767
static constexpr inline bool is_integer = false; 
# 1768
static constexpr inline bool is_exact = false; 
# 1769
static constexpr inline int radix = 2; 
# 1772
static constexpr double epsilon() noexcept { return (double)(2.2204460492503130808E-16L); } 
# 1775
static constexpr double round_error() noexcept { return (0.5); } 
# 1777
static constexpr inline int min_exponent = (-1021); 
# 1778
static constexpr inline int min_exponent10 = (-307); 
# 1779
static constexpr inline int max_exponent = 1024; 
# 1780
static constexpr inline int max_exponent10 = 308; 
# 1782
static constexpr inline bool has_infinity = (1); 
# 1783
static constexpr inline bool has_quiet_NaN = (1); 
# 1784
static constexpr inline bool has_signaling_NaN = has_quiet_NaN; 
# 1785
static constexpr inline float_denorm_style has_denorm = (((bool)1) ? denorm_present : denorm_absent); 
# 1787
static constexpr inline bool has_denorm_loss = false; 
# 1791
static constexpr double infinity() noexcept { return __builtin_huge_val(); } 
# 1794
static constexpr double quiet_NaN() noexcept { return __builtin_nan(""); } 
# 1797
static constexpr double signaling_NaN() noexcept { return __builtin_nans(""); } 
# 1800
static constexpr double denorm_min() noexcept { return (double)(4.940656458412465442E-324L); } 
# 1802
static constexpr inline bool is_iec559 = (has_infinity && has_quiet_NaN && (has_denorm == (denorm_present))); 
# 1804
static constexpr inline bool is_bounded = true; 
# 1805
static constexpr inline bool is_modulo = false; 
# 1807
static constexpr inline bool traps = false; 
# 1808
static constexpr inline bool tinyness_before = false; 
# 1810
static constexpr inline float_round_style round_style = round_to_nearest; 
# 1812
}; 
# 1820
template<> struct numeric_limits< long double>  { 
# 1822
static constexpr inline bool is_specialized = true; 
# 1825
static constexpr long double min() noexcept { return (3.3621031431120935063E-4932L); } 
# 1828
static constexpr long double max() noexcept { return (1.189731495357231765E4932L); } 
# 1832
static constexpr long double lowest() noexcept { return -(1.189731495357231765E4932L); } 
# 1835
static constexpr inline int digits = 64; 
# 1836
static constexpr inline int digits10 = 18; 
# 1838
static constexpr inline int max_digits10 = ((2) + (((64) * 643L) / (2136))); 
# 1841
static constexpr inline bool is_signed = true; 
# 1842
static constexpr inline bool is_integer = false; 
# 1843
static constexpr inline bool is_exact = false; 
# 1844
static constexpr inline int radix = 2; 
# 1847
static constexpr long double epsilon() noexcept { return (1.084202172485504434E-19L); } 
# 1850
static constexpr long double round_error() noexcept { return (0.5L); } 
# 1852
static constexpr inline int min_exponent = (-16381); 
# 1853
static constexpr inline int min_exponent10 = (-4931); 
# 1854
static constexpr inline int max_exponent = 16384; 
# 1855
static constexpr inline int max_exponent10 = 4932; 
# 1857
static constexpr inline bool has_infinity = (1); 
# 1858
static constexpr inline bool has_quiet_NaN = (1); 
# 1859
static constexpr inline bool has_signaling_NaN = has_quiet_NaN; 
# 1860
static constexpr inline float_denorm_style has_denorm = (((bool)1) ? denorm_present : denorm_absent); 
# 1862
static constexpr inline bool has_denorm_loss = false; 
# 1866
static constexpr long double infinity() noexcept { return __builtin_huge_vall(); } 
# 1869
static constexpr long double quiet_NaN() noexcept { return __builtin_nanl(""); } 
# 1872
static constexpr long double signaling_NaN() noexcept { return __builtin_nansl(""); } 
# 1875
static constexpr long double denorm_min() noexcept { return (3.6E-4951L); } 
# 1877
static constexpr inline bool is_iec559 = (has_infinity && has_quiet_NaN && (has_denorm == (denorm_present))); 
# 1879
static constexpr inline bool is_bounded = true; 
# 1880
static constexpr inline bool is_modulo = false; 
# 1882
static constexpr inline bool traps = false; 
# 1883
static constexpr inline bool tinyness_before = false; 
# 1885
static constexpr inline float_round_style round_style = round_to_nearest; 
# 1887
}; 
# 1894
}
# 39 "/usr/include/c++/12/tr1/special_function_util.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 50 "/usr/include/c++/12/tr1/special_function_util.h" 3
namespace __detail { 
# 55
template< class _Tp> 
# 56
struct __floating_point_constant { 
# 58
static const _Tp __value; 
# 59
}; 
# 63
template< class _Tp> 
# 64
struct __numeric_constants { 
# 67
static _Tp __pi() throw() 
# 68
{ return static_cast< _Tp>((3.1415926535897932385L)); } 
# 70
static _Tp __pi_2() throw() 
# 71
{ return static_cast< _Tp>((1.5707963267948966193L)); } 
# 73
static _Tp __pi_3() throw() 
# 74
{ return static_cast< _Tp>((1.0471975511965977461L)); } 
# 76
static _Tp __pi_4() throw() 
# 77
{ return static_cast< _Tp>((0.78539816339744830963L)); } 
# 79
static _Tp __1_pi() throw() 
# 80
{ return static_cast< _Tp>((0.31830988618379067154L)); } 
# 82
static _Tp __2_sqrtpi() throw() 
# 83
{ return static_cast< _Tp>((1.1283791670955125738L)); } 
# 85
static _Tp __sqrt2() throw() 
# 86
{ return static_cast< _Tp>((1.4142135623730950488L)); } 
# 88
static _Tp __sqrt3() throw() 
# 89
{ return static_cast< _Tp>((1.7320508075688772936L)); } 
# 91
static _Tp __sqrtpio2() throw() 
# 92
{ return static_cast< _Tp>((1.2533141373155002512L)); } 
# 94
static _Tp __sqrt1_2() throw() 
# 95
{ return static_cast< _Tp>((0.7071067811865475244L)); } 
# 97
static _Tp __lnpi() throw() 
# 98
{ return static_cast< _Tp>((1.1447298858494001742L)); } 
# 100
static _Tp __gamma_e() throw() 
# 101
{ return static_cast< _Tp>((0.5772156649015328606L)); } 
# 103
static _Tp __euler() throw() 
# 104
{ return static_cast< _Tp>((2.7182818284590452354L)); } 
# 105
}; 
# 114 "/usr/include/c++/12/tr1/special_function_util.h" 3
template< class _Tp> inline bool 
# 115
__isnan(_Tp __x) 
# 116
{ return std::isnan(__x); } 
# 133 "/usr/include/c++/12/tr1/special_function_util.h" 3
}
# 139
}
# 51 "/usr/include/c++/12/tr1/gamma.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/usr/include/c++/12/tr1/gamma.tcc" 3
namespace __detail { 
# 76 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 78
__bernoulli_series(unsigned __n) 
# 79
{ 
# 81
static const _Tp __num[28] = {((_Tp)1UL), ((-((_Tp)1UL)) / ((_Tp)2UL)), (((_Tp)1UL) / ((_Tp)6UL)), ((_Tp)0UL), ((-((_Tp)1UL)) / ((_Tp)30UL)), ((_Tp)0UL), (((_Tp)1UL) / ((_Tp)42UL)), ((_Tp)0UL), ((-((_Tp)1UL)) / ((_Tp)30UL)), ((_Tp)0UL), (((_Tp)5UL) / ((_Tp)66UL)), ((_Tp)0UL), ((-((_Tp)691UL)) / ((_Tp)2730UL)), ((_Tp)0UL), (((_Tp)7UL) / ((_Tp)6UL)), ((_Tp)0UL), ((-((_Tp)3617UL)) / ((_Tp)510UL)), ((_Tp)0UL), (((_Tp)43867UL) / ((_Tp)798UL)), ((_Tp)0UL), ((-((_Tp)174611)) / ((_Tp)330UL)), ((_Tp)0UL), (((_Tp)854513UL) / ((_Tp)138UL)), ((_Tp)0UL), ((-((_Tp)236364091UL)) / ((_Tp)2730UL)), ((_Tp)0UL), (((_Tp)8553103UL) / ((_Tp)6UL)), ((_Tp)0UL)}; 
# 98
if (__n == (0)) { 
# 99
return (_Tp)1; }  
# 101
if (__n == (1)) { 
# 102
return (-((_Tp)1)) / ((_Tp)2); }  
# 105
if ((__n % (2)) == (1)) { 
# 106
return (_Tp)0; }  
# 109
if (__n < (28)) { 
# 110
return __num[__n]; }  
# 113
_Tp __fact = ((_Tp)1); 
# 114
if (((__n / (2)) % (2)) == (0)) { 
# 115
__fact *= ((_Tp)(-1)); }  
# 116
for (unsigned __k = (1); __k <= __n; ++__k) { 
# 117
__fact *= (__k / (((_Tp)2) * __numeric_constants< _Tp> ::__pi())); }  
# 118
__fact *= ((_Tp)2); 
# 120
_Tp __sum = ((_Tp)0); 
# 121
for (unsigned __i = (1); __i < (1000); ++__i) 
# 122
{ 
# 123
_Tp __term = std::pow((_Tp)__i, -((_Tp)__n)); 
# 124
if (__term < std::template numeric_limits< _Tp> ::epsilon()) { 
# 125
break; }  
# 126
__sum += __term; 
# 127
}  
# 129
return __fact * __sum; 
# 130
} 
# 139 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> inline _Tp 
# 141
__bernoulli(int __n) 
# 142
{ return __bernoulli_series< _Tp> (__n); } 
# 153 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 155
__log_gamma_bernoulli(_Tp __x) 
# 156
{ 
# 157
_Tp __lg = (((__x - ((_Tp)(0.5L))) * std::log(__x)) - __x) + (((_Tp)(0.5L)) * std::log(((_Tp)2) * __numeric_constants< _Tp> ::__pi())); 
# 161
const _Tp __xx = __x * __x; 
# 162
_Tp __help = ((_Tp)1) / __x; 
# 163
for (unsigned __i = (1); __i < (20); ++__i) 
# 164
{ 
# 165
const _Tp __2i = (_Tp)((2) * __i); 
# 166
__help /= ((__2i * (__2i - ((_Tp)1))) * __xx); 
# 167
__lg += (__bernoulli< _Tp> ((2) * __i) * __help); 
# 168
}  
# 170
return __lg; 
# 171
} 
# 181 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 183
__log_gamma_lanczos(_Tp __x) 
# 184
{ 
# 185
const _Tp __xm1 = __x - ((_Tp)1); 
# 187
static const _Tp __lanczos_cheb_7[9] = {((_Tp)(0.99999999999980993226L)), ((_Tp)(676.52036812188509857L)), ((_Tp)(-(1259.1392167224028704L))), ((_Tp)(771.32342877765307887L)), ((_Tp)(-(176.61502916214059906L))), ((_Tp)(12.507343278686904814L)), ((_Tp)(-(0.1385710952657201169L))), ((_Tp)(9.9843695780195708595E-6L)), ((_Tp)(1.5056327351493115584E-7L))}; 
# 199
static const _Tp __LOGROOT2PI = ((_Tp)(0.9189385332046727418L)); 
# 202
_Tp __sum = (__lanczos_cheb_7[0]); 
# 203
for (unsigned __k = (1); __k < (9); ++__k) { 
# 204
__sum += ((__lanczos_cheb_7[__k]) / (__xm1 + __k)); }  
# 206
const _Tp __term1 = (__xm1 + ((_Tp)(0.5L))) * std::log((__xm1 + ((_Tp)(7.5L))) / __numeric_constants< _Tp> ::__euler()); 
# 209
const _Tp __term2 = __LOGROOT2PI + std::log(__sum); 
# 210
const _Tp __result = __term1 + (__term2 - ((_Tp)7)); 
# 212
return __result; 
# 213
} 
# 225 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 227
__log_gamma(_Tp __x) 
# 228
{ 
# 229
if (__x > ((_Tp)(0.5L))) { 
# 230
return __log_gamma_lanczos(__x); } else 
# 232
{ 
# 233
const _Tp __sin_fact = std::abs(std::sin(__numeric_constants< _Tp> ::__pi() * __x)); 
# 235
if (__sin_fact == ((_Tp)0)) { 
# 236
std::__throw_domain_error("Argument is nonpositive integer in __log_gamma"); }  
# 238
return (__numeric_constants< _Tp> ::__lnpi() - std::log(__sin_fact)) - __log_gamma_lanczos(((_Tp)1) - __x); 
# 241
}  
# 242
} 
# 252 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 254
__log_gamma_sign(_Tp __x) 
# 255
{ 
# 256
if (__x > ((_Tp)0)) { 
# 257
return (_Tp)1; } else 
# 259
{ 
# 260
const _Tp __sin_fact = std::sin(__numeric_constants< _Tp> ::__pi() * __x); 
# 262
if (__sin_fact > ((_Tp)0)) { 
# 263
return 1; } else { 
# 264
if (__sin_fact < ((_Tp)0)) { 
# 265
return -((_Tp)1); } else { 
# 267
return (_Tp)0; }  }  
# 268
}  
# 269
} 
# 283 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 285
__log_bincoef(unsigned __n, unsigned __k) 
# 286
{ 
# 288
static const _Tp __max_bincoeff = (std::template numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 292
_Tp __coeff = (std::lgamma((_Tp)((1) + __n)) - std::lgamma((_Tp)((1) + __k))) - std::lgamma((_Tp)(((1) + __n) - __k)); 
# 300
} 
# 314 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 316
__bincoef(unsigned __n, unsigned __k) 
# 317
{ 
# 319
static const _Tp __max_bincoeff = (std::template numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 323
const _Tp __log_coeff = __log_bincoef< _Tp> (__n, __k); 
# 324
if (__log_coeff > __max_bincoeff) { 
# 325
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 327
return std::exp(__log_coeff); }  
# 328
} 
# 337 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> inline _Tp 
# 339
__gamma(_Tp __x) 
# 340
{ return std::exp(__log_gamma(__x)); } 
# 356 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 358
__psi_series(_Tp __x) 
# 359
{ 
# 360
_Tp __sum = (-__numeric_constants< _Tp> ::__gamma_e()) - (((_Tp)1) / __x); 
# 361
const unsigned __max_iter = (100000); 
# 362
for (unsigned __k = (1); __k < __max_iter; ++__k) 
# 363
{ 
# 364
const _Tp __term = __x / (__k * (__k + __x)); 
# 365
__sum += __term; 
# 366
if (std::abs(__term / __sum) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 367
break; }  
# 368
}  
# 369
return __sum; 
# 370
} 
# 386 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 388
__psi_asymp(_Tp __x) 
# 389
{ 
# 390
_Tp __sum = std::log(__x) - (((_Tp)(0.5L)) / __x); 
# 391
const _Tp __xx = __x * __x; 
# 392
_Tp __xp = __xx; 
# 393
const unsigned __max_iter = (100); 
# 394
for (unsigned __k = (1); __k < __max_iter; ++__k) 
# 395
{ 
# 396
const _Tp __term = __bernoulli< _Tp> ((2) * __k) / (((2) * __k) * __xp); 
# 397
__sum -= __term; 
# 398
if (std::abs(__term / __sum) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 399
break; }  
# 400
__xp *= __xx; 
# 401
}  
# 402
return __sum; 
# 403
} 
# 417 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 419
__psi(_Tp __x) 
# 420
{ 
# 421
const int __n = static_cast< int>(__x + (0.5L)); 
# 422
const _Tp __eps = ((_Tp)4) * std::template numeric_limits< _Tp> ::epsilon(); 
# 423
if ((__n <= 0) && (std::abs(__x - ((_Tp)__n)) < __eps)) { 
# 424
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 425
if (__x < ((_Tp)0)) 
# 426
{ 
# 427
const _Tp __pi = __numeric_constants< _Tp> ::__pi(); 
# 428
return __psi(((_Tp)1) - __x) - ((__pi * std::cos(__pi * __x)) / std::sin(__pi * __x)); 
# 430
} else { 
# 431
if (__x > ((_Tp)100)) { 
# 432
return __psi_asymp(__x); } else { 
# 434
return __psi_series(__x); }  }  }  
# 435
} 
# 446 "/usr/include/c++/12/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 448
__psi(unsigned __n, _Tp __x) 
# 449
{ 
# 450
if (__x <= ((_Tp)0)) { 
# 451
std::__throw_domain_error("Argument out of range in __psi"); } else { 
# 453
if (__n == (0)) { 
# 454
return __psi(__x); } else 
# 456
{ 
# 457
const _Tp __hzeta = __hurwitz_zeta((_Tp)(__n + (1)), __x); 
# 459
const _Tp __ln_nfact = std::lgamma((_Tp)(__n + (1))); 
# 463
_Tp __result = std::exp(__ln_nfact) * __hzeta; 
# 464
if ((__n % (2)) == (1)) { 
# 465
__result = (-__result); }  
# 466
return __result; 
# 467
}  }  
# 468
} 
# 469
}
# 476
}
# 55 "/usr/include/c++/12/tr1/bessel_function.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 71 "/usr/include/c++/12/tr1/bessel_function.tcc" 3
namespace __detail { 
# 98 "/usr/include/c++/12/tr1/bessel_function.tcc" 3
template< class _Tp> void 
# 100
__gamma_temme(_Tp __mu, _Tp &
# 101
__gam1, _Tp &__gam2, _Tp &__gampl, _Tp &__gammi) 
# 102
{ 
# 104
__gampl = (((_Tp)1) / std::tgamma(((_Tp)1) + __mu)); 
# 105
__gammi = (((_Tp)1) / std::tgamma(((_Tp)1) - __mu)); 
# 111
if (std::abs(__mu) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 112
__gam1 = (-((_Tp)__numeric_constants< _Tp> ::__gamma_e())); } else { 
# 114
__gam1 = ((__gammi - __gampl) / (((_Tp)2) * __mu)); }  
# 116
__gam2 = ((__gammi + __gampl) / ((_Tp)2)); 
# 119
} 
# 136 "/usr/include/c++/12/tr1/bessel_function.tcc" 3
template< class _Tp> void 
# 138
__bessel_jn(_Tp __nu, _Tp __x, _Tp &
# 139
__Jnu, _Tp &__Nnu, _Tp &__Jpnu, _Tp &__Npnu) 
# 140
{ 
# 141
if (__x == ((_Tp)0)) 
# 142
{ 
# 143
if (__nu == ((_Tp)0)) 
# 144
{ 
# 145
__Jnu = ((_Tp)1); 
# 146
__Jpnu = ((_Tp)0); 
# 147
} else { 
# 148
if (__nu == ((_Tp)1)) 
# 149
{ 
# 150
__Jnu = ((_Tp)0); 
# 151
__Jpnu = ((_Tp)(0.5L)); 
# 152
} else 
# 154
{ 
# 155
__Jnu = ((_Tp)0); 
# 156
__Jpnu = ((_Tp)0); 
# 157
}  }  
# 158
__Nnu = (-std::template numeric_limits< _Tp> ::infinity()); 
# 159
__Npnu = std::template numeric_limits< _Tp> ::infinity(); 
# 160
return; 
# 161
}  
# 163
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 168
const _Tp __fp_min = std::sqrt(std::template numeric_limits< _Tp> ::min()); 
# 169
const int __max_iter = 15000; 
# 170
const _Tp __x_min = ((_Tp)2); 
# 172
const int __nl = (__x < __x_min) ? static_cast< int>(__nu + ((_Tp)(0.5L))) : std::max(0, static_cast< int>((__nu - __x) + ((_Tp)(1.5L)))); 
# 176
const _Tp __mu = __nu - __nl; 
# 177
const _Tp __mu2 = __mu * __mu; 
# 178
const _Tp __xi = ((_Tp)1) / __x; 
# 179
const _Tp __xi2 = ((_Tp)2) * __xi; 
# 180
_Tp __w = __xi2 / __numeric_constants< _Tp> ::__pi(); 
# 181
int __isign = 1; 
# 182
_Tp __h = __nu * __xi; 
# 183
if (__h < __fp_min) { 
# 184
__h = __fp_min; }  
# 185
_Tp __b = __xi2 * __nu; 
# 186
_Tp __d = ((_Tp)0); 
# 187
_Tp __c = __h; 
# 188
int __i; 
# 189
for (__i = 1; __i <= __max_iter; ++__i) 
# 190
{ 
# 191
__b += __xi2; 
# 192
__d = (__b - __d); 
# 193
if (std::abs(__d) < __fp_min) { 
# 194
__d = __fp_min; }  
# 195
__c = (__b - (((_Tp)1) / __c)); 
# 196
if (std::abs(__c) < __fp_min) { 
# 197
__c = __fp_min; }  
# 198
__d = (((_Tp)1) / __d); 
# 199
const _Tp __del = __c * __d; 
# 200
__h *= __del; 
# 201
if (__d < ((_Tp)0)) { 
# 202
__isign = (-__isign); }  
# 203
if (std::abs(__del - ((_Tp)1)) < __eps) { 
# 204
break; }  
# 205
}  
# 206
if (__i > __max_iter) { 
# 207
std::__throw_runtime_error("Argument x too large in __bessel_jn; try asymptotic expansion."); }  
# 209
_Tp __Jnul = __isign * __fp_min; 
# 210
_Tp __Jpnul = __h * __Jnul; 
# 211
_Tp __Jnul1 = __Jnul; 
# 212
_Tp __Jpnu1 = __Jpnul; 
# 213
_Tp __fact = __nu * __xi; 
# 214
for (int __l = __nl; __l >= 1; --__l) 
# 215
{ 
# 216
const _Tp __Jnutemp = (__fact * __Jnul) + __Jpnul; 
# 217
__fact -= __xi; 
# 218
__Jpnul = ((__fact * __Jnutemp) - __Jnul); 
# 219
__Jnul = __Jnutemp; 
# 220
}  
# 221
if (__Jnul == ((_Tp)0)) { 
# 222
__Jnul = __eps; }  
# 223
_Tp __f = __Jpnul / __Jnul; 
# 224
_Tp __Nmu, __Nnu1, __Npmu, __Jmu; 
# 225
if (__x < __x_min) 
# 226
{ 
# 227
const _Tp __x2 = __x / ((_Tp)2); 
# 228
const _Tp __pimu = __numeric_constants< _Tp> ::__pi() * __mu; 
# 229
_Tp __fact = (std::abs(__pimu) < __eps) ? (_Tp)1 : (__pimu / std::sin(__pimu)); 
# 231
_Tp __d = (-std::log(__x2)); 
# 232
_Tp __e = __mu * __d; 
# 233
_Tp __fact2 = (std::abs(__e) < __eps) ? (_Tp)1 : (std::sinh(__e) / __e); 
# 235
_Tp __gam1, __gam2, __gampl, __gammi; 
# 236
__gamma_temme(__mu, __gam1, __gam2, __gampl, __gammi); 
# 237
_Tp __ff = ((((_Tp)2) / __numeric_constants< _Tp> ::__pi()) * __fact) * ((__gam1 * std::cosh(__e)) + ((__gam2 * __fact2) * __d)); 
# 239
__e = std::exp(__e); 
# 240
_Tp __p = __e / (__numeric_constants< _Tp> ::__pi() * __gampl); 
# 241
_Tp __q = ((_Tp)1) / ((__e * __numeric_constants< _Tp> ::__pi()) * __gammi); 
# 242
const _Tp __pimu2 = __pimu / ((_Tp)2); 
# 243
_Tp __fact3 = (std::abs(__pimu2) < __eps) ? (_Tp)1 : (std::sin(__pimu2) / __pimu2); 
# 245
_Tp __r = ((__numeric_constants< _Tp> ::__pi() * __pimu2) * __fact3) * __fact3; 
# 246
_Tp __c = ((_Tp)1); 
# 247
__d = ((-__x2) * __x2); 
# 248
_Tp __sum = __ff + (__r * __q); 
# 249
_Tp __sum1 = __p; 
# 250
for (__i = 1; __i <= __max_iter; ++__i) 
# 251
{ 
# 252
__ff = ((((__i * __ff) + __p) + __q) / ((__i * __i) - __mu2)); 
# 253
__c *= (__d / ((_Tp)__i)); 
# 254
__p /= (((_Tp)__i) - __mu); 
# 255
__q /= (((_Tp)__i) + __mu); 
# 256
const _Tp __del = __c * (__ff + (__r * __q)); 
# 257
__sum += __del; 
# 258
const _Tp __del1 = (__c * __p) - (__i * __del); 
# 259
__sum1 += __del1; 
# 260
if (std::abs(__del) < (__eps * (((_Tp)1) + std::abs(__sum)))) { 
# 261
break; }  
# 262
}  
# 263
if (__i > __max_iter) { 
# 264
std::__throw_runtime_error("Bessel y series failed to converge in __bessel_jn."); }  
# 266
__Nmu = (-__sum); 
# 267
__Nnu1 = ((-__sum1) * __xi2); 
# 268
__Npmu = (((__mu * __xi) * __Nmu) - __Nnu1); 
# 269
__Jmu = (__w / (__Npmu - (__f * __Nmu))); 
# 270
} else 
# 272
{ 
# 273
_Tp __a = ((_Tp)(0.25L)) - __mu2; 
# 274
_Tp __q = ((_Tp)1); 
# 275
_Tp __p = ((-__xi) / ((_Tp)2)); 
# 276
_Tp __br = ((_Tp)2) * __x; 
# 277
_Tp __bi = ((_Tp)2); 
# 278
_Tp __fact = (__a * __xi) / ((__p * __p) + (__q * __q)); 
# 279
_Tp __cr = __br + (__q * __fact); 
# 280
_Tp __ci = __bi + (__p * __fact); 
# 281
_Tp __den = (__br * __br) + (__bi * __bi); 
# 282
_Tp __dr = __br / __den; 
# 283
_Tp __di = (-__bi) / __den; 
# 284
_Tp __dlr = (__cr * __dr) - (__ci * __di); 
# 285
_Tp __dli = (__cr * __di) + (__ci * __dr); 
# 286
_Tp __temp = (__p * __dlr) - (__q * __dli); 
# 287
__q = ((__p * __dli) + (__q * __dlr)); 
# 288
__p = __temp; 
# 289
int __i; 
# 290
for (__i = 2; __i <= __max_iter; ++__i) 
# 291
{ 
# 292
__a += ((_Tp)(2 * (__i - 1))); 
# 293
__bi += ((_Tp)2); 
# 294
__dr = ((__a * __dr) + __br); 
# 295
__di = ((__a * __di) + __bi); 
# 296
if ((std::abs(__dr) + std::abs(__di)) < __fp_min) { 
# 297
__dr = __fp_min; }  
# 298
__fact = (__a / ((__cr * __cr) + (__ci * __ci))); 
# 299
__cr = (__br + (__cr * __fact)); 
# 300
__ci = (__bi - (__ci * __fact)); 
# 301
if ((std::abs(__cr) + std::abs(__ci)) < __fp_min) { 
# 302
__cr = __fp_min; }  
# 303
__den = ((__dr * __dr) + (__di * __di)); 
# 304
__dr /= __den; 
# 305
__di /= (-__den); 
# 306
__dlr = ((__cr * __dr) - (__ci * __di)); 
# 307
__dli = ((__cr * __di) + (__ci * __dr)); 
# 308
__temp = ((__p * __dlr) - (__q * __dli)); 
# 309
__q = ((__p * __dli) + (__q * __dlr)); 
# 310
__p = __temp; 
# 311
if ((std::abs(__dlr - ((_Tp)1)) + std::abs(__dli)) < __eps) { 
# 312
break; }  
# 313
}  
# 314
if (__i > __max_iter) { 
# 315
std::__throw_runtime_error("Lentz\'s method failed in __bessel_jn."); }  
# 317
const _Tp __gam = (__p - __f) / __q; 
# 318
__Jmu = std::sqrt(__w / (((__p - __f) * __gam) + __q)); 
# 320
__Jmu = std::copysign(__Jmu, __Jnul); 
# 325
__Nmu = (__gam * __Jmu); 
# 326
__Npmu = ((__p + (__q / __gam)) * __Nmu); 
# 327
__Nnu1 = (((__mu * __xi) * __Nmu) - __Npmu); 
# 328
}  
# 329
__fact = (__Jmu / __Jnul); 
# 330
__Jnu = (__fact * __Jnul1); 
# 331
__Jpnu = (__fact * __Jpnu1); 
# 332
for (__i = 1; __i <= __nl; ++__i) 
# 333
{ 
# 334
const _Tp __Nnutemp = (((__mu + __i) * __xi2) * __Nnu1) - __Nmu; 
# 335
__Nmu = __Nnu1; 
# 336
__Nnu1 = __Nnutemp; 
# 337
}  
# 338
__Nnu = __Nmu; 
# 339
__Npnu = (((__nu * __xi) * __Nmu) - __Nnu1); 
# 342
} 
# 361 "/usr/include/c++/12/tr1/bessel_function.tcc" 3
template< class _Tp> void 
# 363
__cyl_bessel_jn_asymp(_Tp __nu, _Tp __x, _Tp &__Jnu, _Tp &__Nnu) 
# 364
{ 
# 365
const _Tp __mu = (((_Tp)4) * __nu) * __nu; 
# 366
const _Tp __8x = ((_Tp)8) * __x; 
# 368
_Tp __P = ((_Tp)0); 
# 369
_Tp __Q = ((_Tp)0); 
# 371
_Tp __k = ((_Tp)0); 
# 372
_Tp __term = ((_Tp)1); 
# 374
int __epsP = 0; 
# 375
int __epsQ = 0; 
# 377
_Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 379
do 
# 380
{ 
# 381
__term *= ((__k == 0) ? (_Tp)1 : ((-(__mu - (((2 * __k) - 1) * ((2 * __k) - 1)))) / (__k * __8x))); 
# 385
__epsP = (std::abs(__term) < (__eps * std::abs(__P))); 
# 386
__P += __term; 
# 388
__k++; 
# 390
__term *= ((__mu - (((2 * __k) - 1) * ((2 * __k) - 1))) / (__k * __8x)); 
# 391
__epsQ = (std::abs(__term) < (__eps * std::abs(__Q))); 
# 392
__Q += __term; 
# 394
if (__epsP && __epsQ && (__k > (__nu / (2.0)))) { 
# 395
break; }  
# 397
__k++; 
# 398
} 
# 399
while (__k < 1000); 
# 401
const _Tp __chi = __x - ((__nu + ((_Tp)(0.5L))) * __numeric_constants< _Tp> ::__pi_2()); 
# 404
const _Tp __c = std::cos(__chi); 
# 405
const _Tp __s = std::sin(__chi); 
# 407
const _Tp __coef = std::sqrt(((_Tp)2) / (__numeric_constants< _Tp> ::__pi() * __x)); 
# 410
__Jnu = (__coef * ((__c * __P) - (__s * __Q))); 
# 411
__Nnu = (__coef * ((__s * __P) + (__c * __Q))); 
# 414
} 
# 444 "/usr/include/c++/12/tr1/bessel_function.tcc" 3
template< class _Tp> _Tp 
# 446
__cyl_bessel_ij_series(_Tp __nu, _Tp __x, _Tp __sgn, unsigned 
# 447
__max_iter) 
# 448
{ 
# 449
if (__x == ((_Tp)0)) { 
# 450
return (__nu == ((_Tp)0)) ? (_Tp)1 : ((_Tp)0); }  
# 452
const _Tp __x2 = __x / ((_Tp)2); 
# 453
_Tp __fact = __nu * std::log(__x2); 
# 455
__fact -= std::lgamma(__nu + ((_Tp)1)); 
# 459
__fact = std::exp(__fact); 
# 460
const _Tp __xx4 = (__sgn * __x2) * __x2; 
# 461
_Tp __Jn = ((_Tp)1); 
# 462
_Tp __term = ((_Tp)1); 
# 464
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 465
{ 
# 466
__term *= (__xx4 / (((_Tp)__i) * (__nu + ((_Tp)__i)))); 
# 467
__Jn += __term; 
# 468
if (std::abs(__term / __Jn) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 469
break; }  
# 470
}  
# 472
return __fact * __Jn; 
# 473
} 
# 490 "/usr/include/c++/12/tr1/bessel_function.tcc" 3
template< class _Tp> _Tp 
# 492
__cyl_bessel_j(_Tp __nu, _Tp __x) 
# 493
{ 
# 494
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 495
std::__throw_domain_error("Bad argument in __cyl_bessel_j."); } else { 
# 497
if (__isnan(__nu) || __isnan(__x)) { 
# 498
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 499
if ((__x * __x) < (((_Tp)10) * (__nu + ((_Tp)1)))) { 
# 500
return __cyl_bessel_ij_series(__nu, __x, -((_Tp)1), 200); } else { 
# 501
if (__x > ((_Tp)1000)) 
# 502
{ 
# 503
_Tp __J_nu, __N_nu; 
# 504
__cyl_bessel_jn_asymp(__nu, __x, __J_nu, __N_nu); 
# 505
return __J_nu; 
# 506
} else 
# 508
{ 
# 509
_Tp __J_nu, __N_nu, __Jp_nu, __Np_nu; 
# 510
__bessel_jn(__nu, __x, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 511
return __J_nu; 
# 512
}  }  }  }  
# 513
} 
# 532 "/usr/include/c++/12/tr1/bessel_function.tcc" 3
template< class _Tp> _Tp 
# 534
__cyl_neumann_n(_Tp __nu, _Tp __x) 
# 535
{ 
# 536
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 537
std::__throw_domain_error("Bad argument in __cyl_neumann_n."); } else { 
# 539
if (__isnan(__nu) || __isnan(__x)) { 
# 540
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 541
if (__x > ((_Tp)1000)) 
# 542
{ 
# 543
_Tp __J_nu, __N_nu; 
# 544
__cyl_bessel_jn_asymp(__nu, __x, __J_nu, __N_nu); 
# 545
return __N_nu; 
# 546
} else 
# 548
{ 
# 549
_Tp __J_nu, __N_nu, __Jp_nu, __Np_nu; 
# 550
__bessel_jn(__nu, __x, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 551
return __N_nu; 
# 552
}  }  }  
# 553
} 
# 569 "/usr/include/c++/12/tr1/bessel_function.tcc" 3
template< class _Tp> void 
# 571
__sph_bessel_jn(unsigned __n, _Tp __x, _Tp &
# 572
__j_n, _Tp &__n_n, _Tp &__jp_n, _Tp &__np_n) 
# 573
{ 
# 574
const _Tp __nu = ((_Tp)__n) + ((_Tp)(0.5L)); 
# 576
_Tp __J_nu, __N_nu, __Jp_nu, __Np_nu; 
# 577
__bessel_jn(__nu, __x, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 579
const _Tp __factor = __numeric_constants< _Tp> ::__sqrtpio2() / std::sqrt(__x); 
# 582
__j_n = (__factor * __J_nu); 
# 583
__n_n = (__factor * __N_nu); 
# 584
__jp_n = ((__factor * __Jp_nu) - (__j_n / (((_Tp)2) * __x))); 
# 585
__np_n = ((__factor * __Np_nu) - (__n_n / (((_Tp)2) * __x))); 
# 588
} 
# 604 "/usr/include/c++/12/tr1/bessel_function.tcc" 3
template< class _Tp> _Tp 
# 606
__sph_bessel(unsigned __n, _Tp __x) 
# 607
{ 
# 608
if (__x < ((_Tp)0)) { 
# 609
std::__throw_domain_error("Bad argument in __sph_bessel."); } else { 
# 611
if (__isnan(__x)) { 
# 612
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 613
if (__x == ((_Tp)0)) 
# 614
{ 
# 615
if (__n == (0)) { 
# 616
return (_Tp)1; } else { 
# 618
return (_Tp)0; }  
# 619
} else 
# 621
{ 
# 622
_Tp __j_n, __n_n, __jp_n, __np_n; 
# 623
__sph_bessel_jn(__n, __x, __j_n, __n_n, __jp_n, __np_n); 
# 624
return __j_n; 
# 625
}  }  }  
# 626
} 
# 642 "/usr/include/c++/12/tr1/bessel_function.tcc" 3
template< class _Tp> _Tp 
# 644
__sph_neumann(unsigned __n, _Tp __x) 
# 645
{ 
# 646
if (__x < ((_Tp)0)) { 
# 647
std::__throw_domain_error("Bad argument in __sph_neumann."); } else { 
# 649
if (__isnan(__x)) { 
# 650
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 651
if (__x == ((_Tp)0)) { 
# 652
return -std::template numeric_limits< _Tp> ::infinity(); } else 
# 654
{ 
# 655
_Tp __j_n, __n_n, __jp_n, __np_n; 
# 656
__sph_bessel_jn(__n, __x, __j_n, __n_n, __jp_n, __np_n); 
# 657
return __n_n; 
# 658
}  }  }  
# 659
} 
# 660
}
# 667
}
# 49 "/usr/include/c++/12/tr1/beta_function.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/usr/include/c++/12/tr1/beta_function.tcc" 3
namespace __detail { 
# 79 "/usr/include/c++/12/tr1/beta_function.tcc" 3
template< class _Tp> _Tp 
# 81
__beta_gamma(_Tp __x, _Tp __y) 
# 82
{ 
# 84
_Tp __bet; 
# 86
if (__x > __y) 
# 87
{ 
# 88
__bet = (std::tgamma(__x) / std::tgamma(__x + __y)); 
# 90
__bet *= std::tgamma(__y); 
# 91
} else 
# 93
{ 
# 94
__bet = (std::tgamma(__y) / std::tgamma(__x + __y)); 
# 96
__bet *= std::tgamma(__x); 
# 97
}  
# 111 "/usr/include/c++/12/tr1/beta_function.tcc" 3
return __bet; 
# 112
} 
# 127 "/usr/include/c++/12/tr1/beta_function.tcc" 3
template< class _Tp> _Tp 
# 129
__beta_lgamma(_Tp __x, _Tp __y) 
# 130
{ 
# 132
_Tp __bet = (std::lgamma(__x) + std::lgamma(__y)) - std::lgamma(__x + __y); 
# 140
__bet = std::exp(__bet); 
# 141
return __bet; 
# 142
} 
# 158 "/usr/include/c++/12/tr1/beta_function.tcc" 3
template< class _Tp> _Tp 
# 160
__beta_product(_Tp __x, _Tp __y) 
# 161
{ 
# 163
_Tp __bet = (__x + __y) / (__x * __y); 
# 165
unsigned __max_iter = (1000000); 
# 166
for (unsigned __k = (1); __k < __max_iter; ++__k) 
# 167
{ 
# 168
_Tp __term = (((_Tp)1) + ((__x + __y) / __k)) / ((((_Tp)1) + (__x / __k)) * (((_Tp)1) + (__y / __k))); 
# 170
__bet *= __term; 
# 171
}  
# 173
return __bet; 
# 174
} 
# 189 "/usr/include/c++/12/tr1/beta_function.tcc" 3
template< class _Tp> inline _Tp 
# 191
__beta(_Tp __x, _Tp __y) 
# 192
{ 
# 193
if (__isnan(__x) || __isnan(__y)) { 
# 194
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 196
return __beta_lgamma(__x, __y); }  
# 197
} 
# 198
}
# 205
}
# 45 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 59 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
namespace __detail { 
# 76 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 78
__ellint_rf(_Tp __x, _Tp __y, _Tp __z) 
# 79
{ 
# 80
const _Tp __min = std::template numeric_limits< _Tp> ::min(); 
# 81
const _Tp __lolim = ((_Tp)5) * __min; 
# 83
if (((__x < ((_Tp)0)) || (__y < ((_Tp)0))) || (__z < ((_Tp)0))) { 
# 84
std::__throw_domain_error("Argument less than zero in __ellint_rf."); } else { 
# 86
if ((((__x + __y) < __lolim) || ((__x + __z) < __lolim)) || ((__y + __z) < __lolim)) { 
# 88
std::__throw_domain_error("Argument too small in __ellint_rf"); } else 
# 90
{ 
# 91
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 92
const _Tp __c1 = (((_Tp)1) / ((_Tp)24)); 
# 93
const _Tp __c2 = (((_Tp)1) / ((_Tp)10)); 
# 94
const _Tp __c3 = (((_Tp)3) / ((_Tp)44)); 
# 95
const _Tp __c4 = (((_Tp)1) / ((_Tp)14)); 
# 97
_Tp __xn = __x; 
# 98
_Tp __yn = __y; 
# 99
_Tp __zn = __z; 
# 101
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 102
const _Tp __errtol = std::pow(__eps, ((_Tp)1) / ((_Tp)6)); 
# 103
_Tp __mu; 
# 104
_Tp __xndev, __yndev, __zndev; 
# 106
const unsigned __max_iter = (100); 
# 107
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 108
{ 
# 109
__mu = (((__xn + __yn) + __zn) / ((_Tp)3)); 
# 110
__xndev = (2 - ((__mu + __xn) / __mu)); 
# 111
__yndev = (2 - ((__mu + __yn) / __mu)); 
# 112
__zndev = (2 - ((__mu + __zn) / __mu)); 
# 113
_Tp __epsilon = std::max(std::abs(__xndev), std::abs(__yndev)); 
# 114
__epsilon = std::max(__epsilon, std::abs(__zndev)); 
# 115
if (__epsilon < __errtol) { 
# 116
break; }  
# 117
const _Tp __xnroot = std::sqrt(__xn); 
# 118
const _Tp __ynroot = std::sqrt(__yn); 
# 119
const _Tp __znroot = std::sqrt(__zn); 
# 120
const _Tp __lambda = (__xnroot * (__ynroot + __znroot)) + (__ynroot * __znroot); 
# 122
__xn = (__c0 * (__xn + __lambda)); 
# 123
__yn = (__c0 * (__yn + __lambda)); 
# 124
__zn = (__c0 * (__zn + __lambda)); 
# 125
}  
# 127
const _Tp __e2 = (__xndev * __yndev) - (__zndev * __zndev); 
# 128
const _Tp __e3 = (__xndev * __yndev) * __zndev; 
# 129
const _Tp __s = (((_Tp)1) + ((((__c1 * __e2) - __c2) - (__c3 * __e3)) * __e2)) + (__c4 * __e3); 
# 132
return __s / std::sqrt(__mu); 
# 133
}  }  
# 134
} 
# 153 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 155
__comp_ellint_1_series(_Tp __k) 
# 156
{ 
# 158
const _Tp __kk = __k * __k; 
# 160
_Tp __term = __kk / ((_Tp)4); 
# 161
_Tp __sum = ((_Tp)1) + __term; 
# 163
const unsigned __max_iter = (1000); 
# 164
for (unsigned __i = (2); __i < __max_iter; ++__i) 
# 165
{ 
# 166
__term *= (((((2) * __i) - (1)) * __kk) / ((2) * __i)); 
# 167
if (__term < std::template numeric_limits< _Tp> ::epsilon()) { 
# 168
break; }  
# 169
__sum += __term; 
# 170
}  
# 172
return __numeric_constants< _Tp> ::__pi_2() * __sum; 
# 173
} 
# 191 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 193
__comp_ellint_1(_Tp __k) 
# 194
{ 
# 196
if (__isnan(__k)) { 
# 197
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 198
if (std::abs(__k) >= ((_Tp)1)) { 
# 199
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 201
return __ellint_rf((_Tp)0, ((_Tp)1) - (__k * __k), (_Tp)1); }  }  
# 202
} 
# 219 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 221
__ellint_1(_Tp __k, _Tp __phi) 
# 222
{ 
# 224
if (__isnan(__k) || __isnan(__phi)) { 
# 225
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 226
if (std::abs(__k) > ((_Tp)1)) { 
# 227
std::__throw_domain_error("Bad argument in __ellint_1."); } else 
# 229
{ 
# 231
const int __n = std::floor((__phi / __numeric_constants< _Tp> ::__pi()) + ((_Tp)(0.5L))); 
# 233
const _Tp __phi_red = __phi - (__n * __numeric_constants< _Tp> ::__pi()); 
# 236
const _Tp __s = std::sin(__phi_red); 
# 237
const _Tp __c = std::cos(__phi_red); 
# 239
const _Tp __F = __s * __ellint_rf(__c * __c, ((_Tp)1) - (((__k * __k) * __s) * __s), (_Tp)1); 
# 243
if (__n == 0) { 
# 244
return __F; } else { 
# 246
return __F + ((((_Tp)2) * __n) * __comp_ellint_1(__k)); }  
# 247
}  }  
# 248
} 
# 266 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 268
__comp_ellint_2_series(_Tp __k) 
# 269
{ 
# 271
const _Tp __kk = __k * __k; 
# 273
_Tp __term = __kk; 
# 274
_Tp __sum = __term; 
# 276
const unsigned __max_iter = (1000); 
# 277
for (unsigned __i = (2); __i < __max_iter; ++__i) 
# 278
{ 
# 279
const _Tp __i2m = ((2) * __i) - (1); 
# 280
const _Tp __i2 = (2) * __i; 
# 281
__term *= (((__i2m * __i2m) * __kk) / (__i2 * __i2)); 
# 282
if (__term < std::template numeric_limits< _Tp> ::epsilon()) { 
# 283
break; }  
# 284
__sum += (__term / __i2m); 
# 285
}  
# 287
return __numeric_constants< _Tp> ::__pi_2() * (((_Tp)1) - __sum); 
# 288
} 
# 314 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 316
__ellint_rd(_Tp __x, _Tp __y, _Tp __z) 
# 317
{ 
# 318
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 319
const _Tp __errtol = std::pow(__eps / ((_Tp)8), ((_Tp)1) / ((_Tp)6)); 
# 320
const _Tp __max = std::template numeric_limits< _Tp> ::max(); 
# 321
const _Tp __lolim = (((_Tp)2) / std::pow(__max, ((_Tp)2) / ((_Tp)3))); 
# 323
if ((__x < ((_Tp)0)) || (__y < ((_Tp)0))) { 
# 324
std::__throw_domain_error("Argument less than zero in __ellint_rd."); } else { 
# 326
if (((__x + __y) < __lolim) || (__z < __lolim)) { 
# 327
std::__throw_domain_error("Argument too small in __ellint_rd."); } else 
# 330
{ 
# 331
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 332
const _Tp __c1 = (((_Tp)3) / ((_Tp)14)); 
# 333
const _Tp __c2 = (((_Tp)1) / ((_Tp)6)); 
# 334
const _Tp __c3 = (((_Tp)9) / ((_Tp)22)); 
# 335
const _Tp __c4 = (((_Tp)3) / ((_Tp)26)); 
# 337
_Tp __xn = __x; 
# 338
_Tp __yn = __y; 
# 339
_Tp __zn = __z; 
# 340
_Tp __sigma = ((_Tp)0); 
# 341
_Tp __power4 = ((_Tp)1); 
# 343
_Tp __mu; 
# 344
_Tp __xndev, __yndev, __zndev; 
# 346
const unsigned __max_iter = (100); 
# 347
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 348
{ 
# 349
__mu = (((__xn + __yn) + (((_Tp)3) * __zn)) / ((_Tp)5)); 
# 350
__xndev = ((__mu - __xn) / __mu); 
# 351
__yndev = ((__mu - __yn) / __mu); 
# 352
__zndev = ((__mu - __zn) / __mu); 
# 353
_Tp __epsilon = std::max(std::abs(__xndev), std::abs(__yndev)); 
# 354
__epsilon = std::max(__epsilon, std::abs(__zndev)); 
# 355
if (__epsilon < __errtol) { 
# 356
break; }  
# 357
_Tp __xnroot = std::sqrt(__xn); 
# 358
_Tp __ynroot = std::sqrt(__yn); 
# 359
_Tp __znroot = std::sqrt(__zn); 
# 360
_Tp __lambda = (__xnroot * (__ynroot + __znroot)) + (__ynroot * __znroot); 
# 362
__sigma += (__power4 / (__znroot * (__zn + __lambda))); 
# 363
__power4 *= __c0; 
# 364
__xn = (__c0 * (__xn + __lambda)); 
# 365
__yn = (__c0 * (__yn + __lambda)); 
# 366
__zn = (__c0 * (__zn + __lambda)); 
# 367
}  
# 369
_Tp __ea = __xndev * __yndev; 
# 370
_Tp __eb = __zndev * __zndev; 
# 371
_Tp __ec = __ea - __eb; 
# 372
_Tp __ed = __ea - (((_Tp)6) * __eb); 
# 373
_Tp __ef = (__ed + __ec) + __ec; 
# 374
_Tp __s1 = __ed * (((-__c1) + ((__c3 * __ed) / ((_Tp)3))) - ((((((_Tp)3) * __c4) * __zndev) * __ef) / ((_Tp)2))); 
# 377
_Tp __s2 = __zndev * ((__c2 * __ef) + (__zndev * ((((-__c3) * __ec) - (__zndev * __c4)) - __ea))); 
# 381
return (((_Tp)3) * __sigma) + ((__power4 * ((((_Tp)1) + __s1) + __s2)) / (__mu * std::sqrt(__mu))); 
# 383
}  }  
# 384
} 
# 399 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 401
__comp_ellint_2(_Tp __k) 
# 402
{ 
# 404
if (__isnan(__k)) { 
# 405
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 406
if (std::abs(__k) == 1) { 
# 407
return (_Tp)1; } else { 
# 408
if (std::abs(__k) > ((_Tp)1)) { 
# 409
std::__throw_domain_error("Bad argument in __comp_ellint_2."); } else 
# 411
{ 
# 412
const _Tp __kk = __k * __k; 
# 414
return __ellint_rf((_Tp)0, ((_Tp)1) - __kk, (_Tp)1) - ((__kk * __ellint_rd((_Tp)0, ((_Tp)1) - __kk, (_Tp)1)) / ((_Tp)3)); 
# 416
}  }  }  
# 417
} 
# 433 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 435
__ellint_2(_Tp __k, _Tp __phi) 
# 436
{ 
# 438
if (__isnan(__k) || __isnan(__phi)) { 
# 439
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 440
if (std::abs(__k) > ((_Tp)1)) { 
# 441
std::__throw_domain_error("Bad argument in __ellint_2."); } else 
# 443
{ 
# 445
const int __n = std::floor((__phi / __numeric_constants< _Tp> ::__pi()) + ((_Tp)(0.5L))); 
# 447
const _Tp __phi_red = __phi - (__n * __numeric_constants< _Tp> ::__pi()); 
# 450
const _Tp __kk = __k * __k; 
# 451
const _Tp __s = std::sin(__phi_red); 
# 452
const _Tp __ss = __s * __s; 
# 453
const _Tp __sss = __ss * __s; 
# 454
const _Tp __c = std::cos(__phi_red); 
# 455
const _Tp __cc = __c * __c; 
# 457
const _Tp __E = (__s * __ellint_rf(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1)) - (((__kk * __sss) * __ellint_rd(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1)) / ((_Tp)3)); 
# 463
if (__n == 0) { 
# 464
return __E; } else { 
# 466
return __E + ((((_Tp)2) * __n) * __comp_ellint_2(__k)); }  
# 467
}  }  
# 468
} 
# 492 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 494
__ellint_rc(_Tp __x, _Tp __y) 
# 495
{ 
# 496
const _Tp __min = std::template numeric_limits< _Tp> ::min(); 
# 497
const _Tp __lolim = ((_Tp)5) * __min; 
# 499
if (((__x < ((_Tp)0)) || (__y < ((_Tp)0))) || ((__x + __y) < __lolim)) { 
# 500
std::__throw_domain_error("Argument less than zero in __ellint_rc."); } else 
# 503
{ 
# 504
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 505
const _Tp __c1 = (((_Tp)1) / ((_Tp)7)); 
# 506
const _Tp __c2 = (((_Tp)9) / ((_Tp)22)); 
# 507
const _Tp __c3 = (((_Tp)3) / ((_Tp)10)); 
# 508
const _Tp __c4 = (((_Tp)3) / ((_Tp)8)); 
# 510
_Tp __xn = __x; 
# 511
_Tp __yn = __y; 
# 513
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 514
const _Tp __errtol = std::pow(__eps / ((_Tp)30), ((_Tp)1) / ((_Tp)6)); 
# 515
_Tp __mu; 
# 516
_Tp __sn; 
# 518
const unsigned __max_iter = (100); 
# 519
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 520
{ 
# 521
__mu = ((__xn + (((_Tp)2) * __yn)) / ((_Tp)3)); 
# 522
__sn = (((__yn + __mu) / __mu) - ((_Tp)2)); 
# 523
if (std::abs(__sn) < __errtol) { 
# 524
break; }  
# 525
const _Tp __lambda = ((((_Tp)2) * std::sqrt(__xn)) * std::sqrt(__yn)) + __yn; 
# 527
__xn = (__c0 * (__xn + __lambda)); 
# 528
__yn = (__c0 * (__yn + __lambda)); 
# 529
}  
# 531
_Tp __s = (__sn * __sn) * (__c3 + (__sn * (__c1 + (__sn * (__c4 + (__sn * __c2)))))); 
# 534
return (((_Tp)1) + __s) / std::sqrt(__mu); 
# 535
}  
# 536
} 
# 561 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 563
__ellint_rj(_Tp __x, _Tp __y, _Tp __z, _Tp __p) 
# 564
{ 
# 565
const _Tp __min = std::template numeric_limits< _Tp> ::min(); 
# 566
const _Tp __lolim = std::pow(((_Tp)5) * __min, ((_Tp)1) / ((_Tp)3)); 
# 568
if (((__x < ((_Tp)0)) || (__y < ((_Tp)0))) || (__z < ((_Tp)0))) { 
# 569
std::__throw_domain_error("Argument less than zero in __ellint_rj."); } else { 
# 571
if (((((__x + __y) < __lolim) || ((__x + __z) < __lolim)) || ((__y + __z) < __lolim)) || (__p < __lolim)) { 
# 573
std::__throw_domain_error("Argument too small in __ellint_rj"); } else 
# 576
{ 
# 577
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 578
const _Tp __c1 = (((_Tp)3) / ((_Tp)14)); 
# 579
const _Tp __c2 = (((_Tp)1) / ((_Tp)3)); 
# 580
const _Tp __c3 = (((_Tp)3) / ((_Tp)22)); 
# 581
const _Tp __c4 = (((_Tp)3) / ((_Tp)26)); 
# 583
_Tp __xn = __x; 
# 584
_Tp __yn = __y; 
# 585
_Tp __zn = __z; 
# 586
_Tp __pn = __p; 
# 587
_Tp __sigma = ((_Tp)0); 
# 588
_Tp __power4 = ((_Tp)1); 
# 590
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 591
const _Tp __errtol = std::pow(__eps / ((_Tp)8), ((_Tp)1) / ((_Tp)6)); 
# 593
_Tp __mu; 
# 594
_Tp __xndev, __yndev, __zndev, __pndev; 
# 596
const unsigned __max_iter = (100); 
# 597
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 598
{ 
# 599
__mu = ((((__xn + __yn) + __zn) + (((_Tp)2) * __pn)) / ((_Tp)5)); 
# 600
__xndev = ((__mu - __xn) / __mu); 
# 601
__yndev = ((__mu - __yn) / __mu); 
# 602
__zndev = ((__mu - __zn) / __mu); 
# 603
__pndev = ((__mu - __pn) / __mu); 
# 604
_Tp __epsilon = std::max(std::abs(__xndev), std::abs(__yndev)); 
# 605
__epsilon = std::max(__epsilon, std::abs(__zndev)); 
# 606
__epsilon = std::max(__epsilon, std::abs(__pndev)); 
# 607
if (__epsilon < __errtol) { 
# 608
break; }  
# 609
const _Tp __xnroot = std::sqrt(__xn); 
# 610
const _Tp __ynroot = std::sqrt(__yn); 
# 611
const _Tp __znroot = std::sqrt(__zn); 
# 612
const _Tp __lambda = (__xnroot * (__ynroot + __znroot)) + (__ynroot * __znroot); 
# 614
const _Tp __alpha1 = (__pn * ((__xnroot + __ynroot) + __znroot)) + ((__xnroot * __ynroot) * __znroot); 
# 616
const _Tp __alpha2 = __alpha1 * __alpha1; 
# 617
const _Tp __beta = (__pn * (__pn + __lambda)) * (__pn + __lambda); 
# 619
__sigma += (__power4 * __ellint_rc(__alpha2, __beta)); 
# 620
__power4 *= __c0; 
# 621
__xn = (__c0 * (__xn + __lambda)); 
# 622
__yn = (__c0 * (__yn + __lambda)); 
# 623
__zn = (__c0 * (__zn + __lambda)); 
# 624
__pn = (__c0 * (__pn + __lambda)); 
# 625
}  
# 627
_Tp __ea = (__xndev * (__yndev + __zndev)) + (__yndev * __zndev); 
# 628
_Tp __eb = (__xndev * __yndev) * __zndev; 
# 629
_Tp __ec = __pndev * __pndev; 
# 630
_Tp __e2 = __ea - (((_Tp)3) * __ec); 
# 631
_Tp __e3 = __eb + ((((_Tp)2) * __pndev) * (__ea - __ec)); 
# 632
_Tp __s1 = ((_Tp)1) + (__e2 * (((-__c1) + (((((_Tp)3) * __c3) * __e2) / ((_Tp)4))) - (((((_Tp)3) * __c4) * __e3) / ((_Tp)2)))); 
# 634
_Tp __s2 = __eb * ((__c2 / ((_Tp)2)) + (__pndev * (((-__c3) - __c3) + (__pndev * __c4)))); 
# 636
_Tp __s3 = ((__pndev * __ea) * (__c2 - (__pndev * __c3))) - ((__c2 * __pndev) * __ec); 
# 639
return (((_Tp)3) * __sigma) + ((__power4 * ((__s1 + __s2) + __s3)) / (__mu * std::sqrt(__mu))); 
# 641
}  }  
# 642
} 
# 661 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 663
__comp_ellint_3(_Tp __k, _Tp __nu) 
# 664
{ 
# 666
if (__isnan(__k) || __isnan(__nu)) { 
# 667
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 668
if (__nu == ((_Tp)1)) { 
# 669
return std::template numeric_limits< _Tp> ::infinity(); } else { 
# 670
if (std::abs(__k) > ((_Tp)1)) { 
# 671
std::__throw_domain_error("Bad argument in __comp_ellint_3."); } else 
# 673
{ 
# 674
const _Tp __kk = __k * __k; 
# 676
return __ellint_rf((_Tp)0, ((_Tp)1) - __kk, (_Tp)1) + ((__nu * __ellint_rj((_Tp)0, ((_Tp)1) - __kk, (_Tp)1, ((_Tp)1) - __nu)) / ((_Tp)3)); 
# 680
}  }  }  
# 681
} 
# 701 "/usr/include/c++/12/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 703
__ellint_3(_Tp __k, _Tp __nu, _Tp __phi) 
# 704
{ 
# 706
if ((__isnan(__k) || __isnan(__nu)) || __isnan(__phi)) { 
# 707
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 708
if (std::abs(__k) > ((_Tp)1)) { 
# 709
std::__throw_domain_error("Bad argument in __ellint_3."); } else 
# 711
{ 
# 713
const int __n = std::floor((__phi / __numeric_constants< _Tp> ::__pi()) + ((_Tp)(0.5L))); 
# 715
const _Tp __phi_red = __phi - (__n * __numeric_constants< _Tp> ::__pi()); 
# 718
const _Tp __kk = __k * __k; 
# 719
const _Tp __s = std::sin(__phi_red); 
# 720
const _Tp __ss = __s * __s; 
# 721
const _Tp __sss = __ss * __s; 
# 722
const _Tp __c = std::cos(__phi_red); 
# 723
const _Tp __cc = __c * __c; 
# 725
const _Tp __Pi = (__s * __ellint_rf(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1)) + (((__nu * __sss) * __ellint_rj(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1, ((_Tp)1) - (__nu * __ss))) / ((_Tp)3)); 
# 731
if (__n == 0) { 
# 732
return __Pi; } else { 
# 734
return __Pi + ((((_Tp)2) * __n) * __comp_ellint_3(__k, __nu)); }  
# 735
}  }  
# 736
} 
# 737
}
# 743
}
# 50 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 64 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
namespace __detail { 
# 66
template< class _Tp> _Tp __expint_E1(_Tp); 
# 81 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 83
__expint_E1_series(_Tp __x) 
# 84
{ 
# 85
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 86
_Tp __term = ((_Tp)1); 
# 87
_Tp __esum = ((_Tp)0); 
# 88
_Tp __osum = ((_Tp)0); 
# 89
const unsigned __max_iter = (1000); 
# 90
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 91
{ 
# 92
__term *= ((-__x) / __i); 
# 93
if (std::abs(__term) < __eps) { 
# 94
break; }  
# 95
if (__term >= ((_Tp)0)) { 
# 96
__esum += (__term / __i); } else { 
# 98
__osum += (__term / __i); }  
# 99
}  
# 101
return (((-__esum) - __osum) - __numeric_constants< _Tp> ::__gamma_e()) - std::log(__x); 
# 103
} 
# 118 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 120
__expint_E1_asymp(_Tp __x) 
# 121
{ 
# 122
_Tp __term = ((_Tp)1); 
# 123
_Tp __esum = ((_Tp)1); 
# 124
_Tp __osum = ((_Tp)0); 
# 125
const unsigned __max_iter = (1000); 
# 126
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 127
{ 
# 128
_Tp __prev = __term; 
# 129
__term *= ((-__i) / __x); 
# 130
if (std::abs(__term) > std::abs(__prev)) { 
# 131
break; }  
# 132
if (__term >= ((_Tp)0)) { 
# 133
__esum += __term; } else { 
# 135
__osum += __term; }  
# 136
}  
# 138
return (std::exp(-__x) * (__esum + __osum)) / __x; 
# 139
} 
# 155 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 157
__expint_En_series(unsigned __n, _Tp __x) 
# 158
{ 
# 159
const unsigned __max_iter = (1000); 
# 160
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 161
const int __nm1 = __n - (1); 
# 162
_Tp __ans = (__nm1 != 0) ? ((_Tp)1) / __nm1 : ((-std::log(__x)) - __numeric_constants< _Tp> ::__gamma_e()); 
# 165
_Tp __fact = ((_Tp)1); 
# 166
for (int __i = 1; __i <= __max_iter; ++__i) 
# 167
{ 
# 168
__fact *= ((-__x) / ((_Tp)__i)); 
# 169
_Tp __del; 
# 170
if (__i != __nm1) { 
# 171
__del = ((-__fact) / ((_Tp)(__i - __nm1))); } else 
# 173
{ 
# 174
_Tp __psi = (-__numeric_constants< _Tp> ::gamma_e()); 
# 175
for (int __ii = 1; __ii <= __nm1; ++__ii) { 
# 176
__psi += (((_Tp)1) / ((_Tp)__ii)); }  
# 177
__del = (__fact * (__psi - std::log(__x))); 
# 178
}  
# 179
__ans += __del; 
# 180
if (std::abs(__del) < (__eps * std::abs(__ans))) { 
# 181
return __ans; }  
# 182
}  
# 183
std::__throw_runtime_error("Series summation failed in __expint_En_series."); 
# 185
} 
# 201 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 203
__expint_En_cont_frac(unsigned __n, _Tp __x) 
# 204
{ 
# 205
const unsigned __max_iter = (1000); 
# 206
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 207
const _Tp __fp_min = std::template numeric_limits< _Tp> ::min(); 
# 208
const int __nm1 = __n - (1); 
# 209
_Tp __b = __x + ((_Tp)__n); 
# 210
_Tp __c = ((_Tp)1) / __fp_min; 
# 211
_Tp __d = ((_Tp)1) / __b; 
# 212
_Tp __h = __d; 
# 213
for (unsigned __i = (1); __i <= __max_iter; ++__i) 
# 214
{ 
# 215
_Tp __a = (-((_Tp)(__i * (__nm1 + __i)))); 
# 216
__b += ((_Tp)2); 
# 217
__d = (((_Tp)1) / ((__a * __d) + __b)); 
# 218
__c = (__b + (__a / __c)); 
# 219
const _Tp __del = __c * __d; 
# 220
__h *= __del; 
# 221
if (std::abs(__del - ((_Tp)1)) < __eps) 
# 222
{ 
# 223
const _Tp __ans = __h * std::exp(-__x); 
# 224
return __ans; 
# 225
}  
# 226
}  
# 227
std::__throw_runtime_error("Continued fraction failed in __expint_En_cont_frac."); 
# 229
} 
# 246 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 248
__expint_En_recursion(unsigned __n, _Tp __x) 
# 249
{ 
# 250
_Tp __En; 
# 251
_Tp __E1 = __expint_E1(__x); 
# 252
if (__x < ((_Tp)__n)) 
# 253
{ 
# 255
__En = __E1; 
# 256
for (unsigned __j = (2); __j < __n; ++__j) { 
# 257
__En = ((std::exp(-__x) - (__x * __En)) / ((_Tp)(__j - (1)))); }  
# 258
} else 
# 260
{ 
# 262
__En = ((_Tp)1); 
# 263
const int __N = __n + (20); 
# 264
_Tp __save = ((_Tp)0); 
# 265
for (int __j = __N; __j > 0; --__j) 
# 266
{ 
# 267
__En = ((std::exp(-__x) - (__j * __En)) / __x); 
# 268
if (__j == __n) { 
# 269
__save = __En; }  
# 270
}  
# 271
_Tp __norm = __En / __E1; 
# 272
__En /= __norm; 
# 273
}  
# 275
return __En; 
# 276
} 
# 290 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 292
__expint_Ei_series(_Tp __x) 
# 293
{ 
# 294
_Tp __term = ((_Tp)1); 
# 295
_Tp __sum = ((_Tp)0); 
# 296
const unsigned __max_iter = (1000); 
# 297
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 298
{ 
# 299
__term *= (__x / __i); 
# 300
__sum += (__term / __i); 
# 301
if (__term < (std::template numeric_limits< _Tp> ::epsilon() * __sum)) { 
# 302
break; }  
# 303
}  
# 305
return (__numeric_constants< _Tp> ::__gamma_e() + __sum) + std::log(__x); 
# 306
} 
# 321 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 323
__expint_Ei_asymp(_Tp __x) 
# 324
{ 
# 325
_Tp __term = ((_Tp)1); 
# 326
_Tp __sum = ((_Tp)1); 
# 327
const unsigned __max_iter = (1000); 
# 328
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 329
{ 
# 330
_Tp __prev = __term; 
# 331
__term *= (__i / __x); 
# 332
if (__term < std::template numeric_limits< _Tp> ::epsilon()) { 
# 333
break; }  
# 334
if (__term >= __prev) { 
# 335
break; }  
# 336
__sum += __term; 
# 337
}  
# 339
return (std::exp(__x) * __sum) / __x; 
# 340
} 
# 354 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 356
__expint_Ei(_Tp __x) 
# 357
{ 
# 358
if (__x < ((_Tp)0)) { 
# 359
return -__expint_E1(-__x); } else { 
# 360
if (__x < (-std::log(std::template numeric_limits< _Tp> ::epsilon()))) { 
# 361
return __expint_Ei_series(__x); } else { 
# 363
return __expint_Ei_asymp(__x); }  }  
# 364
} 
# 378 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 380
__expint_E1(_Tp __x) 
# 381
{ 
# 382
if (__x < ((_Tp)0)) { 
# 383
return -__expint_Ei(-__x); } else { 
# 384
if (__x < ((_Tp)1)) { 
# 385
return __expint_E1_series(__x); } else { 
# 386
if (__x < ((_Tp)100)) { 
# 387
return __expint_En_cont_frac(1, __x); } else { 
# 389
return __expint_E1_asymp(__x); }  }  }  
# 390
} 
# 408 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 410
__expint_asymp(unsigned __n, _Tp __x) 
# 411
{ 
# 412
_Tp __term = ((_Tp)1); 
# 413
_Tp __sum = ((_Tp)1); 
# 414
for (unsigned __i = (1); __i <= __n; ++__i) 
# 415
{ 
# 416
_Tp __prev = __term; 
# 417
__term *= ((-((__n - __i) + (1))) / __x); 
# 418
if (std::abs(__term) > std::abs(__prev)) { 
# 419
break; }  
# 420
__sum += __term; 
# 421
}  
# 423
return (std::exp(-__x) * __sum) / __x; 
# 424
} 
# 442 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 444
__expint_large_n(unsigned __n, _Tp __x) 
# 445
{ 
# 446
const _Tp __xpn = __x + __n; 
# 447
const _Tp __xpn2 = __xpn * __xpn; 
# 448
_Tp __term = ((_Tp)1); 
# 449
_Tp __sum = ((_Tp)1); 
# 450
for (unsigned __i = (1); __i <= __n; ++__i) 
# 451
{ 
# 452
_Tp __prev = __term; 
# 453
__term *= ((__n - (((2) * (__i - (1))) * __x)) / __xpn2); 
# 454
if (std::abs(__term) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 455
break; }  
# 456
__sum += __term; 
# 457
}  
# 459
return (std::exp(-__x) * __sum) / __xpn; 
# 460
} 
# 476 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 478
__expint(unsigned __n, _Tp __x) 
# 479
{ 
# 481
if (__isnan(__x)) { 
# 482
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 483
if ((__n <= (1)) && (__x == ((_Tp)0))) { 
# 484
return std::template numeric_limits< _Tp> ::infinity(); } else 
# 486
{ 
# 487
_Tp __E0 = std::exp(__x) / __x; 
# 488
if (__n == (0)) { 
# 489
return __E0; }  
# 491
_Tp __E1 = __expint_E1(__x); 
# 492
if (__n == (1)) { 
# 493
return __E1; }  
# 495
if (__x == ((_Tp)0)) { 
# 496
return ((_Tp)1) / (static_cast< _Tp>(__n - (1))); }  
# 498
_Tp __En = __expint_En_recursion(__n, __x); 
# 500
return __En; 
# 501
}  }  
# 502
} 
# 516 "/usr/include/c++/12/tr1/exp_integral.tcc" 3
template< class _Tp> inline _Tp 
# 518
__expint(_Tp __x) 
# 519
{ 
# 520
if (__isnan(__x)) { 
# 521
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 523
return __expint_Ei(__x); }  
# 524
} 
# 525
}
# 531
}
# 44 "/usr/include/c++/12/tr1/hypergeometric.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 60 "/usr/include/c++/12/tr1/hypergeometric.tcc" 3
namespace __detail { 
# 83 "/usr/include/c++/12/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 85
__conf_hyperg_series(_Tp __a, _Tp __c, _Tp __x) 
# 86
{ 
# 87
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 89
_Tp __term = ((_Tp)1); 
# 90
_Tp __Fac = ((_Tp)1); 
# 91
const unsigned __max_iter = (100000); 
# 92
unsigned __i; 
# 93
for (__i = (0); __i < __max_iter; ++__i) 
# 94
{ 
# 95
__term *= (((__a + ((_Tp)__i)) * __x) / ((__c + ((_Tp)__i)) * ((_Tp)((1) + __i)))); 
# 97
if (std::abs(__term) < __eps) 
# 98
{ 
# 99
break; 
# 100
}  
# 101
__Fac += __term; 
# 102
}  
# 103
if (__i == __max_iter) { 
# 104
std::__throw_runtime_error("Series failed to converge in __conf_hyperg_series."); }  
# 107
return __Fac; 
# 108
} 
# 120 "/usr/include/c++/12/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 122
__conf_hyperg_luke(_Tp __a, _Tp __c, _Tp __xin) 
# 123
{ 
# 124
const _Tp __big = std::pow(std::template numeric_limits< _Tp> ::max(), (_Tp)(0.16L)); 
# 125
const int __nmax = 20000; 
# 126
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 127
const _Tp __x = (-__xin); 
# 128
const _Tp __x3 = (__x * __x) * __x; 
# 129
const _Tp __t0 = __a / __c; 
# 130
const _Tp __t1 = (__a + ((_Tp)1)) / (((_Tp)2) * __c); 
# 131
const _Tp __t2 = (__a + ((_Tp)2)) / (((_Tp)2) * (__c + ((_Tp)1))); 
# 132
_Tp __F = ((_Tp)1); 
# 133
_Tp __prec; 
# 135
_Tp __Bnm3 = ((_Tp)1); 
# 136
_Tp __Bnm2 = ((_Tp)1) + (__t1 * __x); 
# 137
_Tp __Bnm1 = ((_Tp)1) + ((__t2 * __x) * (((_Tp)1) + ((__t1 / ((_Tp)3)) * __x))); 
# 139
_Tp __Anm3 = ((_Tp)1); 
# 140
_Tp __Anm2 = __Bnm2 - (__t0 * __x); 
# 141
_Tp __Anm1 = (__Bnm1 - ((__t0 * (((_Tp)1) + (__t2 * __x))) * __x)) + ((((__t0 * __t1) * (__c / (__c + ((_Tp)1)))) * __x) * __x); 
# 144
int __n = 3; 
# 145
while (1) 
# 146
{ 
# 147
_Tp __npam1 = ((_Tp)(__n - 1)) + __a; 
# 148
_Tp __npcm1 = ((_Tp)(__n - 1)) + __c; 
# 149
_Tp __npam2 = ((_Tp)(__n - 2)) + __a; 
# 150
_Tp __npcm2 = ((_Tp)(__n - 2)) + __c; 
# 151
_Tp __tnm1 = (_Tp)((2 * __n) - 1); 
# 152
_Tp __tnm3 = (_Tp)((2 * __n) - 3); 
# 153
_Tp __tnm5 = (_Tp)((2 * __n) - 5); 
# 154
_Tp __F1 = (((_Tp)(__n - 2)) - __a) / ((((_Tp)2) * __tnm3) * __npcm1); 
# 155
_Tp __F2 = ((((_Tp)__n) + __a) * __npam1) / ((((((_Tp)4) * __tnm1) * __tnm3) * __npcm2) * __npcm1); 
# 157
_Tp __F3 = (((-__npam2) * __npam1) * (((_Tp)(__n - 2)) - __a)) / ((((((((_Tp)8) * __tnm3) * __tnm3) * __tnm5) * (((_Tp)(__n - 3)) + __c)) * __npcm2) * __npcm1); 
# 160
_Tp __E = ((-__npam1) * (((_Tp)(__n - 1)) - __c)) / (((((_Tp)2) * __tnm3) * __npcm2) * __npcm1); 
# 163
_Tp __An = (((((_Tp)1) + (__F1 * __x)) * __Anm1) + (((__E + (__F2 * __x)) * __x) * __Anm2)) + ((__F3 * __x3) * __Anm3); 
# 165
_Tp __Bn = (((((_Tp)1) + (__F1 * __x)) * __Bnm1) + (((__E + (__F2 * __x)) * __x) * __Bnm2)) + ((__F3 * __x3) * __Bnm3); 
# 167
_Tp __r = __An / __Bn; 
# 169
__prec = std::abs((__F - __r) / __F); 
# 170
__F = __r; 
# 172
if ((__prec < __eps) || (__n > __nmax)) { 
# 173
break; }  
# 175
if ((std::abs(__An) > __big) || (std::abs(__Bn) > __big)) 
# 176
{ 
# 177
__An /= __big; 
# 178
__Bn /= __big; 
# 179
__Anm1 /= __big; 
# 180
__Bnm1 /= __big; 
# 181
__Anm2 /= __big; 
# 182
__Bnm2 /= __big; 
# 183
__Anm3 /= __big; 
# 184
__Bnm3 /= __big; 
# 185
} else { 
# 186
if ((std::abs(__An) < (((_Tp)1) / __big)) || (std::abs(__Bn) < (((_Tp)1) / __big))) 
# 188
{ 
# 189
__An *= __big; 
# 190
__Bn *= __big; 
# 191
__Anm1 *= __big; 
# 192
__Bnm1 *= __big; 
# 193
__Anm2 *= __big; 
# 194
__Bnm2 *= __big; 
# 195
__Anm3 *= __big; 
# 196
__Bnm3 *= __big; 
# 197
}  }  
# 199
++__n; 
# 200
__Bnm3 = __Bnm2; 
# 201
__Bnm2 = __Bnm1; 
# 202
__Bnm1 = __Bn; 
# 203
__Anm3 = __Anm2; 
# 204
__Anm2 = __Anm1; 
# 205
__Anm1 = __An; 
# 206
}  
# 208
if (__n >= __nmax) { 
# 209
std::__throw_runtime_error("Iteration failed to converge in __conf_hyperg_luke."); }  
# 212
return __F; 
# 213
} 
# 227 "/usr/include/c++/12/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 229
__conf_hyperg(_Tp __a, _Tp __c, _Tp __x) 
# 230
{ 
# 232
const _Tp __c_nint = std::nearbyint(__c); 
# 236
if ((__isnan(__a) || __isnan(__c)) || __isnan(__x)) { 
# 237
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 238
if ((__c_nint == __c) && (__c_nint <= 0)) { 
# 239
return std::template numeric_limits< _Tp> ::infinity(); } else { 
# 240
if (__a == ((_Tp)0)) { 
# 241
return (_Tp)1; } else { 
# 242
if (__c == __a) { 
# 243
return std::exp(__x); } else { 
# 244
if (__x < ((_Tp)0)) { 
# 245
return __conf_hyperg_luke(__a, __c, __x); } else { 
# 247
return __conf_hyperg_series(__a, __c, __x); }  }  }  }  }  
# 248
} 
# 271 "/usr/include/c++/12/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 273
__hyperg_series(_Tp __a, _Tp __b, _Tp __c, _Tp __x) 
# 274
{ 
# 275
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 277
_Tp __term = ((_Tp)1); 
# 278
_Tp __Fabc = ((_Tp)1); 
# 279
const unsigned __max_iter = (100000); 
# 280
unsigned __i; 
# 281
for (__i = (0); __i < __max_iter; ++__i) 
# 282
{ 
# 283
__term *= ((((__a + ((_Tp)__i)) * (__b + ((_Tp)__i))) * __x) / ((__c + ((_Tp)__i)) * ((_Tp)((1) + __i)))); 
# 285
if (std::abs(__term) < __eps) 
# 286
{ 
# 287
break; 
# 288
}  
# 289
__Fabc += __term; 
# 290
}  
# 291
if (__i == __max_iter) { 
# 292
std::__throw_runtime_error("Series failed to converge in __hyperg_series."); }  
# 295
return __Fabc; 
# 296
} 
# 304
template< class _Tp> _Tp 
# 306
__hyperg_luke(_Tp __a, _Tp __b, _Tp __c, _Tp __xin) 
# 307
{ 
# 308
const _Tp __big = std::pow(std::template numeric_limits< _Tp> ::max(), (_Tp)(0.16L)); 
# 309
const int __nmax = 20000; 
# 310
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 311
const _Tp __x = (-__xin); 
# 312
const _Tp __x3 = (__x * __x) * __x; 
# 313
const _Tp __t0 = (__a * __b) / __c; 
# 314
const _Tp __t1 = ((__a + ((_Tp)1)) * (__b + ((_Tp)1))) / (((_Tp)2) * __c); 
# 315
const _Tp __t2 = ((__a + ((_Tp)2)) * (__b + ((_Tp)2))) / (((_Tp)2) * (__c + ((_Tp)1))); 
# 318
_Tp __F = ((_Tp)1); 
# 320
_Tp __Bnm3 = ((_Tp)1); 
# 321
_Tp __Bnm2 = ((_Tp)1) + (__t1 * __x); 
# 322
_Tp __Bnm1 = ((_Tp)1) + ((__t2 * __x) * (((_Tp)1) + ((__t1 / ((_Tp)3)) * __x))); 
# 324
_Tp __Anm3 = ((_Tp)1); 
# 325
_Tp __Anm2 = __Bnm2 - (__t0 * __x); 
# 326
_Tp __Anm1 = (__Bnm1 - ((__t0 * (((_Tp)1) + (__t2 * __x))) * __x)) + ((((__t0 * __t1) * (__c / (__c + ((_Tp)1)))) * __x) * __x); 
# 329
int __n = 3; 
# 330
while (1) 
# 331
{ 
# 332
const _Tp __npam1 = ((_Tp)(__n - 1)) + __a; 
# 333
const _Tp __npbm1 = ((_Tp)(__n - 1)) + __b; 
# 334
const _Tp __npcm1 = ((_Tp)(__n - 1)) + __c; 
# 335
const _Tp __npam2 = ((_Tp)(__n - 2)) + __a; 
# 336
const _Tp __npbm2 = ((_Tp)(__n - 2)) + __b; 
# 337
const _Tp __npcm2 = ((_Tp)(__n - 2)) + __c; 
# 338
const _Tp __tnm1 = (_Tp)((2 * __n) - 1); 
# 339
const _Tp __tnm3 = (_Tp)((2 * __n) - 3); 
# 340
const _Tp __tnm5 = (_Tp)((2 * __n) - 5); 
# 341
const _Tp __n2 = __n * __n; 
# 342
const _Tp __F1 = (((((((_Tp)3) * __n2) + (((__a + __b) - ((_Tp)6)) * __n)) + ((_Tp)2)) - (__a * __b)) - (((_Tp)2) * (__a + __b))) / ((((_Tp)2) * __tnm3) * __npcm1); 
# 345
const _Tp __F2 = (((-((((((_Tp)3) * __n2) - (((__a + __b) + ((_Tp)6)) * __n)) + ((_Tp)2)) - (__a * __b))) * __npam1) * __npbm1) / ((((((_Tp)4) * __tnm1) * __tnm3) * __npcm2) * __npcm1); 
# 348
const _Tp __F3 = (((((__npam2 * __npam1) * __npbm2) * __npbm1) * (((_Tp)(__n - 2)) - __a)) * (((_Tp)(__n - 2)) - __b)) / ((((((((_Tp)8) * __tnm3) * __tnm3) * __tnm5) * (((_Tp)(__n - 3)) + __c)) * __npcm2) * __npcm1); 
# 352
const _Tp __E = (((-__npam1) * __npbm1) * (((_Tp)(__n - 1)) - __c)) / (((((_Tp)2) * __tnm3) * __npcm2) * __npcm1); 
# 355
_Tp __An = (((((_Tp)1) + (__F1 * __x)) * __Anm1) + (((__E + (__F2 * __x)) * __x) * __Anm2)) + ((__F3 * __x3) * __Anm3); 
# 357
_Tp __Bn = (((((_Tp)1) + (__F1 * __x)) * __Bnm1) + (((__E + (__F2 * __x)) * __x) * __Bnm2)) + ((__F3 * __x3) * __Bnm3); 
# 359
const _Tp __r = __An / __Bn; 
# 361
const _Tp __prec = std::abs((__F - __r) / __F); 
# 362
__F = __r; 
# 364
if ((__prec < __eps) || (__n > __nmax)) { 
# 365
break; }  
# 367
if ((std::abs(__An) > __big) || (std::abs(__Bn) > __big)) 
# 368
{ 
# 369
__An /= __big; 
# 370
__Bn /= __big; 
# 371
__Anm1 /= __big; 
# 372
__Bnm1 /= __big; 
# 373
__Anm2 /= __big; 
# 374
__Bnm2 /= __big; 
# 375
__Anm3 /= __big; 
# 376
__Bnm3 /= __big; 
# 377
} else { 
# 378
if ((std::abs(__An) < (((_Tp)1) / __big)) || (std::abs(__Bn) < (((_Tp)1) / __big))) 
# 380
{ 
# 381
__An *= __big; 
# 382
__Bn *= __big; 
# 383
__Anm1 *= __big; 
# 384
__Bnm1 *= __big; 
# 385
__Anm2 *= __big; 
# 386
__Bnm2 *= __big; 
# 387
__Anm3 *= __big; 
# 388
__Bnm3 *= __big; 
# 389
}  }  
# 391
++__n; 
# 392
__Bnm3 = __Bnm2; 
# 393
__Bnm2 = __Bnm1; 
# 394
__Bnm1 = __Bn; 
# 395
__Anm3 = __Anm2; 
# 396
__Anm2 = __Anm1; 
# 397
__Anm1 = __An; 
# 398
}  
# 400
if (__n >= __nmax) { 
# 401
std::__throw_runtime_error("Iteration failed to converge in __hyperg_luke."); }  
# 404
return __F; 
# 405
} 
# 438 "/usr/include/c++/12/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 440
__hyperg_reflect(_Tp __a, _Tp __b, _Tp __c, _Tp __x) 
# 441
{ 
# 442
const _Tp __d = (__c - __a) - __b; 
# 443
const int __intd = std::floor(__d + ((_Tp)(0.5L))); 
# 444
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 445
const _Tp __toler = ((_Tp)1000) * __eps; 
# 446
const _Tp __log_max = std::log(std::template numeric_limits< _Tp> ::max()); 
# 447
const bool __d_integer = std::abs(__d - __intd) < __toler; 
# 449
if (__d_integer) 
# 450
{ 
# 451
const _Tp __ln_omx = std::log(((_Tp)1) - __x); 
# 452
const _Tp __ad = std::abs(__d); 
# 453
_Tp __F1, __F2; 
# 455
_Tp __d1, __d2; 
# 456
if (__d >= ((_Tp)0)) 
# 457
{ 
# 458
__d1 = __d; 
# 459
__d2 = ((_Tp)0); 
# 460
} else 
# 462
{ 
# 463
__d1 = ((_Tp)0); 
# 464
__d2 = __d; 
# 465
}  
# 467
const _Tp __lng_c = __log_gamma(__c); 
# 470
if (__ad < __eps) 
# 471
{ 
# 473
__F1 = ((_Tp)0); 
# 474
} else 
# 476
{ 
# 478
bool __ok_d1 = true; 
# 479
_Tp __lng_ad, __lng_ad1, __lng_bd1; 
# 480
try 
# 481
{ 
# 482
__lng_ad = __log_gamma(__ad); 
# 483
__lng_ad1 = __log_gamma(__a + __d1); 
# 484
__lng_bd1 = __log_gamma(__b + __d1); 
# 485
} 
# 486
catch (...) 
# 487
{ 
# 488
__ok_d1 = false; 
# 489
}  
# 491
if (__ok_d1) 
# 492
{ 
# 496
_Tp __sum1 = ((_Tp)1); 
# 497
_Tp __term = ((_Tp)1); 
# 498
_Tp __ln_pre1 = (((__lng_ad + __lng_c) + (__d2 * __ln_omx)) - __lng_ad1) - __lng_bd1; 
# 503
for (int __i = 1; __i < __ad; ++__i) 
# 504
{ 
# 505
const int __j = __i - 1; 
# 506
__term *= ((((((__a + __d2) + __j) * ((__b + __d2) + __j)) / ((((_Tp)1) + __d2) + __j)) / __i) * (((_Tp)1) - __x)); 
# 508
__sum1 += __term; 
# 509
}  
# 511
if (__ln_pre1 > __log_max) { 
# 512
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_luke."); } else { 
# 515
__F1 = (std::exp(__ln_pre1) * __sum1); }  
# 516
} else 
# 518
{ 
# 521
__F1 = ((_Tp)0); 
# 522
}  
# 523
}  
# 526
bool __ok_d2 = true; 
# 527
_Tp __lng_ad2, __lng_bd2; 
# 528
try 
# 529
{ 
# 530
__lng_ad2 = __log_gamma(__a + __d2); 
# 531
__lng_bd2 = __log_gamma(__b + __d2); 
# 532
} 
# 533
catch (...) 
# 534
{ 
# 535
__ok_d2 = false; 
# 536
}  
# 538
if (__ok_d2) 
# 539
{ 
# 542
const int __maxiter = 2000; 
# 543
const _Tp __psi_1 = (-__numeric_constants< _Tp> ::__gamma_e()); 
# 544
const _Tp __psi_1pd = __psi(((_Tp)1) + __ad); 
# 545
const _Tp __psi_apd1 = __psi(__a + __d1); 
# 546
const _Tp __psi_bpd1 = __psi(__b + __d1); 
# 548
_Tp __psi_term = (((__psi_1 + __psi_1pd) - __psi_apd1) - __psi_bpd1) - __ln_omx; 
# 550
_Tp __fact = ((_Tp)1); 
# 551
_Tp __sum2 = __psi_term; 
# 552
_Tp __ln_pre2 = ((__lng_c + (__d1 * __ln_omx)) - __lng_ad2) - __lng_bd2; 
# 556
int __j; 
# 557
for (__j = 1; __j < __maxiter; ++__j) 
# 558
{ 
# 561
const _Tp __term1 = (((_Tp)1) / ((_Tp)__j)) + (((_Tp)1) / (__ad + __j)); 
# 563
const _Tp __term2 = (((_Tp)1) / ((__a + __d1) + ((_Tp)(__j - 1)))) + (((_Tp)1) / ((__b + __d1) + ((_Tp)(__j - 1)))); 
# 565
__psi_term += (__term1 - __term2); 
# 566
__fact *= (((((__a + __d1) + ((_Tp)(__j - 1))) * ((__b + __d1) + ((_Tp)(__j - 1)))) / ((__ad + __j) * __j)) * (((_Tp)1) - __x)); 
# 569
const _Tp __delta = __fact * __psi_term; 
# 570
__sum2 += __delta; 
# 571
if (std::abs(__delta) < (__eps * std::abs(__sum2))) { 
# 572
break; }  
# 573
}  
# 574
if (__j == __maxiter) { 
# 575
std::__throw_runtime_error("Sum F2 failed to converge in __hyperg_reflect"); }  
# 578
if (__sum2 == ((_Tp)0)) { 
# 579
__F2 = ((_Tp)0); } else { 
# 581
__F2 = (std::exp(__ln_pre2) * __sum2); }  
# 582
} else 
# 584
{ 
# 587
__F2 = ((_Tp)0); 
# 588
}  
# 590
const _Tp __sgn_2 = (((__intd % 2) == 1) ? -((_Tp)1) : ((_Tp)1)); 
# 591
const _Tp __F = __F1 + (__sgn_2 * __F2); 
# 593
return __F; 
# 594
} else 
# 596
{ 
# 601
bool __ok1 = true; 
# 602
_Tp __sgn_g1ca = ((_Tp)0), __ln_g1ca = ((_Tp)0); 
# 603
_Tp __sgn_g1cb = ((_Tp)0), __ln_g1cb = ((_Tp)0); 
# 604
try 
# 605
{ 
# 606
__sgn_g1ca = __log_gamma_sign(__c - __a); 
# 607
__ln_g1ca = __log_gamma(__c - __a); 
# 608
__sgn_g1cb = __log_gamma_sign(__c - __b); 
# 609
__ln_g1cb = __log_gamma(__c - __b); 
# 610
} 
# 611
catch (...) 
# 612
{ 
# 613
__ok1 = false; 
# 614
}  
# 616
bool __ok2 = true; 
# 617
_Tp __sgn_g2a = ((_Tp)0), __ln_g2a = ((_Tp)0); 
# 618
_Tp __sgn_g2b = ((_Tp)0), __ln_g2b = ((_Tp)0); 
# 619
try 
# 620
{ 
# 621
__sgn_g2a = __log_gamma_sign(__a); 
# 622
__ln_g2a = __log_gamma(__a); 
# 623
__sgn_g2b = __log_gamma_sign(__b); 
# 624
__ln_g2b = __log_gamma(__b); 
# 625
} 
# 626
catch (...) 
# 627
{ 
# 628
__ok2 = false; 
# 629
}  
# 631
const _Tp __sgn_gc = __log_gamma_sign(__c); 
# 632
const _Tp __ln_gc = __log_gamma(__c); 
# 633
const _Tp __sgn_gd = __log_gamma_sign(__d); 
# 634
const _Tp __ln_gd = __log_gamma(__d); 
# 635
const _Tp __sgn_gmd = __log_gamma_sign(-__d); 
# 636
const _Tp __ln_gmd = __log_gamma(-__d); 
# 638
const _Tp __sgn1 = ((__sgn_gc * __sgn_gd) * __sgn_g1ca) * __sgn_g1cb; 
# 639
const _Tp __sgn2 = ((__sgn_gc * __sgn_gmd) * __sgn_g2a) * __sgn_g2b; 
# 641
_Tp __pre1, __pre2; 
# 642
if (__ok1 && __ok2) 
# 643
{ 
# 644
_Tp __ln_pre1 = ((__ln_gc + __ln_gd) - __ln_g1ca) - __ln_g1cb; 
# 645
_Tp __ln_pre2 = (((__ln_gc + __ln_gmd) - __ln_g2a) - __ln_g2b) + (__d * std::log(((_Tp)1) - __x)); 
# 647
if ((__ln_pre1 < __log_max) && (__ln_pre2 < __log_max)) 
# 648
{ 
# 649
__pre1 = std::exp(__ln_pre1); 
# 650
__pre2 = std::exp(__ln_pre2); 
# 651
__pre1 *= __sgn1; 
# 652
__pre2 *= __sgn2; 
# 653
} else 
# 655
{ 
# 656
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_reflect"); 
# 658
}  
# 659
} else { 
# 660
if (__ok1 && (!__ok2)) 
# 661
{ 
# 662
_Tp __ln_pre1 = ((__ln_gc + __ln_gd) - __ln_g1ca) - __ln_g1cb; 
# 663
if (__ln_pre1 < __log_max) 
# 664
{ 
# 665
__pre1 = std::exp(__ln_pre1); 
# 666
__pre1 *= __sgn1; 
# 667
__pre2 = ((_Tp)0); 
# 668
} else 
# 670
{ 
# 671
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_reflect"); 
# 673
}  
# 674
} else { 
# 675
if ((!__ok1) && __ok2) 
# 676
{ 
# 677
_Tp __ln_pre2 = (((__ln_gc + __ln_gmd) - __ln_g2a) - __ln_g2b) + (__d * std::log(((_Tp)1) - __x)); 
# 679
if (__ln_pre2 < __log_max) 
# 680
{ 
# 681
__pre1 = ((_Tp)0); 
# 682
__pre2 = std::exp(__ln_pre2); 
# 683
__pre2 *= __sgn2; 
# 684
} else 
# 686
{ 
# 687
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_reflect"); 
# 689
}  
# 690
} else 
# 692
{ 
# 693
__pre1 = ((_Tp)0); 
# 694
__pre2 = ((_Tp)0); 
# 695
std::__throw_runtime_error("Underflow of gamma functions in __hyperg_reflect"); 
# 697
}  }  }  
# 699
const _Tp __F1 = __hyperg_series(__a, __b, ((_Tp)1) - __d, ((_Tp)1) - __x); 
# 701
const _Tp __F2 = __hyperg_series(__c - __a, __c - __b, ((_Tp)1) + __d, ((_Tp)1) - __x); 
# 704
const _Tp __F = (__pre1 * __F1) + (__pre2 * __F2); 
# 706
return __F; 
# 707
}  
# 708
} 
# 728 "/usr/include/c++/12/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 730
__hyperg(_Tp __a, _Tp __b, _Tp __c, _Tp __x) 
# 731
{ 
# 733
const _Tp __a_nint = std::nearbyint(__a); 
# 734
const _Tp __b_nint = std::nearbyint(__b); 
# 735
const _Tp __c_nint = std::nearbyint(__c); 
# 741
const _Tp __toler = ((_Tp)1000) * std::template numeric_limits< _Tp> ::epsilon(); 
# 742
if (std::abs(__x) >= ((_Tp)1)) { 
# 743
std::__throw_domain_error("Argument outside unit circle in __hyperg."); } else { 
# 745
if (((__isnan(__a) || __isnan(__b)) || __isnan(__c)) || __isnan(__x)) { 
# 747
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 748
if ((__c_nint == __c) && (__c_nint <= ((_Tp)0))) { 
# 749
return std::template numeric_limits< _Tp> ::infinity(); } else { 
# 750
if ((std::abs(__c - __b) < __toler) || (std::abs(__c - __a) < __toler)) { 
# 751
return std::pow(((_Tp)1) - __x, (__c - __a) - __b); } else { 
# 752
if ((__a >= ((_Tp)0)) && (__b >= ((_Tp)0)) && (__c >= ((_Tp)0)) && (__x >= ((_Tp)0)) && (__x < ((_Tp)(0.995L)))) { 
# 754
return __hyperg_series(__a, __b, __c, __x); } else { 
# 755
if ((std::abs(__a) < ((_Tp)10)) && (std::abs(__b) < ((_Tp)10))) 
# 756
{ 
# 759
if ((__a < ((_Tp)0)) && (std::abs(__a - __a_nint) < __toler)) { 
# 760
return __hyperg_series(__a_nint, __b, __c, __x); } else { 
# 761
if ((__b < ((_Tp)0)) && (std::abs(__b - __b_nint) < __toler)) { 
# 762
return __hyperg_series(__a, __b_nint, __c, __x); } else { 
# 763
if (__x < (-((_Tp)(0.25L)))) { 
# 764
return __hyperg_luke(__a, __b, __c, __x); } else { 
# 765
if (__x < ((_Tp)(0.5L))) { 
# 766
return __hyperg_series(__a, __b, __c, __x); } else { 
# 768
if (std::abs(__c) > ((_Tp)10)) { 
# 769
return __hyperg_series(__a, __b, __c, __x); } else { 
# 771
return __hyperg_reflect(__a, __b, __c, __x); }  }  }  }  }  
# 772
} else { 
# 774
return __hyperg_luke(__a, __b, __c, __x); }  }  }  }  }  }  
# 775
} 
# 776
}
# 783
}
# 49 "/usr/include/c++/12/tr1/legendre_function.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/usr/include/c++/12/tr1/legendre_function.tcc" 3
namespace __detail { 
# 80 "/usr/include/c++/12/tr1/legendre_function.tcc" 3
template< class _Tp> _Tp 
# 82
__poly_legendre_p(unsigned __l, _Tp __x) 
# 83
{ 
# 85
if (__isnan(__x)) { 
# 86
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 87
if (__x == (+((_Tp)1))) { 
# 88
return +((_Tp)1); } else { 
# 89
if (__x == (-((_Tp)1))) { 
# 90
return (((__l % (2)) == (1)) ? -((_Tp)1) : (+((_Tp)1))); } else 
# 92
{ 
# 93
_Tp __p_lm2 = ((_Tp)1); 
# 94
if (__l == (0)) { 
# 95
return __p_lm2; }  
# 97
_Tp __p_lm1 = __x; 
# 98
if (__l == (1)) { 
# 99
return __p_lm1; }  
# 101
_Tp __p_l = (0); 
# 102
for (unsigned __ll = (2); __ll <= __l; ++__ll) 
# 103
{ 
# 106
__p_l = ((((((_Tp)2) * __x) * __p_lm1) - __p_lm2) - (((__x * __p_lm1) - __p_lm2) / ((_Tp)__ll))); 
# 108
__p_lm2 = __p_lm1; 
# 109
__p_lm1 = __p_l; 
# 110
}  
# 112
return __p_l; 
# 113
}  }  }  
# 114
} 
# 136 "/usr/include/c++/12/tr1/legendre_function.tcc" 3
template< class _Tp> _Tp 
# 138
__assoc_legendre_p(unsigned __l, unsigned __m, _Tp __x, _Tp 
# 139
__phase = (_Tp)(+1)) 
# 140
{ 
# 142
if (__m > __l) { 
# 143
return (_Tp)0; } else { 
# 144
if (__isnan(__x)) { 
# 145
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 146
if (__m == (0)) { 
# 147
return __poly_legendre_p(__l, __x); } else 
# 149
{ 
# 150
_Tp __p_mm = ((_Tp)1); 
# 151
if (__m > (0)) 
# 152
{ 
# 155
_Tp __root = std::sqrt(((_Tp)1) - __x) * std::sqrt(((_Tp)1) + __x); 
# 156
_Tp __fact = ((_Tp)1); 
# 157
for (unsigned __i = (1); __i <= __m; ++__i) 
# 158
{ 
# 159
__p_mm *= ((__phase * __fact) * __root); 
# 160
__fact += ((_Tp)2); 
# 161
}  
# 162
}  
# 163
if (__l == __m) { 
# 164
return __p_mm; }  
# 166
_Tp __p_mp1m = (((_Tp)(((2) * __m) + (1))) * __x) * __p_mm; 
# 167
if (__l == (__m + (1))) { 
# 168
return __p_mp1m; }  
# 170
_Tp __p_lm2m = __p_mm; 
# 171
_Tp __P_lm1m = __p_mp1m; 
# 172
_Tp __p_lm = ((_Tp)0); 
# 173
for (unsigned __j = __m + (2); __j <= __l; ++__j) 
# 174
{ 
# 175
__p_lm = ((((((_Tp)(((2) * __j) - (1))) * __x) * __P_lm1m) - (((_Tp)((__j + __m) - (1))) * __p_lm2m)) / ((_Tp)(__j - __m))); 
# 177
__p_lm2m = __P_lm1m; 
# 178
__P_lm1m = __p_lm; 
# 179
}  
# 181
return __p_lm; 
# 182
}  }  }  
# 183
} 
# 214 "/usr/include/c++/12/tr1/legendre_function.tcc" 3
template< class _Tp> _Tp 
# 216
__sph_legendre(unsigned __l, unsigned __m, _Tp __theta) 
# 217
{ 
# 218
if (__isnan(__theta)) { 
# 219
return std::template numeric_limits< _Tp> ::quiet_NaN(); }  
# 221
const _Tp __x = std::cos(__theta); 
# 223
if (__m > __l) { 
# 224
return (_Tp)0; } else { 
# 225
if (__m == (0)) 
# 226
{ 
# 227
_Tp __P = __poly_legendre_p(__l, __x); 
# 228
_Tp __fact = std::sqrt(((_Tp)(((2) * __l) + (1))) / (((_Tp)4) * __numeric_constants< _Tp> ::__pi())); 
# 230
__P *= __fact; 
# 231
return __P; 
# 232
} else { 
# 233
if ((__x == ((_Tp)1)) || (__x == (-((_Tp)1)))) 
# 234
{ 
# 236
return (_Tp)0; 
# 237
} else 
# 239
{ 
# 245
const _Tp __sgn = ((__m % (2)) == (1)) ? -((_Tp)1) : ((_Tp)1); 
# 246
const _Tp __y_mp1m_factor = __x * std::sqrt((_Tp)(((2) * __m) + (3))); 
# 248
const _Tp __lncirc = std::log1p((-__x) * __x); 
# 254
const _Tp __lnpoch = std::lgamma((_Tp)(__m + ((_Tp)(0.5L)))) - std::lgamma((_Tp)__m); 
# 260
const _Tp __lnpre_val = ((-((_Tp)(0.25L))) * __numeric_constants< _Tp> ::__lnpi()) + (((_Tp)(0.5L)) * (__lnpoch + (__m * __lncirc))); 
# 263
const _Tp __sr = std::sqrt((((_Tp)2) + (((_Tp)1) / __m)) / (((_Tp)4) * __numeric_constants< _Tp> ::__pi())); 
# 265
_Tp __y_mm = (__sgn * __sr) * std::exp(__lnpre_val); 
# 266
_Tp __y_mp1m = __y_mp1m_factor * __y_mm; 
# 268
if (__l == __m) { 
# 269
return __y_mm; } else { 
# 270
if (__l == (__m + (1))) { 
# 271
return __y_mp1m; } else 
# 273
{ 
# 274
_Tp __y_lm = ((_Tp)0); 
# 277
for (unsigned __ll = __m + (2); __ll <= __l; ++__ll) 
# 278
{ 
# 279
const _Tp __rat1 = ((_Tp)(__ll - __m)) / ((_Tp)(__ll + __m)); 
# 280
const _Tp __rat2 = ((_Tp)((__ll - __m) - (1))) / ((_Tp)((__ll + __m) - (1))); 
# 281
const _Tp __fact1 = std::sqrt((__rat1 * ((_Tp)(((2) * __ll) + (1)))) * ((_Tp)(((2) * __ll) - (1)))); 
# 283
const _Tp __fact2 = std::sqrt(((__rat1 * __rat2) * ((_Tp)(((2) * __ll) + (1)))) / ((_Tp)(((2) * __ll) - (3)))); 
# 285
__y_lm = ((((__x * __y_mp1m) * __fact1) - ((((__ll + __m) - (1)) * __y_mm) * __fact2)) / ((_Tp)(__ll - __m))); 
# 287
__y_mm = __y_mp1m; 
# 288
__y_mp1m = __y_lm; 
# 289
}  
# 291
return __y_lm; 
# 292
}  }  
# 293
}  }  }  
# 294
} 
# 295
}
# 302
}
# 51 "/usr/include/c++/12/tr1/modified_bessel_func.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/usr/include/c++/12/tr1/modified_bessel_func.tcc" 3
namespace __detail { 
# 83 "/usr/include/c++/12/tr1/modified_bessel_func.tcc" 3
template< class _Tp> void 
# 85
__bessel_ik(_Tp __nu, _Tp __x, _Tp &
# 86
__Inu, _Tp &__Knu, _Tp &__Ipnu, _Tp &__Kpnu) 
# 87
{ 
# 88
if (__x == ((_Tp)0)) 
# 89
{ 
# 90
if (__nu == ((_Tp)0)) 
# 91
{ 
# 92
__Inu = ((_Tp)1); 
# 93
__Ipnu = ((_Tp)0); 
# 94
} else { 
# 95
if (__nu == ((_Tp)1)) 
# 96
{ 
# 97
__Inu = ((_Tp)0); 
# 98
__Ipnu = ((_Tp)(0.5L)); 
# 99
} else 
# 101
{ 
# 102
__Inu = ((_Tp)0); 
# 103
__Ipnu = ((_Tp)0); 
# 104
}  }  
# 105
__Knu = std::template numeric_limits< _Tp> ::infinity(); 
# 106
__Kpnu = (-std::template numeric_limits< _Tp> ::infinity()); 
# 107
return; 
# 108
}  
# 110
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 111
const _Tp __fp_min = ((_Tp)10) * std::template numeric_limits< _Tp> ::epsilon(); 
# 112
const int __max_iter = 15000; 
# 113
const _Tp __x_min = ((_Tp)2); 
# 115
const int __nl = static_cast< int>(__nu + ((_Tp)(0.5L))); 
# 117
const _Tp __mu = __nu - __nl; 
# 118
const _Tp __mu2 = __mu * __mu; 
# 119
const _Tp __xi = ((_Tp)1) / __x; 
# 120
const _Tp __xi2 = ((_Tp)2) * __xi; 
# 121
_Tp __h = __nu * __xi; 
# 122
if (__h < __fp_min) { 
# 123
__h = __fp_min; }  
# 124
_Tp __b = __xi2 * __nu; 
# 125
_Tp __d = ((_Tp)0); 
# 126
_Tp __c = __h; 
# 127
int __i; 
# 128
for (__i = 1; __i <= __max_iter; ++__i) 
# 129
{ 
# 130
__b += __xi2; 
# 131
__d = (((_Tp)1) / (__b + __d)); 
# 132
__c = (__b + (((_Tp)1) / __c)); 
# 133
const _Tp __del = __c * __d; 
# 134
__h *= __del; 
# 135
if (std::abs(__del - ((_Tp)1)) < __eps) { 
# 136
break; }  
# 137
}  
# 138
if (__i > __max_iter) { 
# 139
std::__throw_runtime_error("Argument x too large in __bessel_ik; try asymptotic expansion."); }  
# 142
_Tp __Inul = __fp_min; 
# 143
_Tp __Ipnul = __h * __Inul; 
# 144
_Tp __Inul1 = __Inul; 
# 145
_Tp __Ipnu1 = __Ipnul; 
# 146
_Tp __fact = __nu * __xi; 
# 147
for (int __l = __nl; __l >= 1; --__l) 
# 148
{ 
# 149
const _Tp __Inutemp = (__fact * __Inul) + __Ipnul; 
# 150
__fact -= __xi; 
# 151
__Ipnul = ((__fact * __Inutemp) + __Inul); 
# 152
__Inul = __Inutemp; 
# 153
}  
# 154
_Tp __f = __Ipnul / __Inul; 
# 155
_Tp __Kmu, __Knu1; 
# 156
if (__x < __x_min) 
# 157
{ 
# 158
const _Tp __x2 = __x / ((_Tp)2); 
# 159
const _Tp __pimu = __numeric_constants< _Tp> ::__pi() * __mu; 
# 160
const _Tp __fact = (std::abs(__pimu) < __eps) ? (_Tp)1 : (__pimu / std::sin(__pimu)); 
# 162
_Tp __d = (-std::log(__x2)); 
# 163
_Tp __e = __mu * __d; 
# 164
const _Tp __fact2 = (std::abs(__e) < __eps) ? (_Tp)1 : (std::sinh(__e) / __e); 
# 166
_Tp __gam1, __gam2, __gampl, __gammi; 
# 167
__gamma_temme(__mu, __gam1, __gam2, __gampl, __gammi); 
# 168
_Tp __ff = __fact * ((__gam1 * std::cosh(__e)) + ((__gam2 * __fact2) * __d)); 
# 170
_Tp __sum = __ff; 
# 171
__e = std::exp(__e); 
# 172
_Tp __p = __e / (((_Tp)2) * __gampl); 
# 173
_Tp __q = ((_Tp)1) / ((((_Tp)2) * __e) * __gammi); 
# 174
_Tp __c = ((_Tp)1); 
# 175
__d = (__x2 * __x2); 
# 176
_Tp __sum1 = __p; 
# 177
int __i; 
# 178
for (__i = 1; __i <= __max_iter; ++__i) 
# 179
{ 
# 180
__ff = ((((__i * __ff) + __p) + __q) / ((__i * __i) - __mu2)); 
# 181
__c *= (__d / __i); 
# 182
__p /= (__i - __mu); 
# 183
__q /= (__i + __mu); 
# 184
const _Tp __del = __c * __ff; 
# 185
__sum += __del; 
# 186
const _Tp __del1 = __c * (__p - (__i * __ff)); 
# 187
__sum1 += __del1; 
# 188
if (std::abs(__del) < (__eps * std::abs(__sum))) { 
# 189
break; }  
# 190
}  
# 191
if (__i > __max_iter) { 
# 192
std::__throw_runtime_error("Bessel k series failed to converge in __bessel_ik."); }  
# 194
__Kmu = __sum; 
# 195
__Knu1 = (__sum1 * __xi2); 
# 196
} else 
# 198
{ 
# 199
_Tp __b = ((_Tp)2) * (((_Tp)1) + __x); 
# 200
_Tp __d = ((_Tp)1) / __b; 
# 201
_Tp __delh = __d; 
# 202
_Tp __h = __delh; 
# 203
_Tp __q1 = ((_Tp)0); 
# 204
_Tp __q2 = ((_Tp)1); 
# 205
_Tp __a1 = ((_Tp)(0.25L)) - __mu2; 
# 206
_Tp __q = __c = __a1; 
# 207
_Tp __a = (-__a1); 
# 208
_Tp __s = ((_Tp)1) + (__q * __delh); 
# 209
int __i; 
# 210
for (__i = 2; __i <= __max_iter; ++__i) 
# 211
{ 
# 212
__a -= (2 * (__i - 1)); 
# 213
__c = (((-__a) * __c) / __i); 
# 214
const _Tp __qnew = (__q1 - (__b * __q2)) / __a; 
# 215
__q1 = __q2; 
# 216
__q2 = __qnew; 
# 217
__q += (__c * __qnew); 
# 218
__b += ((_Tp)2); 
# 219
__d = (((_Tp)1) / (__b + (__a * __d))); 
# 220
__delh = (((__b * __d) - ((_Tp)1)) * __delh); 
# 221
__h += __delh; 
# 222
const _Tp __dels = __q * __delh; 
# 223
__s += __dels; 
# 224
if (std::abs(__dels / __s) < __eps) { 
# 225
break; }  
# 226
}  
# 227
if (__i > __max_iter) { 
# 228
std::__throw_runtime_error("Steed\'s method failed in __bessel_ik."); }  
# 230
__h = (__a1 * __h); 
# 231
__Kmu = ((std::sqrt(__numeric_constants< _Tp> ::__pi() / (((_Tp)2) * __x)) * std::exp(-__x)) / __s); 
# 233
__Knu1 = ((__Kmu * (((__mu + __x) + ((_Tp)(0.5L))) - __h)) * __xi); 
# 234
}  
# 236
_Tp __Kpmu = ((__mu * __xi) * __Kmu) - __Knu1; 
# 237
_Tp __Inumu = __xi / ((__f * __Kmu) - __Kpmu); 
# 238
__Inu = ((__Inumu * __Inul1) / __Inul); 
# 239
__Ipnu = ((__Inumu * __Ipnu1) / __Inul); 
# 240
for (__i = 1; __i <= __nl; ++__i) 
# 241
{ 
# 242
const _Tp __Knutemp = (((__mu + __i) * __xi2) * __Knu1) + __Kmu; 
# 243
__Kmu = __Knu1; 
# 244
__Knu1 = __Knutemp; 
# 245
}  
# 246
__Knu = __Kmu; 
# 247
__Kpnu = (((__nu * __xi) * __Kmu) - __Knu1); 
# 250
} 
# 267 "/usr/include/c++/12/tr1/modified_bessel_func.tcc" 3
template< class _Tp> _Tp 
# 269
__cyl_bessel_i(_Tp __nu, _Tp __x) 
# 270
{ 
# 271
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 272
std::__throw_domain_error("Bad argument in __cyl_bessel_i."); } else { 
# 274
if (__isnan(__nu) || __isnan(__x)) { 
# 275
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 276
if ((__x * __x) < (((_Tp)10) * (__nu + ((_Tp)1)))) { 
# 277
return __cyl_bessel_ij_series(__nu, __x, +((_Tp)1), 200); } else 
# 279
{ 
# 280
_Tp __I_nu, __K_nu, __Ip_nu, __Kp_nu; 
# 281
__bessel_ik(__nu, __x, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 282
return __I_nu; 
# 283
}  }  }  
# 284
} 
# 303 "/usr/include/c++/12/tr1/modified_bessel_func.tcc" 3
template< class _Tp> _Tp 
# 305
__cyl_bessel_k(_Tp __nu, _Tp __x) 
# 306
{ 
# 307
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 308
std::__throw_domain_error("Bad argument in __cyl_bessel_k."); } else { 
# 310
if (__isnan(__nu) || __isnan(__x)) { 
# 311
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else 
# 313
{ 
# 314
_Tp __I_nu, __K_nu, __Ip_nu, __Kp_nu; 
# 315
__bessel_ik(__nu, __x, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 316
return __K_nu; 
# 317
}  }  
# 318
} 
# 337 "/usr/include/c++/12/tr1/modified_bessel_func.tcc" 3
template< class _Tp> void 
# 339
__sph_bessel_ik(unsigned __n, _Tp __x, _Tp &
# 340
__i_n, _Tp &__k_n, _Tp &__ip_n, _Tp &__kp_n) 
# 341
{ 
# 342
const _Tp __nu = ((_Tp)__n) + ((_Tp)(0.5L)); 
# 344
_Tp __I_nu, __Ip_nu, __K_nu, __Kp_nu; 
# 345
__bessel_ik(__nu, __x, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 347
const _Tp __factor = __numeric_constants< _Tp> ::__sqrtpio2() / std::sqrt(__x); 
# 350
__i_n = (__factor * __I_nu); 
# 351
__k_n = (__factor * __K_nu); 
# 352
__ip_n = ((__factor * __Ip_nu) - (__i_n / (((_Tp)2) * __x))); 
# 353
__kp_n = ((__factor * __Kp_nu) - (__k_n / (((_Tp)2) * __x))); 
# 356
} 
# 373 "/usr/include/c++/12/tr1/modified_bessel_func.tcc" 3
template< class _Tp> void 
# 375
__airy(_Tp __x, _Tp &__Ai, _Tp &__Bi, _Tp &__Aip, _Tp &__Bip) 
# 376
{ 
# 377
const _Tp __absx = std::abs(__x); 
# 378
const _Tp __rootx = std::sqrt(__absx); 
# 379
const _Tp __z = ((((_Tp)2) * __absx) * __rootx) / ((_Tp)3); 
# 380
const _Tp _S_inf = std::template numeric_limits< _Tp> ::infinity(); 
# 382
if (__isnan(__x)) { 
# 383
__Bip = (__Aip = (__Bi = (__Ai = std::template numeric_limits< _Tp> ::quiet_NaN()))); } else { 
# 384
if (__z == _S_inf) 
# 385
{ 
# 386
__Aip = (__Ai = ((_Tp)0)); 
# 387
__Bip = (__Bi = _S_inf); 
# 388
} else { 
# 389
if (__z == (-_S_inf)) { 
# 390
__Bip = (__Aip = (__Bi = (__Ai = ((_Tp)0)))); } else { 
# 391
if (__x > ((_Tp)0)) 
# 392
{ 
# 393
_Tp __I_nu, __Ip_nu, __K_nu, __Kp_nu; 
# 395
__bessel_ik(((_Tp)1) / ((_Tp)3), __z, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 396
__Ai = ((__rootx * __K_nu) / (__numeric_constants< _Tp> ::__sqrt3() * __numeric_constants< _Tp> ::__pi())); 
# 399
__Bi = (__rootx * ((__K_nu / __numeric_constants< _Tp> ::__pi()) + ((((_Tp)2) * __I_nu) / __numeric_constants< _Tp> ::__sqrt3()))); 
# 402
__bessel_ik(((_Tp)2) / ((_Tp)3), __z, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 403
__Aip = (((-__x) * __K_nu) / (__numeric_constants< _Tp> ::__sqrt3() * __numeric_constants< _Tp> ::__pi())); 
# 406
__Bip = (__x * ((__K_nu / __numeric_constants< _Tp> ::__pi()) + ((((_Tp)2) * __I_nu) / __numeric_constants< _Tp> ::__sqrt3()))); 
# 409
} else { 
# 410
if (__x < ((_Tp)0)) 
# 411
{ 
# 412
_Tp __J_nu, __Jp_nu, __N_nu, __Np_nu; 
# 414
__bessel_jn(((_Tp)1) / ((_Tp)3), __z, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 415
__Ai = ((__rootx * (__J_nu - (__N_nu / __numeric_constants< _Tp> ::__sqrt3()))) / ((_Tp)2)); 
# 417
__Bi = (((-__rootx) * (__N_nu + (__J_nu / __numeric_constants< _Tp> ::__sqrt3()))) / ((_Tp)2)); 
# 420
__bessel_jn(((_Tp)2) / ((_Tp)3), __z, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 421
__Aip = ((__absx * ((__N_nu / __numeric_constants< _Tp> ::__sqrt3()) + __J_nu)) / ((_Tp)2)); 
# 423
__Bip = ((__absx * ((__J_nu / __numeric_constants< _Tp> ::__sqrt3()) - __N_nu)) / ((_Tp)2)); 
# 425
} else 
# 427
{ 
# 431
__Ai = ((_Tp)(0.35502805388781723926L)); 
# 432
__Bi = (__Ai * __numeric_constants< _Tp> ::__sqrt3()); 
# 437
__Aip = (-((_Tp)(0.2588194037928067984L))); 
# 438
__Bip = ((-__Aip) * __numeric_constants< _Tp> ::__sqrt3()); 
# 439
}  }  }  }  }  
# 442
} 
# 443
}
# 449
}
# 42 "/usr/include/c++/12/tr1/poly_hermite.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 56 "/usr/include/c++/12/tr1/poly_hermite.tcc" 3
namespace __detail { 
# 72 "/usr/include/c++/12/tr1/poly_hermite.tcc" 3
template< class _Tp> _Tp 
# 74
__poly_hermite_recursion(unsigned __n, _Tp __x) 
# 75
{ 
# 77
_Tp __H_0 = (1); 
# 78
if (__n == (0)) { 
# 79
return __H_0; }  
# 82
_Tp __H_1 = 2 * __x; 
# 83
if (__n == (1)) { 
# 84
return __H_1; }  
# 87
_Tp __H_n, __H_nm1, __H_nm2; 
# 88
unsigned __i; 
# 89
for (((__H_nm2 = __H_0), (__H_nm1 = __H_1)), (__i = (2)); __i <= __n; ++__i) 
# 90
{ 
# 91
__H_n = (2 * ((__x * __H_nm1) - ((__i - (1)) * __H_nm2))); 
# 92
__H_nm2 = __H_nm1; 
# 93
__H_nm1 = __H_n; 
# 94
}  
# 96
return __H_n; 
# 97
} 
# 114 "/usr/include/c++/12/tr1/poly_hermite.tcc" 3
template< class _Tp> inline _Tp 
# 116
__poly_hermite(unsigned __n, _Tp __x) 
# 117
{ 
# 118
if (__isnan(__x)) { 
# 119
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 121
return __poly_hermite_recursion(__n, __x); }  
# 122
} 
# 123
}
# 129
}
# 44 "/usr/include/c++/12/tr1/poly_laguerre.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 60 "/usr/include/c++/12/tr1/poly_laguerre.tcc" 3
namespace __detail { 
# 75 "/usr/include/c++/12/tr1/poly_laguerre.tcc" 3
template< class _Tpa, class _Tp> _Tp 
# 77
__poly_laguerre_large_n(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 78
{ 
# 79
const _Tp __a = (-((_Tp)__n)); 
# 80
const _Tp __b = ((_Tp)__alpha1) + ((_Tp)1); 
# 81
const _Tp __eta = (((_Tp)2) * __b) - (((_Tp)4) * __a); 
# 82
const _Tp __cos2th = __x / __eta; 
# 83
const _Tp __sin2th = ((_Tp)1) - __cos2th; 
# 84
const _Tp __th = std::acos(std::sqrt(__cos2th)); 
# 85
const _Tp __pre_h = ((((__numeric_constants< _Tp> ::__pi_2() * __numeric_constants< _Tp> ::__pi_2()) * __eta) * __eta) * __cos2th) * __sin2th; 
# 90
const _Tp __lg_b = std::lgamma(((_Tp)__n) + __b); 
# 91
const _Tp __lnfact = std::lgamma((_Tp)(__n + (1))); 
# 97
_Tp __pre_term1 = (((_Tp)(0.5L)) * (((_Tp)1) - __b)) * std::log((((_Tp)(0.25L)) * __x) * __eta); 
# 99
_Tp __pre_term2 = ((_Tp)(0.25L)) * std::log(__pre_h); 
# 100
_Tp __lnpre = (((__lg_b - __lnfact) + (((_Tp)(0.5L)) * __x)) + __pre_term1) - __pre_term2; 
# 102
_Tp __ser_term1 = std::sin(__a * __numeric_constants< _Tp> ::__pi()); 
# 103
_Tp __ser_term2 = std::sin(((((_Tp)(0.25L)) * __eta) * ((((_Tp)2) * __th) - std::sin(((_Tp)2) * __th))) + __numeric_constants< _Tp> ::__pi_4()); 
# 107
_Tp __ser = __ser_term1 + __ser_term2; 
# 109
return std::exp(__lnpre) * __ser; 
# 110
} 
# 129 "/usr/include/c++/12/tr1/poly_laguerre.tcc" 3
template< class _Tpa, class _Tp> _Tp 
# 131
__poly_laguerre_hyperg(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 132
{ 
# 133
const _Tp __b = ((_Tp)__alpha1) + ((_Tp)1); 
# 134
const _Tp __mx = (-__x); 
# 135
const _Tp __tc_sgn = (__x < ((_Tp)0)) ? (_Tp)1 : (((__n % (2)) == (1)) ? -((_Tp)1) : ((_Tp)1)); 
# 138
_Tp __tc = ((_Tp)1); 
# 139
const _Tp __ax = std::abs(__x); 
# 140
for (unsigned __k = (1); __k <= __n; ++__k) { 
# 141
__tc *= (__ax / __k); }  
# 143
_Tp __term = __tc * __tc_sgn; 
# 144
_Tp __sum = __term; 
# 145
for (int __k = ((int)__n) - 1; __k >= 0; --__k) 
# 146
{ 
# 147
__term *= ((((__b + ((_Tp)__k)) / ((_Tp)(((int)__n) - __k))) * ((_Tp)(__k + 1))) / __mx); 
# 149
__sum += __term; 
# 150
}  
# 152
return __sum; 
# 153
} 
# 185 "/usr/include/c++/12/tr1/poly_laguerre.tcc" 3
template< class _Tpa, class _Tp> _Tp 
# 187
__poly_laguerre_recursion(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 188
{ 
# 190
_Tp __l_0 = ((_Tp)1); 
# 191
if (__n == (0)) { 
# 192
return __l_0; }  
# 195
_Tp __l_1 = (((-__x) + ((_Tp)1)) + ((_Tp)__alpha1)); 
# 196
if (__n == (1)) { 
# 197
return __l_1; }  
# 200
_Tp __l_n2 = __l_0; 
# 201
_Tp __l_n1 = __l_1; 
# 202
_Tp __l_n = ((_Tp)0); 
# 203
for (unsigned __nn = (2); __nn <= __n; ++__nn) 
# 204
{ 
# 205
__l_n = (((((((_Tp)(((2) * __nn) - (1))) + ((_Tp)__alpha1)) - __x) * __l_n1) / ((_Tp)__nn)) - (((((_Tp)(__nn - (1))) + ((_Tp)__alpha1)) * __l_n2) / ((_Tp)__nn))); 
# 208
__l_n2 = __l_n1; 
# 209
__l_n1 = __l_n; 
# 210
}  
# 212
return __l_n; 
# 213
} 
# 244 "/usr/include/c++/12/tr1/poly_laguerre.tcc" 3
template< class _Tpa, class _Tp> _Tp 
# 246
__poly_laguerre(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 247
{ 
# 248
if (__x < ((_Tp)0)) { 
# 249
std::__throw_domain_error("Negative argument in __poly_laguerre."); } else { 
# 252
if (__isnan(__x)) { 
# 253
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 254
if (__n == (0)) { 
# 255
return (_Tp)1; } else { 
# 256
if (__n == (1)) { 
# 257
return (((_Tp)1) + ((_Tp)__alpha1)) - __x; } else { 
# 258
if (__x == ((_Tp)0)) 
# 259
{ 
# 260
_Tp __prod = ((_Tp)__alpha1) + ((_Tp)1); 
# 261
for (unsigned __k = (2); __k <= __n; ++__k) { 
# 262
__prod *= ((((_Tp)__alpha1) + ((_Tp)__k)) / ((_Tp)__k)); }  
# 263
return __prod; 
# 264
} else { 
# 265
if ((__n > (10000000)) && (((_Tp)__alpha1) > (-((_Tp)1))) && (__x < ((((_Tp)2) * (((_Tp)__alpha1) + ((_Tp)1))) + ((_Tp)((4) * __n))))) { 
# 267
return __poly_laguerre_large_n(__n, __alpha1, __x); } else { 
# 268
if ((((_Tp)__alpha1) >= ((_Tp)0)) || ((__x > ((_Tp)0)) && (((_Tp)__alpha1) < (-((_Tp)(__n + (1))))))) { 
# 270
return __poly_laguerre_recursion(__n, __alpha1, __x); } else { 
# 272
return __poly_laguerre_hyperg(__n, __alpha1, __x); }  }  }  }  }  }  }  
# 273
} 
# 296 "/usr/include/c++/12/tr1/poly_laguerre.tcc" 3
template< class _Tp> inline _Tp 
# 298
__assoc_laguerre(unsigned __n, unsigned __m, _Tp __x) 
# 299
{ return __poly_laguerre< unsigned, _Tp> (__n, __m, __x); } 
# 316 "/usr/include/c++/12/tr1/poly_laguerre.tcc" 3
template< class _Tp> inline _Tp 
# 318
__laguerre(unsigned __n, _Tp __x) 
# 319
{ return __poly_laguerre< unsigned, _Tp> (__n, 0, __x); } 
# 320
}
# 327
}
# 47 "/usr/include/c++/12/tr1/riemann_zeta.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 63 "/usr/include/c++/12/tr1/riemann_zeta.tcc" 3
namespace __detail { 
# 78 "/usr/include/c++/12/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 80
__riemann_zeta_sum(_Tp __s) 
# 81
{ 
# 83
if (__s < ((_Tp)1)) { 
# 84
std::__throw_domain_error("Bad argument in zeta sum."); }  
# 86
const unsigned max_iter = (10000); 
# 87
_Tp __zeta = ((_Tp)0); 
# 88
for (unsigned __k = (1); __k < max_iter; ++__k) 
# 89
{ 
# 90
_Tp __term = std::pow(static_cast< _Tp>(__k), -__s); 
# 91
if (__term < std::template numeric_limits< _Tp> ::epsilon()) 
# 92
{ 
# 93
break; 
# 94
}  
# 95
__zeta += __term; 
# 96
}  
# 98
return __zeta; 
# 99
} 
# 115 "/usr/include/c++/12/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 117
__riemann_zeta_alt(_Tp __s) 
# 118
{ 
# 119
_Tp __sgn = ((_Tp)1); 
# 120
_Tp __zeta = ((_Tp)0); 
# 121
for (unsigned __i = (1); __i < (10000000); ++__i) 
# 122
{ 
# 123
_Tp __term = __sgn / std::pow(__i, __s); 
# 124
if (std::abs(__term) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 125
break; }  
# 126
__zeta += __term; 
# 127
__sgn *= ((_Tp)(-1)); 
# 128
}  
# 129
__zeta /= (((_Tp)1) - std::pow((_Tp)2, ((_Tp)1) - __s)); 
# 131
return __zeta; 
# 132
} 
# 157 "/usr/include/c++/12/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 159
__riemann_zeta_glob(_Tp __s) 
# 160
{ 
# 161
_Tp __zeta = ((_Tp)0); 
# 163
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 165
const _Tp __max_bincoeff = (std::template numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 170
if (__s < ((_Tp)0)) 
# 171
{ 
# 173
if (std::fmod(__s, (_Tp)2) == ((_Tp)0)) { 
# 174
return (_Tp)0; } else 
# 177
{ 
# 178
_Tp __zeta = __riemann_zeta_glob(((_Tp)1) - __s); 
# 179
__zeta *= (((std::pow(((_Tp)2) * __numeric_constants< _Tp> ::__pi(), __s) * std::sin(__numeric_constants< _Tp> ::__pi_2() * __s)) * std::exp(std::lgamma(((_Tp)1) - __s))) / __numeric_constants< _Tp> ::__pi()); 
# 188
return __zeta; 
# 189
}  
# 190
}  
# 192
_Tp __num = ((_Tp)(0.5L)); 
# 193
const unsigned __maxit = (10000); 
# 194
for (unsigned __i = (0); __i < __maxit; ++__i) 
# 195
{ 
# 196
bool __punt = false; 
# 197
_Tp __sgn = ((_Tp)1); 
# 198
_Tp __term = ((_Tp)0); 
# 199
for (unsigned __j = (0); __j <= __i; ++__j) 
# 200
{ 
# 202
_Tp __bincoeff = (std::lgamma((_Tp)((1) + __i)) - std::lgamma((_Tp)((1) + __j))) - std::lgamma((_Tp)(((1) + __i) - __j)); 
# 210
if (__bincoeff > __max_bincoeff) 
# 211
{ 
# 213
__punt = true; 
# 214
break; 
# 215
}  
# 216
__bincoeff = std::exp(__bincoeff); 
# 217
__term += ((__sgn * __bincoeff) * std::pow((_Tp)((1) + __j), -__s)); 
# 218
__sgn *= ((_Tp)(-1)); 
# 219
}  
# 220
if (__punt) { 
# 221
break; }  
# 222
__term *= __num; 
# 223
__zeta += __term; 
# 224
if (std::abs(__term / __zeta) < __eps) { 
# 225
break; }  
# 226
__num *= ((_Tp)(0.5L)); 
# 227
}  
# 229
__zeta /= (((_Tp)1) - std::pow((_Tp)2, ((_Tp)1) - __s)); 
# 231
return __zeta; 
# 232
} 
# 252 "/usr/include/c++/12/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 254
__riemann_zeta_product(_Tp __s) 
# 255
{ 
# 256
static const _Tp __prime[] = {((_Tp)2), ((_Tp)3), ((_Tp)5), ((_Tp)7), ((_Tp)11), ((_Tp)13), ((_Tp)17), ((_Tp)19), ((_Tp)23), ((_Tp)29), ((_Tp)31), ((_Tp)37), ((_Tp)41), ((_Tp)43), ((_Tp)47), ((_Tp)53), ((_Tp)59), ((_Tp)61), ((_Tp)67), ((_Tp)71), ((_Tp)73), ((_Tp)79), ((_Tp)83), ((_Tp)89), ((_Tp)97), ((_Tp)101), ((_Tp)103), ((_Tp)107), ((_Tp)109)}; 
# 262
static const unsigned __num_primes = (sizeof(__prime) / sizeof(_Tp)); 
# 264
_Tp __zeta = ((_Tp)1); 
# 265
for (unsigned __i = (0); __i < __num_primes; ++__i) 
# 266
{ 
# 267
const _Tp __fact = ((_Tp)1) - std::pow(__prime[__i], -__s); 
# 268
__zeta *= __fact; 
# 269
if ((((_Tp)1) - __fact) < std::template numeric_limits< _Tp> ::epsilon()) { 
# 270
break; }  
# 271
}  
# 273
__zeta = (((_Tp)1) / __zeta); 
# 275
return __zeta; 
# 276
} 
# 293 "/usr/include/c++/12/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 295
__riemann_zeta(_Tp __s) 
# 296
{ 
# 297
if (__isnan(__s)) { 
# 298
return std::template numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 299
if (__s == ((_Tp)1)) { 
# 300
return std::template numeric_limits< _Tp> ::infinity(); } else { 
# 301
if (__s < (-((_Tp)19))) 
# 302
{ 
# 303
_Tp __zeta = __riemann_zeta_product(((_Tp)1) - __s); 
# 304
__zeta *= (((std::pow(((_Tp)2) * __numeric_constants< _Tp> ::__pi(), __s) * std::sin(__numeric_constants< _Tp> ::__pi_2() * __s)) * std::exp(std::lgamma(((_Tp)1) - __s))) / __numeric_constants< _Tp> ::__pi()); 
# 312
return __zeta; 
# 313
} else { 
# 314
if (__s < ((_Tp)20)) 
# 315
{ 
# 317
bool __glob = true; 
# 318
if (__glob) { 
# 319
return __riemann_zeta_glob(__s); } else 
# 321
{ 
# 322
if (__s > ((_Tp)1)) { 
# 323
return __riemann_zeta_sum(__s); } else 
# 325
{ 
# 326
_Tp __zeta = ((std::pow(((_Tp)2) * __numeric_constants< _Tp> ::__pi(), __s) * std::sin(__numeric_constants< _Tp> ::__pi_2() * __s)) * std::tgamma(((_Tp)1) - __s)) * __riemann_zeta_sum(((_Tp)1) - __s); 
# 335
return __zeta; 
# 336
}  
# 337
}  
# 338
} else { 
# 340
return __riemann_zeta_product(__s); }  }  }  }  
# 341
} 
# 365 "/usr/include/c++/12/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 367
__hurwitz_zeta_glob(_Tp __a, _Tp __s) 
# 368
{ 
# 369
_Tp __zeta = ((_Tp)0); 
# 371
const _Tp __eps = std::template numeric_limits< _Tp> ::epsilon(); 
# 373
const _Tp __max_bincoeff = (std::template numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 376
const unsigned __maxit = (10000); 
# 377
for (unsigned __i = (0); __i < __maxit; ++__i) 
# 378
{ 
# 379
bool __punt = false; 
# 380
_Tp __sgn = ((_Tp)1); 
# 381
_Tp __term = ((_Tp)0); 
# 382
for (unsigned __j = (0); __j <= __i; ++__j) 
# 383
{ 
# 385
_Tp __bincoeff = (std::lgamma((_Tp)((1) + __i)) - std::lgamma((_Tp)((1) + __j))) - std::lgamma((_Tp)(((1) + __i) - __j)); 
# 393
if (__bincoeff > __max_bincoeff) 
# 394
{ 
# 396
__punt = true; 
# 397
break; 
# 398
}  
# 399
__bincoeff = std::exp(__bincoeff); 
# 400
__term += ((__sgn * __bincoeff) * std::pow((_Tp)(__a + __j), -__s)); 
# 401
__sgn *= ((_Tp)(-1)); 
# 402
}  
# 403
if (__punt) { 
# 404
break; }  
# 405
__term /= ((_Tp)(__i + (1))); 
# 406
if (std::abs(__term / __zeta) < __eps) { 
# 407
break; }  
# 408
__zeta += __term; 
# 409
}  
# 411
__zeta /= (__s - ((_Tp)1)); 
# 413
return __zeta; 
# 414
} 
# 430 "/usr/include/c++/12/tr1/riemann_zeta.tcc" 3
template< class _Tp> inline _Tp 
# 432
__hurwitz_zeta(_Tp __a, _Tp __s) 
# 433
{ return __hurwitz_zeta_glob(__a, __s); } 
# 434
}
# 441
}
# 61 "/usr/include/c++/12/bits/specfun.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 206 "/usr/include/c++/12/bits/specfun.h" 3
inline float assoc_laguerref(unsigned __n, unsigned __m, float __x) 
# 207
{ return __detail::__assoc_laguerre< float> (__n, __m, __x); } 
# 216
inline long double assoc_laguerrel(unsigned __n, unsigned __m, long double __x) 
# 217
{ return __detail::__assoc_laguerre< long double> (__n, __m, __x); } 
# 250 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 252
assoc_laguerre(unsigned __n, unsigned __m, _Tp __x) 
# 253
{ 
# 254
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 255
return __detail::__assoc_laguerre< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __m, __x); 
# 256
} 
# 267 "/usr/include/c++/12/bits/specfun.h" 3
inline float assoc_legendref(unsigned __l, unsigned __m, float __x) 
# 268
{ return __detail::__assoc_legendre_p< float> (__l, __m, __x); } 
# 276
inline long double assoc_legendrel(unsigned __l, unsigned __m, long double __x) 
# 277
{ return __detail::__assoc_legendre_p< long double> (__l, __m, __x); } 
# 296 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 298
assoc_legendre(unsigned __l, unsigned __m, _Tp __x) 
# 299
{ 
# 300
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 301
return __detail::__assoc_legendre_p< typename __gnu_cxx::__promote< _Tp> ::__type> (__l, __m, __x); 
# 302
} 
# 312 "/usr/include/c++/12/bits/specfun.h" 3
inline float betaf(float __a, float __b) 
# 313
{ return __detail::__beta< float> (__a, __b); } 
# 322
inline long double betal(long double __a, long double __b) 
# 323
{ return __detail::__beta< long double> (__a, __b); } 
# 341 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tpa, class _Tpb> inline typename __gnu_cxx::__promote_2< _Tpa, _Tpb> ::__type 
# 343
beta(_Tpa __a, _Tpb __b) 
# 344
{ 
# 345
typedef typename __gnu_cxx::__promote_2< _Tpa, _Tpb> ::__type __type; 
# 346
return __detail::__beta< typename __gnu_cxx::__promote_2< _Tpa, _Tpb> ::__type> (__a, __b); 
# 347
} 
# 358 "/usr/include/c++/12/bits/specfun.h" 3
inline float comp_ellint_1f(float __k) 
# 359
{ return __detail::__comp_ellint_1< float> (__k); } 
# 368
inline long double comp_ellint_1l(long double __k) 
# 369
{ return __detail::__comp_ellint_1< long double> (__k); } 
# 389 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 391
comp_ellint_1(_Tp __k) 
# 392
{ 
# 393
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 394
return __detail::__comp_ellint_1< typename __gnu_cxx::__promote< _Tp> ::__type> (__k); 
# 395
} 
# 406 "/usr/include/c++/12/bits/specfun.h" 3
inline float comp_ellint_2f(float __k) 
# 407
{ return __detail::__comp_ellint_2< float> (__k); } 
# 416
inline long double comp_ellint_2l(long double __k) 
# 417
{ return __detail::__comp_ellint_2< long double> (__k); } 
# 436 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 438
comp_ellint_2(_Tp __k) 
# 439
{ 
# 440
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 441
return __detail::__comp_ellint_2< typename __gnu_cxx::__promote< _Tp> ::__type> (__k); 
# 442
} 
# 453 "/usr/include/c++/12/bits/specfun.h" 3
inline float comp_ellint_3f(float __k, float __nu) 
# 454
{ return __detail::__comp_ellint_3< float> (__k, __nu); } 
# 463
inline long double comp_ellint_3l(long double __k, long double __nu) 
# 464
{ return __detail::__comp_ellint_3< long double> (__k, __nu); } 
# 487 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp, class _Tpn> inline typename __gnu_cxx::__promote_2< _Tp, _Tpn> ::__type 
# 489
comp_ellint_3(_Tp __k, _Tpn __nu) 
# 490
{ 
# 491
typedef typename __gnu_cxx::__promote_2< _Tp, _Tpn> ::__type __type; 
# 492
return __detail::__comp_ellint_3< typename __gnu_cxx::__promote_2< _Tp, _Tpn> ::__type> (__k, __nu); 
# 493
} 
# 504 "/usr/include/c++/12/bits/specfun.h" 3
inline float cyl_bessel_if(float __nu, float __x) 
# 505
{ return __detail::__cyl_bessel_i< float> (__nu, __x); } 
# 514
inline long double cyl_bessel_il(long double __nu, long double __x) 
# 515
{ return __detail::__cyl_bessel_i< long double> (__nu, __x); } 
# 533 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 535
cyl_bessel_i(_Tpnu __nu, _Tp __x) 
# 536
{ 
# 537
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 538
return __detail::__cyl_bessel_i< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 539
} 
# 550 "/usr/include/c++/12/bits/specfun.h" 3
inline float cyl_bessel_jf(float __nu, float __x) 
# 551
{ return __detail::__cyl_bessel_j< float> (__nu, __x); } 
# 560
inline long double cyl_bessel_jl(long double __nu, long double __x) 
# 561
{ return __detail::__cyl_bessel_j< long double> (__nu, __x); } 
# 579 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 581
cyl_bessel_j(_Tpnu __nu, _Tp __x) 
# 582
{ 
# 583
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 584
return __detail::__cyl_bessel_j< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 585
} 
# 596 "/usr/include/c++/12/bits/specfun.h" 3
inline float cyl_bessel_kf(float __nu, float __x) 
# 597
{ return __detail::__cyl_bessel_k< float> (__nu, __x); } 
# 606
inline long double cyl_bessel_kl(long double __nu, long double __x) 
# 607
{ return __detail::__cyl_bessel_k< long double> (__nu, __x); } 
# 631 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 633
cyl_bessel_k(_Tpnu __nu, _Tp __x) 
# 634
{ 
# 635
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 636
return __detail::__cyl_bessel_k< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 637
} 
# 648 "/usr/include/c++/12/bits/specfun.h" 3
inline float cyl_neumannf(float __nu, float __x) 
# 649
{ return __detail::__cyl_neumann_n< float> (__nu, __x); } 
# 658
inline long double cyl_neumannl(long double __nu, long double __x) 
# 659
{ return __detail::__cyl_neumann_n< long double> (__nu, __x); } 
# 679 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 681
cyl_neumann(_Tpnu __nu, _Tp __x) 
# 682
{ 
# 683
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 684
return __detail::__cyl_neumann_n< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 685
} 
# 696 "/usr/include/c++/12/bits/specfun.h" 3
inline float ellint_1f(float __k, float __phi) 
# 697
{ return __detail::__ellint_1< float> (__k, __phi); } 
# 706
inline long double ellint_1l(long double __k, long double __phi) 
# 707
{ return __detail::__ellint_1< long double> (__k, __phi); } 
# 727 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp, class _Tpp> inline typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type 
# 729
ellint_1(_Tp __k, _Tpp __phi) 
# 730
{ 
# 731
typedef typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type __type; 
# 732
return __detail::__ellint_1< typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type> (__k, __phi); 
# 733
} 
# 744 "/usr/include/c++/12/bits/specfun.h" 3
inline float ellint_2f(float __k, float __phi) 
# 745
{ return __detail::__ellint_2< float> (__k, __phi); } 
# 754
inline long double ellint_2l(long double __k, long double __phi) 
# 755
{ return __detail::__ellint_2< long double> (__k, __phi); } 
# 775 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp, class _Tpp> inline typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type 
# 777
ellint_2(_Tp __k, _Tpp __phi) 
# 778
{ 
# 779
typedef typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type __type; 
# 780
return __detail::__ellint_2< typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type> (__k, __phi); 
# 781
} 
# 792 "/usr/include/c++/12/bits/specfun.h" 3
inline float ellint_3f(float __k, float __nu, float __phi) 
# 793
{ return __detail::__ellint_3< float> (__k, __nu, __phi); } 
# 802
inline long double ellint_3l(long double __k, long double __nu, long double __phi) 
# 803
{ return __detail::__ellint_3< long double> (__k, __nu, __phi); } 
# 828 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp, class _Tpn, class _Tpp> inline typename __gnu_cxx::__promote_3< _Tp, _Tpn, _Tpp> ::__type 
# 830
ellint_3(_Tp __k, _Tpn __nu, _Tpp __phi) 
# 831
{ 
# 832
typedef typename __gnu_cxx::__promote_3< _Tp, _Tpn, _Tpp> ::__type __type; 
# 833
return __detail::__ellint_3< typename __gnu_cxx::__promote_3< _Tp, _Tpn, _Tpp> ::__type> (__k, __nu, __phi); 
# 834
} 
# 844 "/usr/include/c++/12/bits/specfun.h" 3
inline float expintf(float __x) 
# 845
{ return __detail::__expint< float> (__x); } 
# 854
inline long double expintl(long double __x) 
# 855
{ return __detail::__expint< long double> (__x); } 
# 868 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 870
expint(_Tp __x) 
# 871
{ 
# 872
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 873
return __detail::__expint< typename __gnu_cxx::__promote< _Tp> ::__type> (__x); 
# 874
} 
# 885 "/usr/include/c++/12/bits/specfun.h" 3
inline float hermitef(unsigned __n, float __x) 
# 886
{ return __detail::__poly_hermite< float> (__n, __x); } 
# 895
inline long double hermitel(unsigned __n, long double __x) 
# 896
{ return __detail::__poly_hermite< long double> (__n, __x); } 
# 916 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 918
hermite(unsigned __n, _Tp __x) 
# 919
{ 
# 920
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 921
return __detail::__poly_hermite< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 922
} 
# 933 "/usr/include/c++/12/bits/specfun.h" 3
inline float laguerref(unsigned __n, float __x) 
# 934
{ return __detail::__laguerre< float> (__n, __x); } 
# 943
inline long double laguerrel(unsigned __n, long double __x) 
# 944
{ return __detail::__laguerre< long double> (__n, __x); } 
# 960 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 962
laguerre(unsigned __n, _Tp __x) 
# 963
{ 
# 964
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 965
return __detail::__laguerre< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 966
} 
# 977 "/usr/include/c++/12/bits/specfun.h" 3
inline float legendref(unsigned __l, float __x) 
# 978
{ return __detail::__poly_legendre_p< float> (__l, __x); } 
# 987
inline long double legendrel(unsigned __l, long double __x) 
# 988
{ return __detail::__poly_legendre_p< long double> (__l, __x); } 
# 1005 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1007
legendre(unsigned __l, _Tp __x) 
# 1008
{ 
# 1009
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1010
return __detail::__poly_legendre_p< typename __gnu_cxx::__promote< _Tp> ::__type> (__l, __x); 
# 1011
} 
# 1022 "/usr/include/c++/12/bits/specfun.h" 3
inline float riemann_zetaf(float __s) 
# 1023
{ return __detail::__riemann_zeta< float> (__s); } 
# 1032
inline long double riemann_zetal(long double __s) 
# 1033
{ return __detail::__riemann_zeta< long double> (__s); } 
# 1056 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1058
riemann_zeta(_Tp __s) 
# 1059
{ 
# 1060
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1061
return __detail::__riemann_zeta< typename __gnu_cxx::__promote< _Tp> ::__type> (__s); 
# 1062
} 
# 1073 "/usr/include/c++/12/bits/specfun.h" 3
inline float sph_besself(unsigned __n, float __x) 
# 1074
{ return __detail::__sph_bessel< float> (__n, __x); } 
# 1083
inline long double sph_bessell(unsigned __n, long double __x) 
# 1084
{ return __detail::__sph_bessel< long double> (__n, __x); } 
# 1100 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1102
sph_bessel(unsigned __n, _Tp __x) 
# 1103
{ 
# 1104
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1105
return __detail::__sph_bessel< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 1106
} 
# 1117 "/usr/include/c++/12/bits/specfun.h" 3
inline float sph_legendref(unsigned __l, unsigned __m, float __theta) 
# 1118
{ return __detail::__sph_legendre< float> (__l, __m, __theta); } 
# 1128 "/usr/include/c++/12/bits/specfun.h" 3
inline long double sph_legendrel(unsigned __l, unsigned __m, long double __theta) 
# 1129
{ return __detail::__sph_legendre< long double> (__l, __m, __theta); } 
# 1147 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1149
sph_legendre(unsigned __l, unsigned __m, _Tp __theta) 
# 1150
{ 
# 1151
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1152
return __detail::__sph_legendre< typename __gnu_cxx::__promote< _Tp> ::__type> (__l, __m, __theta); 
# 1153
} 
# 1164 "/usr/include/c++/12/bits/specfun.h" 3
inline float sph_neumannf(unsigned __n, float __x) 
# 1165
{ return __detail::__sph_neumann< float> (__n, __x); } 
# 1174
inline long double sph_neumannl(unsigned __n, long double __x) 
# 1175
{ return __detail::__sph_neumann< long double> (__n, __x); } 
# 1191 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1193
sph_neumann(unsigned __n, _Tp __x) 
# 1194
{ 
# 1195
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1196
return __detail::__sph_neumann< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 1197
} 
# 1202
}
# 1205
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 1219 "/usr/include/c++/12/bits/specfun.h" 3
inline float airy_aif(float __x) 
# 1220
{ 
# 1221
float __Ai, __Bi, __Aip, __Bip; 
# 1222
std::__detail::__airy< float> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1223
return __Ai; 
# 1224
} 
# 1230
inline long double airy_ail(long double __x) 
# 1231
{ 
# 1232
long double __Ai, __Bi, __Aip, __Bip; 
# 1233
std::__detail::__airy< long double> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1234
return __Ai; 
# 1235
} 
# 1240
template< class _Tp> inline typename __promote< _Tp> ::__type 
# 1242
airy_ai(_Tp __x) 
# 1243
{ 
# 1244
typedef typename __promote< _Tp> ::__type __type; 
# 1245
__type __Ai, __Bi, __Aip, __Bip; 
# 1246
std::__detail::__airy< typename __promote< _Tp> ::__type> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1247
return __Ai; 
# 1248
} 
# 1254
inline float airy_bif(float __x) 
# 1255
{ 
# 1256
float __Ai, __Bi, __Aip, __Bip; 
# 1257
std::__detail::__airy< float> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1258
return __Bi; 
# 1259
} 
# 1265
inline long double airy_bil(long double __x) 
# 1266
{ 
# 1267
long double __Ai, __Bi, __Aip, __Bip; 
# 1268
std::__detail::__airy< long double> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1269
return __Bi; 
# 1270
} 
# 1275
template< class _Tp> inline typename __promote< _Tp> ::__type 
# 1277
airy_bi(_Tp __x) 
# 1278
{ 
# 1279
typedef typename __promote< _Tp> ::__type __type; 
# 1280
__type __Ai, __Bi, __Aip, __Bip; 
# 1281
std::__detail::__airy< typename __promote< _Tp> ::__type> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1282
return __Bi; 
# 1283
} 
# 1295 "/usr/include/c++/12/bits/specfun.h" 3
inline float conf_hypergf(float __a, float __c, float __x) 
# 1296
{ return std::__detail::__conf_hyperg< float> (__a, __c, __x); } 
# 1306 "/usr/include/c++/12/bits/specfun.h" 3
inline long double conf_hypergl(long double __a, long double __c, long double __x) 
# 1307
{ return std::__detail::__conf_hyperg< long double> (__a, __c, __x); } 
# 1325 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tpa, class _Tpc, class _Tp> inline typename __promote_3< _Tpa, _Tpc, _Tp> ::__type 
# 1327
conf_hyperg(_Tpa __a, _Tpc __c, _Tp __x) 
# 1328
{ 
# 1329
typedef typename __promote_3< _Tpa, _Tpc, _Tp> ::__type __type; 
# 1330
return std::__detail::__conf_hyperg< typename __promote_3< _Tpa, _Tpc, _Tp> ::__type> (__a, __c, __x); 
# 1331
} 
# 1343 "/usr/include/c++/12/bits/specfun.h" 3
inline float hypergf(float __a, float __b, float __c, float __x) 
# 1344
{ return std::__detail::__hyperg< float> (__a, __b, __c, __x); } 
# 1354 "/usr/include/c++/12/bits/specfun.h" 3
inline long double hypergl(long double __a, long double __b, long double __c, long double __x) 
# 1355
{ return std::__detail::__hyperg< long double> (__a, __b, __c, __x); } 
# 1374 "/usr/include/c++/12/bits/specfun.h" 3
template< class _Tpa, class _Tpb, class _Tpc, class _Tp> inline typename __promote_4< _Tpa, _Tpb, _Tpc, _Tp> ::__type 
# 1376
hyperg(_Tpa __a, _Tpb __b, _Tpc __c, _Tp __x) 
# 1377
{ 
# 1379
typedef typename __promote_4< _Tpa, _Tpb, _Tpc, _Tp> ::__type __type; 
# 1380
return std::__detail::__hyperg< typename __promote_4< _Tpa, _Tpb, _Tpc, _Tp> ::__type> (__a, __b, __c, __x); 
# 1381
} 
# 1385
}
# 1388
#pragma GCC visibility pop
# 1938 "/usr/include/c++/12/cmath" 3
}
# 38 "/usr/include/c++/12/math.h" 3
using std::abs;
# 39
using std::acos;
# 40
using std::asin;
# 41
using std::atan;
# 42
using std::atan2;
# 43
using std::cos;
# 44
using std::sin;
# 45
using std::tan;
# 46
using std::cosh;
# 47
using std::sinh;
# 48
using std::tanh;
# 49
using std::exp;
# 50
using std::frexp;
# 51
using std::ldexp;
# 52
using std::log;
# 53
using std::log10;
# 54
using std::modf;
# 55
using std::pow;
# 56
using std::sqrt;
# 57
using std::ceil;
# 58
using std::fabs;
# 59
using std::floor;
# 60
using std::fmod;
# 63
using std::fpclassify;
# 64
using std::isfinite;
# 65
using std::isinf;
# 66
using std::isnan;
# 67
using std::isnormal;
# 68
using std::signbit;
# 69
using std::isgreater;
# 70
using std::isgreaterequal;
# 71
using std::isless;
# 72
using std::islessequal;
# 73
using std::islessgreater;
# 74
using std::isunordered;
# 78
using std::acosh;
# 79
using std::asinh;
# 80
using std::atanh;
# 81
using std::cbrt;
# 82
using std::copysign;
# 83
using std::erf;
# 84
using std::erfc;
# 85
using std::exp2;
# 86
using std::expm1;
# 87
using std::fdim;
# 88
using std::fma;
# 89
using std::fmax;
# 90
using std::fmin;
# 91
using std::hypot;
# 92
using std::ilogb;
# 93
using std::lgamma;
# 94
using std::llrint;
# 95
using std::llround;
# 96
using std::log1p;
# 97
using std::log2;
# 98
using std::logb;
# 99
using std::lrint;
# 100
using std::lround;
# 101
using std::nearbyint;
# 102
using std::nextafter;
# 103
using std::nexttoward;
# 104
using std::remainder;
# 105
using std::remquo;
# 106
using std::rint;
# 107
using std::round;
# 108
using std::scalbln;
# 109
using std::scalbn;
# 110
using std::tgamma;
# 111
using std::trunc;
# 10623 "/usr/include/crt/math_functions.h" 3
namespace std { 
# 10624
constexpr bool signbit(float x); 
# 10625
constexpr bool signbit(double x); 
# 10626
constexpr bool signbit(long double x); 
# 10627
constexpr bool isfinite(float x); 
# 10628
constexpr bool isfinite(double x); 
# 10629
constexpr bool isfinite(long double x); 
# 10630
constexpr bool isnan(float x); 
# 10635
constexpr bool isnan(double x); 
# 10637
constexpr bool isnan(long double x); 
# 10638
constexpr bool isinf(float x); 
# 10643
constexpr bool isinf(double x); 
# 10645
constexpr bool isinf(long double x); 
# 10646
}
# 10800 "/usr/include/crt/math_functions.h" 3
namespace std { 
# 10802
template< class T> extern T __pow_helper(T, int); 
# 10803
template< class T> extern T __cmath_power(T, unsigned); 
# 10804
}
# 10806
using std::abs;
# 10807
using std::fabs;
# 10808
using std::ceil;
# 10809
using std::floor;
# 10810
using std::sqrt;
# 10812
using std::pow;
# 10814
using std::log;
# 10815
using std::log10;
# 10816
using std::fmod;
# 10817
using std::modf;
# 10818
using std::exp;
# 10819
using std::frexp;
# 10820
using std::ldexp;
# 10821
using std::asin;
# 10822
using std::sin;
# 10823
using std::sinh;
# 10824
using std::acos;
# 10825
using std::cos;
# 10826
using std::cosh;
# 10827
using std::atan;
# 10828
using std::atan2;
# 10829
using std::tan;
# 10830
using std::tanh;
# 11201 "/usr/include/crt/math_functions.h" 3
namespace std { 
# 11210 "/usr/include/crt/math_functions.h" 3
extern inline long long abs(long long); 
# 11220 "/usr/include/crt/math_functions.h" 3
extern inline long abs(long); 
# 11221
extern constexpr float abs(float); 
# 11222
extern constexpr double abs(double); 
# 11223
extern constexpr float fabs(float); 
# 11224
extern constexpr float ceil(float); 
# 11225
extern constexpr float floor(float); 
# 11226
extern constexpr float sqrt(float); 
# 11227
extern constexpr float pow(float, float); 
# 11232
template< class _Tp, class _Up> extern constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type pow(_Tp, _Up); 
# 11242
extern constexpr float log(float); 
# 11243
extern constexpr float log10(float); 
# 11244
extern constexpr float fmod(float, float); 
# 11245
extern inline float modf(float, float *); 
# 11246
extern constexpr float exp(float); 
# 11247
extern inline float frexp(float, int *); 
# 11248
extern constexpr float ldexp(float, int); 
# 11249
extern constexpr float asin(float); 
# 11250
extern constexpr float sin(float); 
# 11251
extern constexpr float sinh(float); 
# 11252
extern constexpr float acos(float); 
# 11253
extern constexpr float cos(float); 
# 11254
extern constexpr float cosh(float); 
# 11255
extern constexpr float atan(float); 
# 11256
extern constexpr float atan2(float, float); 
# 11257
extern constexpr float tan(float); 
# 11258
extern constexpr float tanh(float); 
# 11337 "/usr/include/crt/math_functions.h" 3
}
# 11443 "/usr/include/crt/math_functions.h" 3
namespace std { 
# 11444
constexpr float logb(float a); 
# 11445
constexpr int ilogb(float a); 
# 11446
constexpr float scalbn(float a, int b); 
# 11447
constexpr float scalbln(float a, long b); 
# 11448
constexpr float exp2(float a); 
# 11449
constexpr float expm1(float a); 
# 11450
constexpr float log2(float a); 
# 11451
constexpr float log1p(float a); 
# 11452
constexpr float acosh(float a); 
# 11453
constexpr float asinh(float a); 
# 11454
constexpr float atanh(float a); 
# 11455
constexpr float hypot(float a, float b); 
# 11456
constexpr float cbrt(float a); 
# 11457
constexpr float erf(float a); 
# 11458
constexpr float erfc(float a); 
# 11459
constexpr float lgamma(float a); 
# 11460
constexpr float tgamma(float a); 
# 11461
constexpr float copysign(float a, float b); 
# 11462
constexpr float nextafter(float a, float b); 
# 11463
constexpr float remainder(float a, float b); 
# 11464
inline float remquo(float a, float b, int * quo); 
# 11465
constexpr float round(float a); 
# 11466
constexpr long lround(float a); 
# 11467
constexpr long long llround(float a); 
# 11468
constexpr float trunc(float a); 
# 11469
constexpr float rint(float a); 
# 11470
constexpr long lrint(float a); 
# 11471
constexpr long long llrint(float a); 
# 11472
constexpr float nearbyint(float a); 
# 11473
constexpr float fdim(float a, float b); 
# 11474
constexpr float fma(float a, float b, float c); 
# 11475
constexpr float fmax(float a, float b); 
# 11476
constexpr float fmin(float a, float b); 
# 11477
}
# 11582 "/usr/include/crt/math_functions.h" 3
static inline float exp10(const float a); 
# 11584
static inline float rsqrt(const float a); 
# 11586
static inline float rcbrt(const float a); 
# 11588
static inline float sinpi(const float a); 
# 11590
static inline float cospi(const float a); 
# 11592
static inline void sincospi(const float a, float *const sptr, float *const cptr); 
# 11594
static inline void sincos(const float a, float *const sptr, float *const cptr); 
# 11596
static inline float j0(const float a); 
# 11598
static inline float j1(const float a); 
# 11600
static inline float jn(const int n, const float a); 
# 11602
static inline float y0(const float a); 
# 11604
static inline float y1(const float a); 
# 11606
static inline float yn(const int n, const float a); 
# 11608
__attribute__((unused)) static inline float cyl_bessel_i0(const float a); 
# 11610
__attribute__((unused)) static inline float cyl_bessel_i1(const float a); 
# 11612
static inline float erfinv(const float a); 
# 11614
static inline float erfcinv(const float a); 
# 11616
static inline float normcdfinv(const float a); 
# 11618
static inline float normcdf(const float a); 
# 11620
static inline float erfcx(const float a); 
# 11622
static inline double copysign(const double a, const float b); 
# 11624
static inline double copysign(const float a, const double b); 
# 11632
static inline unsigned min(const unsigned a, const unsigned b); 
# 11640
static inline unsigned min(const int a, const unsigned b); 
# 11648
static inline unsigned min(const unsigned a, const int b); 
# 11656
static inline long min(const long a, const long b); 
# 11664
static inline unsigned long min(const unsigned long a, const unsigned long b); 
# 11672
static inline unsigned long min(const long a, const unsigned long b); 
# 11680
static inline unsigned long min(const unsigned long a, const long b); 
# 11688
static inline long long min(const long long a, const long long b); 
# 11696
static inline unsigned long long min(const unsigned long long a, const unsigned long long b); 
# 11704
static inline unsigned long long min(const long long a, const unsigned long long b); 
# 11712
static inline unsigned long long min(const unsigned long long a, const long long b); 
# 11723 "/usr/include/crt/math_functions.h" 3
static inline float min(const float a, const float b); 
# 11734 "/usr/include/crt/math_functions.h" 3
static inline double min(const double a, const double b); 
# 11744 "/usr/include/crt/math_functions.h" 3
static inline double min(const float a, const double b); 
# 11754 "/usr/include/crt/math_functions.h" 3
static inline double min(const double a, const float b); 
# 11762
static inline unsigned max(const unsigned a, const unsigned b); 
# 11770
static inline unsigned max(const int a, const unsigned b); 
# 11778
static inline unsigned max(const unsigned a, const int b); 
# 11786
static inline long max(const long a, const long b); 
# 11794
static inline unsigned long max(const unsigned long a, const unsigned long b); 
# 11802
static inline unsigned long max(const long a, const unsigned long b); 
# 11810
static inline unsigned long max(const unsigned long a, const long b); 
# 11818
static inline long long max(const long long a, const long long b); 
# 11826
static inline unsigned long long max(const unsigned long long a, const unsigned long long b); 
# 11834
static inline unsigned long long max(const long long a, const unsigned long long b); 
# 11842
static inline unsigned long long max(const unsigned long long a, const long long b); 
# 11853 "/usr/include/crt/math_functions.h" 3
static inline float max(const float a, const float b); 
# 11864 "/usr/include/crt/math_functions.h" 3
static inline double max(const double a, const double b); 
# 11874 "/usr/include/crt/math_functions.h" 3
static inline double max(const float a, const double b); 
# 11884 "/usr/include/crt/math_functions.h" 3
static inline double max(const double a, const float b); 
# 11895 "/usr/include/crt/math_functions.h" 3
extern "C" {
# 11896
__attribute__((unused)) inline void *__nv_aligned_device_malloc(size_t size, size_t align) 
# 11897
{int volatile ___ = 1;(void)size;(void)align;
# 11900
::exit(___);}
#if 0
# 11897
{ 
# 11898
__attribute__((unused)) void *__nv_aligned_device_malloc_impl(size_t, size_t); 
# 11899
return __nv_aligned_device_malloc_impl(size, align); 
# 11900
} 
#endif
# 11901 "/usr/include/crt/math_functions.h" 3
}
# 758 "/usr/include/crt/math_functions.hpp" 3
static inline float exp10(const float a) 
# 759
{ 
# 760
return exp10f(a); 
# 761
} 
# 763
static inline float rsqrt(const float a) 
# 764
{ 
# 765
return rsqrtf(a); 
# 766
} 
# 768
static inline float rcbrt(const float a) 
# 769
{ 
# 770
return rcbrtf(a); 
# 771
} 
# 773
static inline float sinpi(const float a) 
# 774
{ 
# 775
return sinpif(a); 
# 776
} 
# 778
static inline float cospi(const float a) 
# 779
{ 
# 780
return cospif(a); 
# 781
} 
# 783
static inline void sincospi(const float a, float *const sptr, float *const cptr) 
# 784
{ 
# 785
sincospif(a, sptr, cptr); 
# 786
} 
# 788
static inline void sincos(const float a, float *const sptr, float *const cptr) 
# 789
{ 
# 790
sincosf(a, sptr, cptr); 
# 791
} 
# 793
static inline float j0(const float a) 
# 794
{ 
# 795
return j0f(a); 
# 796
} 
# 798
static inline float j1(const float a) 
# 799
{ 
# 800
return j1f(a); 
# 801
} 
# 803
static inline float jn(const int n, const float a) 
# 804
{ 
# 805
return jnf(n, a); 
# 806
} 
# 808
static inline float y0(const float a) 
# 809
{ 
# 810
return y0f(a); 
# 811
} 
# 813
static inline float y1(const float a) 
# 814
{ 
# 815
return y1f(a); 
# 816
} 
# 818
static inline float yn(const int n, const float a) 
# 819
{ 
# 820
return ynf(n, a); 
# 821
} 
# 823
__attribute__((unused)) static inline float cyl_bessel_i0(const float a) 
# 824
{int volatile ___ = 1;(void)a;
# 826
::exit(___);}
#if 0
# 824
{ 
# 825
return cyl_bessel_i0f(a); 
# 826
} 
#endif
# 828 "/usr/include/crt/math_functions.hpp" 3
__attribute__((unused)) static inline float cyl_bessel_i1(const float a) 
# 829
{int volatile ___ = 1;(void)a;
# 831
::exit(___);}
#if 0
# 829
{ 
# 830
return cyl_bessel_i1f(a); 
# 831
} 
#endif
# 833 "/usr/include/crt/math_functions.hpp" 3
static inline float erfinv(const float a) 
# 834
{ 
# 835
return erfinvf(a); 
# 836
} 
# 838
static inline float erfcinv(const float a) 
# 839
{ 
# 840
return erfcinvf(a); 
# 841
} 
# 843
static inline float normcdfinv(const float a) 
# 844
{ 
# 845
return normcdfinvf(a); 
# 846
} 
# 848
static inline float normcdf(const float a) 
# 849
{ 
# 850
return normcdff(a); 
# 851
} 
# 853
static inline float erfcx(const float a) 
# 854
{ 
# 855
return erfcxf(a); 
# 856
} 
# 858
static inline double copysign(const double a, const float b) 
# 859
{ 
# 860
return copysign(a, static_cast< double>(b)); 
# 861
} 
# 863
static inline double copysign(const float a, const double b) 
# 864
{ 
# 865
return copysign(static_cast< double>(a), b); 
# 866
} 
# 868
static inline unsigned min(const unsigned a, const unsigned b) 
# 869
{ 
# 870
return umin(a, b); 
# 871
} 
# 873
static inline unsigned min(const int a, const unsigned b) 
# 874
{ 
# 875
return umin(static_cast< unsigned>(a), b); 
# 876
} 
# 878
static inline unsigned min(const unsigned a, const int b) 
# 879
{ 
# 880
return umin(a, static_cast< unsigned>(b)); 
# 881
} 
# 883
static inline long min(const long a, const long b) 
# 884
{ 
# 885
long retval; 
# 891
if (sizeof(long) == sizeof(int)) { 
# 895
retval = (static_cast< long>(min(static_cast< int>(a), static_cast< int>(b)))); 
# 896
} else { 
# 897
retval = (static_cast< long>(llmin(static_cast< long long>(a), static_cast< long long>(b)))); 
# 898
}  
# 899
return retval; 
# 900
} 
# 902
static inline unsigned long min(const unsigned long a, const unsigned long b) 
# 903
{ 
# 904
unsigned long retval; 
# 908
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 912
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 913
} else { 
# 914
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 915
}  
# 916
return retval; 
# 917
} 
# 919
static inline unsigned long min(const long a, const unsigned long b) 
# 920
{ 
# 921
unsigned long retval; 
# 925
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 929
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 930
} else { 
# 931
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 932
}  
# 933
return retval; 
# 934
} 
# 936
static inline unsigned long min(const unsigned long a, const long b) 
# 937
{ 
# 938
unsigned long retval; 
# 942
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 946
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 947
} else { 
# 948
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 949
}  
# 950
return retval; 
# 951
} 
# 953
static inline long long min(const long long a, const long long b) 
# 954
{ 
# 955
return llmin(a, b); 
# 956
} 
# 958
static inline unsigned long long min(const unsigned long long a, const unsigned long long b) 
# 959
{ 
# 960
return ullmin(a, b); 
# 961
} 
# 963
static inline unsigned long long min(const long long a, const unsigned long long b) 
# 964
{ 
# 965
return ullmin(static_cast< unsigned long long>(a), b); 
# 966
} 
# 968
static inline unsigned long long min(const unsigned long long a, const long long b) 
# 969
{ 
# 970
return ullmin(a, static_cast< unsigned long long>(b)); 
# 971
} 
# 973
static inline float min(const float a, const float b) 
# 974
{ 
# 975
return fminf(a, b); 
# 976
} 
# 978
static inline double min(const double a, const double b) 
# 979
{ 
# 980
return fmin(a, b); 
# 981
} 
# 983
static inline double min(const float a, const double b) 
# 984
{ 
# 985
return fmin(static_cast< double>(a), b); 
# 986
} 
# 988
static inline double min(const double a, const float b) 
# 989
{ 
# 990
return fmin(a, static_cast< double>(b)); 
# 991
} 
# 993
static inline unsigned max(const unsigned a, const unsigned b) 
# 994
{ 
# 995
return umax(a, b); 
# 996
} 
# 998
static inline unsigned max(const int a, const unsigned b) 
# 999
{ 
# 1000
return umax(static_cast< unsigned>(a), b); 
# 1001
} 
# 1003
static inline unsigned max(const unsigned a, const int b) 
# 1004
{ 
# 1005
return umax(a, static_cast< unsigned>(b)); 
# 1006
} 
# 1008
static inline long max(const long a, const long b) 
# 1009
{ 
# 1010
long retval; 
# 1015
if (sizeof(long) == sizeof(int)) { 
# 1019
retval = (static_cast< long>(max(static_cast< int>(a), static_cast< int>(b)))); 
# 1020
} else { 
# 1021
retval = (static_cast< long>(llmax(static_cast< long long>(a), static_cast< long long>(b)))); 
# 1022
}  
# 1023
return retval; 
# 1024
} 
# 1026
static inline unsigned long max(const unsigned long a, const unsigned long b) 
# 1027
{ 
# 1028
unsigned long retval; 
# 1032
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1036
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1037
} else { 
# 1038
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1039
}  
# 1040
return retval; 
# 1041
} 
# 1043
static inline unsigned long max(const long a, const unsigned long b) 
# 1044
{ 
# 1045
unsigned long retval; 
# 1049
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1053
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1054
} else { 
# 1055
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1056
}  
# 1057
return retval; 
# 1058
} 
# 1060
static inline unsigned long max(const unsigned long a, const long b) 
# 1061
{ 
# 1062
unsigned long retval; 
# 1066
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1070
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1071
} else { 
# 1072
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1073
}  
# 1074
return retval; 
# 1075
} 
# 1077
static inline long long max(const long long a, const long long b) 
# 1078
{ 
# 1079
return llmax(a, b); 
# 1080
} 
# 1082
static inline unsigned long long max(const unsigned long long a, const unsigned long long b) 
# 1083
{ 
# 1084
return ullmax(a, b); 
# 1085
} 
# 1087
static inline unsigned long long max(const long long a, const unsigned long long b) 
# 1088
{ 
# 1089
return ullmax(static_cast< unsigned long long>(a), b); 
# 1090
} 
# 1092
static inline unsigned long long max(const unsigned long long a, const long long b) 
# 1093
{ 
# 1094
return ullmax(a, static_cast< unsigned long long>(b)); 
# 1095
} 
# 1097
static inline float max(const float a, const float b) 
# 1098
{ 
# 1099
return fmaxf(a, b); 
# 1100
} 
# 1102
static inline double max(const double a, const double b) 
# 1103
{ 
# 1104
return fmax(a, b); 
# 1105
} 
# 1107
static inline double max(const float a, const double b) 
# 1108
{ 
# 1109
return fmax(static_cast< double>(a), b); 
# 1110
} 
# 1112
static inline double max(const double a, const float b) 
# 1113
{ 
# 1114
return fmax(a, static_cast< double>(b)); 
# 1115
} 
# 1126 "/usr/include/crt/math_functions.hpp" 3
inline int min(const int a, const int b) 
# 1127
{ 
# 1128
return (a < b) ? a : b; 
# 1129
} 
# 1131
inline unsigned umin(const unsigned a, const unsigned b) 
# 1132
{ 
# 1133
return (a < b) ? a : b; 
# 1134
} 
# 1136
inline long long llmin(const long long a, const long long b) 
# 1137
{ 
# 1138
return (a < b) ? a : b; 
# 1139
} 
# 1141
inline unsigned long long ullmin(const unsigned long long a, const unsigned long long 
# 1142
b) 
# 1143
{ 
# 1144
return (a < b) ? a : b; 
# 1145
} 
# 1147
inline int max(const int a, const int b) 
# 1148
{ 
# 1149
return (a > b) ? a : b; 
# 1150
} 
# 1152
inline unsigned umax(const unsigned a, const unsigned b) 
# 1153
{ 
# 1154
return (a > b) ? a : b; 
# 1155
} 
# 1157
inline long long llmax(const long long a, const long long b) 
# 1158
{ 
# 1159
return (a > b) ? a : b; 
# 1160
} 
# 1162
inline unsigned long long ullmax(const unsigned long long a, const unsigned long long 
# 1163
b) 
# 1164
{ 
# 1165
return (a > b) ? a : b; 
# 1166
} 
# 91 "/usr/include/crt/device_functions.h" 3
extern "C" {
# 3211 "/usr/include/crt/device_functions.h" 3
static inline int __vimax_s32_relu(const int a, const int b); 
# 3223 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vimax_s16x2_relu(const unsigned a, const unsigned b); 
# 3232 "/usr/include/crt/device_functions.h" 3
static inline int __vimin_s32_relu(const int a, const int b); 
# 3244 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vimin_s16x2_relu(const unsigned a, const unsigned b); 
# 3253 "/usr/include/crt/device_functions.h" 3
static inline int __vimax3_s32(const int a, const int b, const int c); 
# 3265 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vimax3_s16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3274 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vimax3_u32(const unsigned a, const unsigned b, const unsigned c); 
# 3286 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vimax3_u16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3295 "/usr/include/crt/device_functions.h" 3
static inline int __vimin3_s32(const int a, const int b, const int c); 
# 3307 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vimin3_s16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3316 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vimin3_u32(const unsigned a, const unsigned b, const unsigned c); 
# 3328 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vimin3_u16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3337 "/usr/include/crt/device_functions.h" 3
static inline int __vimax3_s32_relu(const int a, const int b, const int c); 
# 3349 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vimax3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
# 3358 "/usr/include/crt/device_functions.h" 3
static inline int __vimin3_s32_relu(const int a, const int b, const int c); 
# 3370 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vimin3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
# 3379 "/usr/include/crt/device_functions.h" 3
static inline int __viaddmax_s32(const int a, const int b, const int c); 
# 3391 "/usr/include/crt/device_functions.h" 3
static inline unsigned __viaddmax_s16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3400 "/usr/include/crt/device_functions.h" 3
static inline unsigned __viaddmax_u32(const unsigned a, const unsigned b, const unsigned c); 
# 3412 "/usr/include/crt/device_functions.h" 3
static inline unsigned __viaddmax_u16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3421 "/usr/include/crt/device_functions.h" 3
static inline int __viaddmin_s32(const int a, const int b, const int c); 
# 3433 "/usr/include/crt/device_functions.h" 3
static inline unsigned __viaddmin_s16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3442 "/usr/include/crt/device_functions.h" 3
static inline unsigned __viaddmin_u32(const unsigned a, const unsigned b, const unsigned c); 
# 3454 "/usr/include/crt/device_functions.h" 3
static inline unsigned __viaddmin_u16x2(const unsigned a, const unsigned b, const unsigned c); 
# 3464 "/usr/include/crt/device_functions.h" 3
static inline int __viaddmax_s32_relu(const int a, const int b, const int c); 
# 3476 "/usr/include/crt/device_functions.h" 3
static inline unsigned __viaddmax_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
# 3486 "/usr/include/crt/device_functions.h" 3
static inline int __viaddmin_s32_relu(const int a, const int b, const int c); 
# 3498 "/usr/include/crt/device_functions.h" 3
static inline unsigned __viaddmin_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
# 3507 "/usr/include/crt/device_functions.h" 3
static inline int __vibmax_s32(const int a, const int b, bool *const pred); 
# 3516 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vibmax_u32(const unsigned a, const unsigned b, bool *const pred); 
# 3525 "/usr/include/crt/device_functions.h" 3
static inline int __vibmin_s32(const int a, const int b, bool *const pred); 
# 3534 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vibmin_u32(const unsigned a, const unsigned b, bool *const pred); 
# 3548 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vibmax_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
# 3562 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vibmax_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
# 3576 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vibmin_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
# 3590 "/usr/include/crt/device_functions.h" 3
static inline unsigned __vibmin_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
# 3597
}
# 102 "/usr/include/crt/device_functions.hpp" 3
static inline int __vimax_s32_relu(const int a, const int b) { 
# 109
int ans = max(a, b); 
# 111
return (ans > 0) ? ans : 0; 
# 113
} 
# 115
static inline unsigned __vimax_s16x2_relu(const unsigned a, const unsigned b) { 
# 123
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 124
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 126
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 127
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 130
short aS_lo = *((short *)(&aU_lo)); 
# 131
short aS_hi = *((short *)(&aU_hi)); 
# 133
short bS_lo = *((short *)(&bU_lo)); 
# 134
short bS_hi = *((short *)(&bU_hi)); 
# 137
short ansS_lo = (short)max(aS_lo, bS_lo); 
# 138
short ansS_hi = (short)max(aS_hi, bS_hi); 
# 141
if (ansS_lo < 0) { ansS_lo = (0); }  
# 142
if (ansS_hi < 0) { ansS_hi = (0); }  
# 145
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 146
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 149
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 151
return ans; 
# 153
} 
# 155
static inline int __vimin_s32_relu(const int a, const int b) { 
# 162
int ans = min(a, b); 
# 164
return (ans > 0) ? ans : 0; 
# 166
} 
# 168
static inline unsigned __vimin_s16x2_relu(const unsigned a, const unsigned b) { 
# 176
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 177
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 179
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 180
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 183
short aS_lo = *((short *)(&aU_lo)); 
# 184
short aS_hi = *((short *)(&aU_hi)); 
# 186
short bS_lo = *((short *)(&bU_lo)); 
# 187
short bS_hi = *((short *)(&bU_hi)); 
# 190
short ansS_lo = (short)min(aS_lo, bS_lo); 
# 191
short ansS_hi = (short)min(aS_hi, bS_hi); 
# 194
if (ansS_lo < 0) { ansS_lo = (0); }  
# 195
if (ansS_hi < 0) { ansS_hi = (0); }  
# 198
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 199
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 202
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 204
return ans; 
# 206
} 
# 208
static inline int __vimax3_s32(const int a, const int b, const int c) { 
# 218 "/usr/include/crt/device_functions.hpp" 3
return max(max(a, b), c); 
# 220
} 
# 222
static inline unsigned __vimax3_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 234 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 235
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 237
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 238
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 240
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 241
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 244
short aS_lo = *((short *)(&aU_lo)); 
# 245
short aS_hi = *((short *)(&aU_hi)); 
# 247
short bS_lo = *((short *)(&bU_lo)); 
# 248
short bS_hi = *((short *)(&bU_hi)); 
# 250
short cS_lo = *((short *)(&cU_lo)); 
# 251
short cS_hi = *((short *)(&cU_hi)); 
# 254
short ansS_lo = (short)max(max(aS_lo, bS_lo), cS_lo); 
# 255
short ansS_hi = (short)max(max(aS_hi, bS_hi), cS_hi); 
# 258
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 259
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 262
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 264
return ans; 
# 266
} 
# 268
static inline unsigned __vimax3_u32(const unsigned a, const unsigned b, const unsigned c) { 
# 278 "/usr/include/crt/device_functions.hpp" 3
return max(max(a, b), c); 
# 280
} 
# 282
static inline unsigned __vimax3_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 293 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 294
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 296
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 297
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 299
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 300
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 303
unsigned short ansU_lo = (unsigned short)max(max(aU_lo, bU_lo), cU_lo); 
# 304
unsigned short ansU_hi = (unsigned short)max(max(aU_hi, bU_hi), cU_hi); 
# 307
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 309
return ans; 
# 311
} 
# 313
static inline int __vimin3_s32(const int a, const int b, const int c) { 
# 323 "/usr/include/crt/device_functions.hpp" 3
return min(min(a, b), c); 
# 325
} 
# 327
static inline unsigned __vimin3_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 338 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 339
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 341
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 342
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 344
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 345
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 348
short aS_lo = *((short *)(&aU_lo)); 
# 349
short aS_hi = *((short *)(&aU_hi)); 
# 351
short bS_lo = *((short *)(&bU_lo)); 
# 352
short bS_hi = *((short *)(&bU_hi)); 
# 354
short cS_lo = *((short *)(&cU_lo)); 
# 355
short cS_hi = *((short *)(&cU_hi)); 
# 358
short ansS_lo = (short)min(min(aS_lo, bS_lo), cS_lo); 
# 359
short ansS_hi = (short)min(min(aS_hi, bS_hi), cS_hi); 
# 362
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 363
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 366
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 368
return ans; 
# 370
} 
# 372
static inline unsigned __vimin3_u32(const unsigned a, const unsigned b, const unsigned c) { 
# 382 "/usr/include/crt/device_functions.hpp" 3
return min(min(a, b), c); 
# 384
} 
# 386
static inline unsigned __vimin3_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 397 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 398
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 400
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 401
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 403
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 404
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 407
unsigned short ansU_lo = (unsigned short)min(min(aU_lo, bU_lo), cU_lo); 
# 408
unsigned short ansU_hi = (unsigned short)min(min(aU_hi, bU_hi), cU_hi); 
# 411
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 413
return ans; 
# 415
} 
# 417
static inline int __vimax3_s32_relu(const int a, const int b, const int c) { 
# 427 "/usr/include/crt/device_functions.hpp" 3
int ans = max(max(a, b), c); 
# 429
return (ans > 0) ? ans : 0; 
# 431
} 
# 433
static inline unsigned __vimax3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
# 444 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 445
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 447
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 448
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 450
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 451
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 454
short aS_lo = *((short *)(&aU_lo)); 
# 455
short aS_hi = *((short *)(&aU_hi)); 
# 457
short bS_lo = *((short *)(&bU_lo)); 
# 458
short bS_hi = *((short *)(&bU_hi)); 
# 460
short cS_lo = *((short *)(&cU_lo)); 
# 461
short cS_hi = *((short *)(&cU_hi)); 
# 464
short ansS_lo = (short)max(max(aS_lo, bS_lo), cS_lo); 
# 465
short ansS_hi = (short)max(max(aS_hi, bS_hi), cS_hi); 
# 468
if (ansS_lo < 0) { ansS_lo = (0); }  
# 469
if (ansS_hi < 0) { ansS_hi = (0); }  
# 472
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 473
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 476
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 478
return ans; 
# 480
} 
# 482
static inline int __vimin3_s32_relu(const int a, const int b, const int c) { 
# 492 "/usr/include/crt/device_functions.hpp" 3
int ans = min(min(a, b), c); 
# 494
return (ans > 0) ? ans : 0; 
# 496
} 
# 498
static inline unsigned __vimin3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
# 509 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 510
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 512
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 513
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 515
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 516
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 519
short aS_lo = *((short *)(&aU_lo)); 
# 520
short aS_hi = *((short *)(&aU_hi)); 
# 522
short bS_lo = *((short *)(&bU_lo)); 
# 523
short bS_hi = *((short *)(&bU_hi)); 
# 525
short cS_lo = *((short *)(&cU_lo)); 
# 526
short cS_hi = *((short *)(&cU_hi)); 
# 529
short ansS_lo = (short)min(min(aS_lo, bS_lo), cS_lo); 
# 530
short ansS_hi = (short)min(min(aS_hi, bS_hi), cS_hi); 
# 533
if (ansS_lo < 0) { ansS_lo = (0); }  
# 534
if (ansS_hi < 0) { ansS_hi = (0); }  
# 537
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 538
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 541
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 543
return ans; 
# 545
} 
# 547
static inline int __viaddmax_s32(const int a, const int b, const int c) { 
# 557 "/usr/include/crt/device_functions.hpp" 3
return max(a + b, c); 
# 559
} 
# 561
static inline unsigned __viaddmax_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 572 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 573
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 575
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 576
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 578
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 579
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 582
short aS_lo = *((short *)(&aU_lo)); 
# 583
short aS_hi = *((short *)(&aU_hi)); 
# 585
short bS_lo = *((short *)(&bU_lo)); 
# 586
short bS_hi = *((short *)(&bU_hi)); 
# 588
short cS_lo = *((short *)(&cU_lo)); 
# 589
short cS_hi = *((short *)(&cU_hi)); 
# 592
short ansS_lo = (short)max((short)(aS_lo + bS_lo), cS_lo); 
# 593
short ansS_hi = (short)max((short)(aS_hi + bS_hi), cS_hi); 
# 596
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 597
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 600
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 602
return ans; 
# 604
} 
# 606
static inline unsigned __viaddmax_u32(const unsigned a, const unsigned b, const unsigned c) { 
# 616 "/usr/include/crt/device_functions.hpp" 3
return max(a + b, c); 
# 618
} 
# 620
static inline unsigned __viaddmax_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 631 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 632
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 634
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 635
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 637
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 638
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 641
unsigned short ansU_lo = (unsigned short)max((unsigned short)(aU_lo + bU_lo), cU_lo); 
# 642
unsigned short ansU_hi = (unsigned short)max((unsigned short)(aU_hi + bU_hi), cU_hi); 
# 645
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 647
return ans; 
# 649
} 
# 651
static inline int __viaddmin_s32(const int a, const int b, const int c) { 
# 661 "/usr/include/crt/device_functions.hpp" 3
return min(a + b, c); 
# 663
} 
# 665
static inline unsigned __viaddmin_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 676 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 677
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 679
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 680
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 682
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 683
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 686
short aS_lo = *((short *)(&aU_lo)); 
# 687
short aS_hi = *((short *)(&aU_hi)); 
# 689
short bS_lo = *((short *)(&bU_lo)); 
# 690
short bS_hi = *((short *)(&bU_hi)); 
# 692
short cS_lo = *((short *)(&cU_lo)); 
# 693
short cS_hi = *((short *)(&cU_hi)); 
# 696
short ansS_lo = (short)min((short)(aS_lo + bS_lo), cS_lo); 
# 697
short ansS_hi = (short)min((short)(aS_hi + bS_hi), cS_hi); 
# 700
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 701
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 704
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 706
return ans; 
# 708
} 
# 710
static inline unsigned __viaddmin_u32(const unsigned a, const unsigned b, const unsigned c) { 
# 720 "/usr/include/crt/device_functions.hpp" 3
return min(a + b, c); 
# 722
} 
# 724
static inline unsigned __viaddmin_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
# 735 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 736
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 738
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 739
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 741
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 742
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 745
unsigned short ansU_lo = (unsigned short)min((unsigned short)(aU_lo + bU_lo), cU_lo); 
# 746
unsigned short ansU_hi = (unsigned short)min((unsigned short)(aU_hi + bU_hi), cU_hi); 
# 749
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 751
return ans; 
# 753
} 
# 755
static inline int __viaddmax_s32_relu(const int a, const int b, const int c) { 
# 765 "/usr/include/crt/device_functions.hpp" 3
int ans = max(a + b, c); 
# 767
return (ans > 0) ? ans : 0; 
# 769
} 
# 771
static inline unsigned __viaddmax_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
# 782 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 783
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 785
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 786
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 788
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 789
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 792
short aS_lo = *((short *)(&aU_lo)); 
# 793
short aS_hi = *((short *)(&aU_hi)); 
# 795
short bS_lo = *((short *)(&bU_lo)); 
# 796
short bS_hi = *((short *)(&bU_hi)); 
# 798
short cS_lo = *((short *)(&cU_lo)); 
# 799
short cS_hi = *((short *)(&cU_hi)); 
# 802
short ansS_lo = (short)max((short)(aS_lo + bS_lo), cS_lo); 
# 803
short ansS_hi = (short)max((short)(aS_hi + bS_hi), cS_hi); 
# 805
if (ansS_lo < 0) { ansS_lo = (0); }  
# 806
if (ansS_hi < 0) { ansS_hi = (0); }  
# 809
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 810
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 813
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 815
return ans; 
# 817
} 
# 819
static inline int __viaddmin_s32_relu(const int a, const int b, const int c) { 
# 829 "/usr/include/crt/device_functions.hpp" 3
int ans = min(a + b, c); 
# 831
return (ans > 0) ? ans : 0; 
# 833
} 
# 835
static inline unsigned __viaddmin_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
# 846 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 847
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 849
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 850
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 852
unsigned short cU_lo = (unsigned short)(c & 65535U); 
# 853
unsigned short cU_hi = (unsigned short)(c >> 16); 
# 856
short aS_lo = *((short *)(&aU_lo)); 
# 857
short aS_hi = *((short *)(&aU_hi)); 
# 859
short bS_lo = *((short *)(&bU_lo)); 
# 860
short bS_hi = *((short *)(&bU_hi)); 
# 862
short cS_lo = *((short *)(&cU_lo)); 
# 863
short cS_hi = *((short *)(&cU_hi)); 
# 866
short ansS_lo = (short)min((short)(aS_lo + bS_lo), cS_lo); 
# 867
short ansS_hi = (short)min((short)(aS_hi + bS_hi), cS_hi); 
# 869
if (ansS_lo < 0) { ansS_lo = (0); }  
# 870
if (ansS_hi < 0) { ansS_hi = (0); }  
# 873
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 874
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 877
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 879
return ans; 
# 881
} 
# 885
static inline int __vibmax_s32(const int a, const int b, bool *const pred) { 
# 899 "/usr/include/crt/device_functions.hpp" 3
int ans = max(a, b); 
# 901
(*pred) = (a >= b); 
# 902
return ans; 
# 904
} 
# 906
static inline unsigned __vibmax_u32(const unsigned a, const unsigned b, bool *const pred) { 
# 920 "/usr/include/crt/device_functions.hpp" 3
unsigned ans = max(a, b); 
# 922
(*pred) = (a >= b); 
# 923
return ans; 
# 925
} 
# 928
static inline int __vibmin_s32(const int a, const int b, bool *const pred) { 
# 942 "/usr/include/crt/device_functions.hpp" 3
int ans = min(a, b); 
# 944
(*pred) = (a <= b); 
# 945
return ans; 
# 947
} 
# 950
static inline unsigned __vibmin_u32(const unsigned a, const unsigned b, bool *const pred) { 
# 964 "/usr/include/crt/device_functions.hpp" 3
unsigned ans = min(a, b); 
# 966
(*pred) = (a <= b); 
# 967
return ans; 
# 969
} 
# 971
static inline unsigned __vibmax_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
# 993 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 994
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 996
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 997
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 1000
short aS_lo = *((short *)(&aU_lo)); 
# 1001
short aS_hi = *((short *)(&aU_hi)); 
# 1003
short bS_lo = *((short *)(&bU_lo)); 
# 1004
short bS_hi = *((short *)(&bU_hi)); 
# 1007
short ansS_lo = (short)max(aS_lo, bS_lo); 
# 1008
short ansS_hi = (short)max(aS_hi, bS_hi); 
# 1010
(*pred_hi) = (aS_hi >= bS_hi); 
# 1011
(*pred_lo) = (aS_lo >= bS_lo); 
# 1014
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 1015
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 1018
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 1020
return ans; 
# 1022
} 
# 1024
static inline unsigned __vibmax_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
# 1046 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 1047
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 1049
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 1050
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 1053
unsigned short ansU_lo = (unsigned short)max(aU_lo, bU_lo); 
# 1054
unsigned short ansU_hi = (unsigned short)max(aU_hi, bU_hi); 
# 1056
(*pred_hi) = (aU_hi >= bU_hi); 
# 1057
(*pred_lo) = (aU_lo >= bU_lo); 
# 1060
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 1062
return ans; 
# 1064
} 
# 1066
static inline unsigned __vibmin_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
# 1088 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 1089
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 1091
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 1092
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 1095
short aS_lo = *((short *)(&aU_lo)); 
# 1096
short aS_hi = *((short *)(&aU_hi)); 
# 1098
short bS_lo = *((short *)(&bU_lo)); 
# 1099
short bS_hi = *((short *)(&bU_hi)); 
# 1102
short ansS_lo = (short)min(aS_lo, bS_lo); 
# 1103
short ansS_hi = (short)min(aS_hi, bS_hi); 
# 1105
(*pred_hi) = (aS_hi <= bS_hi); 
# 1106
(*pred_lo) = (aS_lo <= bS_lo); 
# 1109
unsigned short ansU_lo = *((unsigned short *)(&ansS_lo)); 
# 1110
unsigned short ansU_hi = *((unsigned short *)(&ansS_hi)); 
# 1113
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 1115
return ans; 
# 1117
} 
# 1119
static inline unsigned __vibmin_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
# 1141 "/usr/include/crt/device_functions.hpp" 3
unsigned short aU_lo = (unsigned short)(a & 65535U); 
# 1142
unsigned short aU_hi = (unsigned short)(a >> 16); 
# 1144
unsigned short bU_lo = (unsigned short)(b & 65535U); 
# 1145
unsigned short bU_hi = (unsigned short)(b >> 16); 
# 1148
unsigned short ansU_lo = (unsigned short)min(aU_lo, bU_lo); 
# 1149
unsigned short ansU_hi = (unsigned short)min(aU_hi, bU_hi); 
# 1151
(*pred_hi) = (aU_hi <= bU_hi); 
# 1152
(*pred_lo) = (aU_lo <= bU_lo); 
# 1155
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
# 1157
return ans; 
# 1159
} 
# 110 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 120 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 120
{ } 
#endif
# 122 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 122
{ } 
#endif
# 124 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 126
{ } 
#endif
# 128 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 128
{ } 
#endif
# 130 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 130
{ } 
#endif
# 132 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 132
{ } 
#endif
# 134 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 134
{ } 
#endif
# 136 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 136
{ } 
#endif
# 138 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 138
{ } 
#endif
# 140 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 140
{ } 
#endif
# 142 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 142
{ } 
#endif
# 144 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 144
{ } 
#endif
# 146 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 146
{ } 
#endif
# 148 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 148
{ } 
#endif
# 150 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 150
{ } 
#endif
# 177 "/usr/include/device_atomic_functions.h" 3
extern "C" {
# 186
}
# 195 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAdd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 195
{ } 
#endif
# 197 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicExch(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 197
{ } 
#endif
# 199 "/usr/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 199
{ } 
#endif
# 201 "/usr/include/device_atomic_functions.h" 3
__attribute((deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 201
{ } 
#endif
# 203 "/usr/include/device_atomic_functions.h" 3
__attribute((deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 203
{ } 
#endif
# 87 "/usr/include/crt/device_double_functions.h" 3
extern "C" {
# 1139 "/usr/include/crt/device_double_functions.h" 3
}
# 1147
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode); 
# 1149
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1151
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1153
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1155
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
# 1157
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
# 1159
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
# 1161
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
# 1163
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1165
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1167
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
# 1169
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 1171
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
# 93 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode) 
# 94
{int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;
# 99
::exit(___);}
#if 0
# 94
{ 
# 95
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
# 99
} 
#endif
# 101 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode) 
# 102
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 107
::exit(___);}
#if 0
# 102
{ 
# 103
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
# 107
} 
#endif
# 109 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode) 
# 110
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 115
::exit(___);}
#if 0
# 110
{ 
# 111
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
# 115
} 
#endif
# 117 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode) 
# 118
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 123
::exit(___);}
#if 0
# 118
{ 
# 119
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
# 123
} 
#endif
# 125 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode) 
# 126
{int volatile ___ = 1;(void)a;(void)mode;
# 131
::exit(___);}
#if 0
# 126
{ 
# 127
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
# 131
} 
#endif
# 133 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode) 
# 134
{int volatile ___ = 1;(void)a;(void)mode;
# 139
::exit(___);}
#if 0
# 134
{ 
# 135
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
# 139
} 
#endif
# 141 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode) 
# 142
{int volatile ___ = 1;(void)a;(void)mode;
# 147
::exit(___);}
#if 0
# 142
{ 
# 143
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
# 147
} 
#endif
# 149 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode) 
# 150
{int volatile ___ = 1;(void)a;(void)mode;
# 155
::exit(___);}
#if 0
# 150
{ 
# 151
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
# 155
} 
#endif
# 157 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode) 
# 158
{int volatile ___ = 1;(void)a;(void)mode;
# 163
::exit(___);}
#if 0
# 158
{ 
# 159
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
# 163
} 
#endif
# 165 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode) 
# 166
{int volatile ___ = 1;(void)a;(void)mode;
# 171
::exit(___);}
#if 0
# 166
{ 
# 167
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
# 171
} 
#endif
# 173 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode) 
# 174
{int volatile ___ = 1;(void)a;(void)mode;
# 176
::exit(___);}
#if 0
# 174
{ 
# 175
return (double)a; 
# 176
} 
#endif
# 178 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode) 
# 179
{int volatile ___ = 1;(void)a;(void)mode;
# 181
::exit(___);}
#if 0
# 179
{ 
# 180
return (double)a; 
# 181
} 
#endif
# 183 "/usr/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode) 
# 184
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 184
{ 
# 185
return (double)a; 
# 186
} 
#endif
# 103 "/usr/include/sm_20_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 103
{ } 
#endif
# 110 "/usr/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMin(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/usr/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMax(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/usr/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicAnd(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/usr/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicOr(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/usr/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicXor(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 120 "/usr/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMin(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 120
{ } 
#endif
# 122 "/usr/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMax(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 122
{ } 
#endif
# 124 "/usr/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/usr/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicOr(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 126
{ } 
#endif
# 128 "/usr/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicXor(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 128
{ } 
#endif
# 307 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 307
{ } 
#endif
# 310 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 310
{ } 
#endif
# 313 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 313
{ } 
#endif
# 316 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 316
{ } 
#endif
# 319 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 319
{ } 
#endif
# 322 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 322
{ } 
#endif
# 325 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 325
{ } 
#endif
# 328 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 328
{ } 
#endif
# 331 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 331
{ } 
#endif
# 334 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 334
{ } 
#endif
# 337 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 337
{ } 
#endif
# 340 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicSub_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 340
{ } 
#endif
# 343 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicSub_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 343
{ } 
#endif
# 346 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicSub_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 346
{ } 
#endif
# 349 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicSub_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 349
{ } 
#endif
# 352 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 352
{ } 
#endif
# 355 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 355
{ } 
#endif
# 358 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 358
{ } 
#endif
# 361 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 361
{ } 
#endif
# 364 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 364
{ } 
#endif
# 367 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 367
{ } 
#endif
# 370 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 370
{ } 
#endif
# 373 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 373
{ } 
#endif
# 376 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 376
{ } 
#endif
# 379 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 379
{ } 
#endif
# 382 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMin_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 382
{ } 
#endif
# 385 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMin_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 385
{ } 
#endif
# 388 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 388
{ } 
#endif
# 391 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 391
{ } 
#endif
# 394 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 394
{ } 
#endif
# 397 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 397
{ } 
#endif
# 400 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 400
{ } 
#endif
# 403 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 403
{ } 
#endif
# 406 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMax_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 406
{ } 
#endif
# 409 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMax_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 409
{ } 
#endif
# 412 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 412
{ } 
#endif
# 415 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 415
{ } 
#endif
# 418 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 418
{ } 
#endif
# 421 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 421
{ } 
#endif
# 424 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 424
{ } 
#endif
# 427 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 427
{ } 
#endif
# 430 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 430
{ } 
#endif
# 433 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 433
{ } 
#endif
# 436 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 436
{ } 
#endif
# 439 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 439
{ } 
#endif
# 442 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
# 443
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 443
{ } 
#endif
# 446 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
# 447
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 447
{ } 
#endif
# 450 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicCAS_block(unsigned long long *address, unsigned long long 
# 451
compare, unsigned long long 
# 452
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 452
{ } 
#endif
# 455 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicCAS_system(unsigned long long *address, unsigned long long 
# 456
compare, unsigned long long 
# 457
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 457
{ } 
#endif
# 460 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 460
{ } 
#endif
# 463 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 463
{ } 
#endif
# 466 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicAnd_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 466
{ } 
#endif
# 469 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicAnd_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 469
{ } 
#endif
# 472 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 472
{ } 
#endif
# 475 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 475
{ } 
#endif
# 478 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 478
{ } 
#endif
# 481 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 481
{ } 
#endif
# 484 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 484
{ } 
#endif
# 487 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 487
{ } 
#endif
# 490 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicOr_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 490
{ } 
#endif
# 493 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicOr_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 493
{ } 
#endif
# 496 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 496
{ } 
#endif
# 499 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 499
{ } 
#endif
# 502 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 502
{ } 
#endif
# 505 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 505
{ } 
#endif
# 508 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 508
{ } 
#endif
# 511 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 511
{ } 
#endif
# 514 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicXor_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 514
{ } 
#endif
# 517 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicXor_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 517
{ } 
#endif
# 520 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 520
{ } 
#endif
# 523 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 523
{ } 
#endif
# 526 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 526
{ } 
#endif
# 529 "/usr/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 529
{ } 
#endif
# 97 "/usr/include/sm_20_intrinsics.h" 3
extern "C" {
# 1510 "/usr/include/sm_20_intrinsics.h" 3
}
# 1522 "/usr/include/sm_20_intrinsics.h" 3
__attribute((deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning)."))) __attribute__((unused)) static inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1522
{ } 
#endif
# 1524 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1524
{ } 
#endif
# 1526 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1526
{ } 
#endif
# 1528 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1528
{ } 
#endif
# 1533 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1533
{ } 
#endif
# 1534 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __isShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1534
{ } 
#endif
# 1535 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __isConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1535
{ } 
#endif
# 1536 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __isLocal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1536
{ } 
#endif
# 1538 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __isGridConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1538
{ } 
#endif
# 1540 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline size_t __cvta_generic_to_global(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1540
{ } 
#endif
# 1541 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline size_t __cvta_generic_to_shared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1541
{ } 
#endif
# 1542 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline size_t __cvta_generic_to_constant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1542
{ } 
#endif
# 1543 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline size_t __cvta_generic_to_local(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1543
{ } 
#endif
# 1545 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline size_t __cvta_generic_to_grid_constant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1545
{ } 
#endif
# 1548 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline void *__cvta_global_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1548
{ } 
#endif
# 1549 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline void *__cvta_shared_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1549
{ } 
#endif
# 1550 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline void *__cvta_constant_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1550
{ } 
#endif
# 1551 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline void *__cvta_local_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1551
{ } 
#endif
# 1553 "/usr/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline void *__cvta_grid_constant_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1553
{ } 
#endif
# 108 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __fns(unsigned mask, unsigned base, int offset) {int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline void __barrier_sync(unsigned id) {int volatile ___ = 1;(void)id;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline void __barrier_sync_count(unsigned id, unsigned cnt) {int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline void __syncwarp(unsigned mask = 4294967295U) {int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __all_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __any_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 113
{ } 
#endif
# 114 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __uni_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 114
{ } 
#endif
# 115 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ballot_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 115
{ } 
#endif
# 116 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __activemask() {int volatile ___ = 1;::exit(___);}
#if 0
# 116
{ } 
#endif
# 125 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 125
{ } 
#endif
# 126 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 131
{ } 
#endif
# 132 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 136
{ } 
#endif
# 139 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 149
{ } 
#endif
# 150 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 150
{ } 
#endif
# 154 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl(unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long long __shfl(long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 155
{ } 
#endif
# 156 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_up(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 156
{ } 
#endif
# 157 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_up(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 157
{ } 
#endif
# 158 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_down(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 158
{ } 
#endif
# 159 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_down(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 159
{ } 
#endif
# 160 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_xor(long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 160
{ } 
#endif
# 161 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 161
{ } 
#endif
# 162 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 165
{ } 
#endif
# 168 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long long __shfl_sync(unsigned mask, long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long long __shfl_up_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long long __shfl_down_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 173
{ } 
#endif
# 174 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 174
{ } 
#endif
# 175 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 175
{ } 
#endif
# 176 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 176
{ } 
#endif
# 177 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 179
{ } 
#endif
# 183 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 185
{ } 
#endif
# 186 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 186
{ } 
#endif
# 187 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/usr/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 190
{ } 
#endif
# 193 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 193
{ } 
#endif
# 194 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 194
{ } 
#endif
# 195 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 195
{ } 
#endif
# 196 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 196
{ } 
#endif
# 197 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 197
{ } 
#endif
# 198 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 198
{ } 
#endif
# 199 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 199
{ } 
#endif
# 200 "/usr/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 200
{ } 
#endif
# 87 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 87
{ } 
#endif
# 88 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 88
{ } 
#endif
# 90 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 90
{ } 
#endif
# 91 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 91
{ } 
#endif
# 92 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 101
{ } 
#endif
# 103 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 115
{ } 
#endif
# 116 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 116
{ } 
#endif
# 117 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 117
{ } 
#endif
# 118 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 118
{ } 
#endif
# 119 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 119
{ } 
#endif
# 123 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 123
{ } 
#endif
# 124 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldcg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 131
{ } 
#endif
# 132 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 137
{ } 
#endif
# 139 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldcg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 149
{ } 
#endif
# 151 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 151
{ } 
#endif
# 152 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 152
{ } 
#endif
# 153 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 155
{ } 
#endif
# 159 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 159
{ } 
#endif
# 160 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 160
{ } 
#endif
# 162 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldca(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 173
{ } 
#endif
# 175 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 175
{ } 
#endif
# 176 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 176
{ } 
#endif
# 177 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldca(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 185
{ } 
#endif
# 187 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 191
{ } 
#endif
# 195 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 195
{ } 
#endif
# 196 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 196
{ } 
#endif
# 198 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 198
{ } 
#endif
# 199 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 199
{ } 
#endif
# 200 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 200
{ } 
#endif
# 201 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 201
{ } 
#endif
# 202 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldcs(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 202
{ } 
#endif
# 203 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 203
{ } 
#endif
# 204 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 204
{ } 
#endif
# 205 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 205
{ } 
#endif
# 206 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 206
{ } 
#endif
# 207 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 207
{ } 
#endif
# 208 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 208
{ } 
#endif
# 209 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 209
{ } 
#endif
# 211 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 211
{ } 
#endif
# 212 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 212
{ } 
#endif
# 213 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 213
{ } 
#endif
# 214 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldcs(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 214
{ } 
#endif
# 215 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 215
{ } 
#endif
# 216 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 216
{ } 
#endif
# 217 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 217
{ } 
#endif
# 218 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 218
{ } 
#endif
# 219 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 219
{ } 
#endif
# 220 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 220
{ } 
#endif
# 221 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 221
{ } 
#endif
# 223 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 223
{ } 
#endif
# 224 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 224
{ } 
#endif
# 225 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 225
{ } 
#endif
# 226 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 226
{ } 
#endif
# 227 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 227
{ } 
#endif
# 231 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldlu(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 231
{ } 
#endif
# 232 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldlu(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 232
{ } 
#endif
# 234 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldlu(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 234
{ } 
#endif
# 235 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldlu(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 235
{ } 
#endif
# 236 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldlu(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 236
{ } 
#endif
# 237 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldlu(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 237
{ } 
#endif
# 238 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldlu(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 238
{ } 
#endif
# 239 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldlu(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 239
{ } 
#endif
# 240 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldlu(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 240
{ } 
#endif
# 241 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldlu(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 241
{ } 
#endif
# 242 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldlu(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 242
{ } 
#endif
# 243 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldlu(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 243
{ } 
#endif
# 244 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldlu(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 244
{ } 
#endif
# 245 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldlu(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 245
{ } 
#endif
# 247 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldlu(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 247
{ } 
#endif
# 248 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldlu(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 248
{ } 
#endif
# 249 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldlu(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 249
{ } 
#endif
# 250 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldlu(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 250
{ } 
#endif
# 251 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldlu(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 251
{ } 
#endif
# 252 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldlu(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 252
{ } 
#endif
# 253 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldlu(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 253
{ } 
#endif
# 254 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldlu(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 254
{ } 
#endif
# 255 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldlu(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 255
{ } 
#endif
# 256 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldlu(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 256
{ } 
#endif
# 257 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldlu(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 257
{ } 
#endif
# 259 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldlu(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 259
{ } 
#endif
# 260 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldlu(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 260
{ } 
#endif
# 261 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldlu(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 261
{ } 
#endif
# 262 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldlu(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 262
{ } 
#endif
# 263 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldlu(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 263
{ } 
#endif
# 267 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldcv(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 267
{ } 
#endif
# 268 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldcv(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 268
{ } 
#endif
# 270 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldcv(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 270
{ } 
#endif
# 271 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldcv(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 271
{ } 
#endif
# 272 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldcv(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 272
{ } 
#endif
# 273 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldcv(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 273
{ } 
#endif
# 274 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldcv(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 274
{ } 
#endif
# 275 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldcv(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 275
{ } 
#endif
# 276 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldcv(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 276
{ } 
#endif
# 277 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldcv(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 277
{ } 
#endif
# 278 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldcv(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 278
{ } 
#endif
# 279 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldcv(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 279
{ } 
#endif
# 280 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldcv(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 280
{ } 
#endif
# 281 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldcv(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 281
{ } 
#endif
# 283 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldcv(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 283
{ } 
#endif
# 284 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldcv(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 284
{ } 
#endif
# 285 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldcv(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 285
{ } 
#endif
# 286 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldcv(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 286
{ } 
#endif
# 287 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldcv(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 287
{ } 
#endif
# 288 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldcv(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 288
{ } 
#endif
# 289 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldcv(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 289
{ } 
#endif
# 290 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldcv(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 290
{ } 
#endif
# 291 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldcv(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 291
{ } 
#endif
# 292 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldcv(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 292
{ } 
#endif
# 293 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldcv(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 293
{ } 
#endif
# 295 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldcv(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 295
{ } 
#endif
# 296 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldcv(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 296
{ } 
#endif
# 297 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldcv(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 297
{ } 
#endif
# 298 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldcv(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 298
{ } 
#endif
# 299 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldcv(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 299
{ } 
#endif
# 303 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 303
{ } 
#endif
# 304 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 304
{ } 
#endif
# 306 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 306
{ } 
#endif
# 307 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 307
{ } 
#endif
# 308 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 308
{ } 
#endif
# 309 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 309
{ } 
#endif
# 310 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 310
{ } 
#endif
# 311 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 311
{ } 
#endif
# 312 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 312
{ } 
#endif
# 313 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 313
{ } 
#endif
# 314 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 314
{ } 
#endif
# 315 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 315
{ } 
#endif
# 316 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 316
{ } 
#endif
# 317 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 317
{ } 
#endif
# 319 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 319
{ } 
#endif
# 320 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 320
{ } 
#endif
# 321 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 321
{ } 
#endif
# 322 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 322
{ } 
#endif
# 323 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 323
{ } 
#endif
# 324 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 324
{ } 
#endif
# 325 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 325
{ } 
#endif
# 326 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 326
{ } 
#endif
# 327 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 327
{ } 
#endif
# 328 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 328
{ } 
#endif
# 329 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 329
{ } 
#endif
# 331 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 331
{ } 
#endif
# 332 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 332
{ } 
#endif
# 333 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 333
{ } 
#endif
# 334 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 334
{ } 
#endif
# 335 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 335
{ } 
#endif
# 339 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 339
{ } 
#endif
# 340 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 340
{ } 
#endif
# 342 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 342
{ } 
#endif
# 343 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 343
{ } 
#endif
# 344 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 344
{ } 
#endif
# 345 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 345
{ } 
#endif
# 346 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 346
{ } 
#endif
# 347 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 347
{ } 
#endif
# 348 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 348
{ } 
#endif
# 349 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 349
{ } 
#endif
# 350 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 350
{ } 
#endif
# 351 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 351
{ } 
#endif
# 352 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 352
{ } 
#endif
# 353 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 353
{ } 
#endif
# 355 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 355
{ } 
#endif
# 356 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 356
{ } 
#endif
# 357 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 357
{ } 
#endif
# 358 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 358
{ } 
#endif
# 359 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 359
{ } 
#endif
# 360 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 360
{ } 
#endif
# 361 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 361
{ } 
#endif
# 362 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 362
{ } 
#endif
# 363 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 363
{ } 
#endif
# 364 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 364
{ } 
#endif
# 365 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 365
{ } 
#endif
# 367 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 367
{ } 
#endif
# 368 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 368
{ } 
#endif
# 369 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 369
{ } 
#endif
# 370 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 370
{ } 
#endif
# 371 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 371
{ } 
#endif
# 375 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 375
{ } 
#endif
# 376 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 376
{ } 
#endif
# 378 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 378
{ } 
#endif
# 379 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 379
{ } 
#endif
# 380 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 380
{ } 
#endif
# 381 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 381
{ } 
#endif
# 382 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 382
{ } 
#endif
# 383 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 383
{ } 
#endif
# 384 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 384
{ } 
#endif
# 385 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 385
{ } 
#endif
# 386 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 386
{ } 
#endif
# 387 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 387
{ } 
#endif
# 388 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 388
{ } 
#endif
# 389 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 389
{ } 
#endif
# 391 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 391
{ } 
#endif
# 392 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 392
{ } 
#endif
# 393 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 393
{ } 
#endif
# 394 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 394
{ } 
#endif
# 395 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 395
{ } 
#endif
# 396 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 396
{ } 
#endif
# 397 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 397
{ } 
#endif
# 398 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 398
{ } 
#endif
# 399 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 399
{ } 
#endif
# 400 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 400
{ } 
#endif
# 401 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 401
{ } 
#endif
# 403 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 403
{ } 
#endif
# 404 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 404
{ } 
#endif
# 405 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 405
{ } 
#endif
# 406 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 406
{ } 
#endif
# 407 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 407
{ } 
#endif
# 411 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 411
{ } 
#endif
# 412 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 412
{ } 
#endif
# 414 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 414
{ } 
#endif
# 415 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 415
{ } 
#endif
# 416 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 416
{ } 
#endif
# 417 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 417
{ } 
#endif
# 418 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 418
{ } 
#endif
# 419 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 419
{ } 
#endif
# 420 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 420
{ } 
#endif
# 421 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 421
{ } 
#endif
# 422 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 422
{ } 
#endif
# 423 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 423
{ } 
#endif
# 424 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 424
{ } 
#endif
# 425 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 425
{ } 
#endif
# 427 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 427
{ } 
#endif
# 428 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 428
{ } 
#endif
# 429 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 429
{ } 
#endif
# 430 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 430
{ } 
#endif
# 431 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 431
{ } 
#endif
# 432 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 432
{ } 
#endif
# 433 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 433
{ } 
#endif
# 434 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 434
{ } 
#endif
# 435 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 435
{ } 
#endif
# 436 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 436
{ } 
#endif
# 437 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 437
{ } 
#endif
# 439 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 439
{ } 
#endif
# 440 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 440
{ } 
#endif
# 441 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 441
{ } 
#endif
# 442 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 442
{ } 
#endif
# 443 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 443
{ } 
#endif
# 460 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 460
{ } 
#endif
# 472 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 472
{ } 
#endif
# 485 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 485
{ } 
#endif
# 497 "/usr/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 497
{ } 
#endif
# 89 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 89
{ } 
#endif
# 90 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 90
{ } 
#endif
# 92 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 93
{ } 
#endif
# 95 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 96
{ } 
#endif
# 98 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 99
{ } 
#endif
# 106 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 107
{ } 
#endif
# 109 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/usr/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 110
{ } 
#endif
# 93 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, float value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, double value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 100
{ } 
#endif
# 102 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, int value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, float value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, double value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 109
{ } 
#endif
# 111 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline void __nanosleep(unsigned ns) {int volatile ___ = 1;(void)ns;::exit(___);}
#if 0
# 111
{ } 
#endif
# 113 "/usr/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned short atomicCAS(unsigned short *address, unsigned short compare, unsigned short val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 113
{ } 
#endif
# 93 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_add_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_min_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_max_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 95
{ } 
#endif
# 97 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline int __reduce_add_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline int __reduce_min_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline int __reduce_max_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 101 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_and_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_or_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_xor_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 103
{ } 
#endif
# 106 "/usr/include/crt/sm_80_rt.h" 3
extern "C" {
# 107
__attribute__((unused)) inline void *__nv_associate_access_property(const void *ptr, unsigned long long 
# 108
property) {int volatile ___ = 1;(void)ptr;(void)property;
# 112
::exit(___);}
#if 0
# 108
{ 
# 109
__attribute__((unused)) extern void *__nv_associate_access_property_impl(const void *, unsigned long long); 
# 111
return __nv_associate_access_property_impl(ptr, property); 
# 112
} 
#endif
# 114 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_4(void *dst, const void *
# 115
src, unsigned 
# 116
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;
# 121
::exit(___);}
#if 0
# 116
{ 
# 117
__attribute__((unused)) extern void __nv_memcpy_async_shared_global_4_impl(void *, const void *, unsigned); 
# 120
__nv_memcpy_async_shared_global_4_impl(dst, src, src_size); 
# 121
} 
#endif
# 123 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_8(void *dst, const void *
# 124
src, unsigned 
# 125
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;
# 130
::exit(___);}
#if 0
# 125
{ 
# 126
__attribute__((unused)) extern void __nv_memcpy_async_shared_global_8_impl(void *, const void *, unsigned); 
# 129
__nv_memcpy_async_shared_global_8_impl(dst, src, src_size); 
# 130
} 
#endif
# 132 "/usr/include/crt/sm_80_rt.h" 3
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_16(void *dst, const void *
# 133
src, unsigned 
# 134
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;
# 139
::exit(___);}
#if 0
# 134
{ 
# 135
__attribute__((unused)) extern void __nv_memcpy_async_shared_global_16_impl(void *, const void *, unsigned); 
# 138
__nv_memcpy_async_shared_global_16_impl(dst, src, src_size); 
# 139
} 
#endif
# 141 "/usr/include/crt/sm_80_rt.h" 3
}
# 89 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __isCtaShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 89
{ } 
#endif
# 90 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __isClusterShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 90
{ } 
#endif
# 91 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline void *__cluster_map_shared_rank(const void *ptr, unsigned target_block_rank) {int volatile ___ = 1;(void)ptr;(void)target_block_rank;::exit(___);}
#if 0
# 91
{ } 
#endif
# 92 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __cluster_query_shared_rank(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline uint2 __cluster_map_shared_multicast(const void *ptr, unsigned cluster_cta_mask) {int volatile ___ = 1;(void)ptr;(void)cluster_cta_mask;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __clusterDimIsSpecified() {int volatile ___ = 1;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline dim3 __clusterDim() {int volatile ___ = 1;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline dim3 __clusterRelativeBlockIdx() {int volatile ___ = 1;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline dim3 __clusterGridDimInClusters() {int volatile ___ = 1;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline dim3 __clusterIdx() {int volatile ___ = 1;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __clusterRelativeBlockRank() {int volatile ___ = 1;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __clusterSizeInBlocks() {int volatile ___ = 1;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline void __cluster_barrier_arrive() {int volatile ___ = 1;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline void __cluster_barrier_arrive_relaxed() {int volatile ___ = 1;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline void __cluster_barrier_wait() {int volatile ___ = 1;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/usr/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline void __threadfence_cluster() {int volatile ___ = 1;::exit(___);}
#if 0
# 104
{ } 
#endif
# 65 "/usr/include/texture_indirect_functions.h" 3
template< class T> struct __nv_itex_trait { }; 
# 66
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
# 67
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
# 68
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
# 69
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
# 70
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
# 71
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
# 72
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
# 73
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
# 74
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
# 75
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
# 76
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
# 77
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
# 78
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
# 79
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
# 80
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
# 81
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
# 82
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
# 83
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
# 84
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
# 85
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
# 86
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
# 87
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
# 88
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
# 89
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
# 90
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
# 101 "/usr/include/texture_indirect_functions.h" 3
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
# 102
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
# 103
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
# 104
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
# 108
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 109
tex1Dfetch(T *ptr, cudaTextureObject_t obj, int x) 
# 110
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 112
::exit(___);}
#if 0
# 110
{ 
# 111
__nv_tex_surf_handler("__itex1Dfetch", ptr, obj, x); 
# 112
} 
#endif
# 114 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 115
tex1Dfetch(cudaTextureObject_t texObject, int x) 
# 116
{int volatile ___ = 1;(void)texObject;(void)x;
# 120
::exit(___);}
#if 0
# 116
{ 
# 117
T ret; 
# 118
tex1Dfetch(&ret, texObject, x); 
# 119
return ret; 
# 120
} 
#endif
# 122 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 123
tex1D(T *ptr, cudaTextureObject_t obj, float x) 
# 124
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 126
::exit(___);}
#if 0
# 124
{ 
# 125
__nv_tex_surf_handler("__itex1D", ptr, obj, x); 
# 126
} 
#endif
# 129 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 130
tex1D(cudaTextureObject_t texObject, float x) 
# 131
{int volatile ___ = 1;(void)texObject;(void)x;
# 135
::exit(___);}
#if 0
# 131
{ 
# 132
T ret; 
# 133
tex1D(&ret, texObject, x); 
# 134
return ret; 
# 135
} 
#endif
# 138 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 139
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y) 
# 140
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;
# 142
::exit(___);}
#if 0
# 140
{ 
# 141
__nv_tex_surf_handler("__itex2D", ptr, obj, x, y); 
# 142
} 
#endif
# 144 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 145
tex2D(cudaTextureObject_t texObject, float x, float y) 
# 146
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;
# 150
::exit(___);}
#if 0
# 146
{ 
# 147
T ret; 
# 148
tex2D(&ret, texObject, x, y); 
# 149
return ret; 
# 150
} 
#endif
# 153 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 154
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y, bool *
# 155
isResident) 
# 156
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;
# 160
::exit(___);}
#if 0
# 156
{ 
# 157
unsigned char res; 
# 158
__nv_tex_surf_handler("__itex2D_sparse", ptr, obj, x, y, &res); 
# 159
(*isResident) = (res != 0); 
# 160
} 
#endif
# 162 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 163
tex2D(cudaTextureObject_t texObject, float x, float y, bool *isResident) 
# 164
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)isResident;
# 168
::exit(___);}
#if 0
# 164
{ 
# 165
T ret; 
# 166
tex2D(&ret, texObject, x, y, isResident); 
# 167
return ret; 
# 168
} 
#endif
# 173 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 174
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 175
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 177
::exit(___);}
#if 0
# 175
{ 
# 176
__nv_tex_surf_handler("__itex3D", ptr, obj, x, y, z); 
# 177
} 
#endif
# 179 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 180
tex3D(cudaTextureObject_t texObject, float x, float y, float z) 
# 181
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 185
::exit(___);}
#if 0
# 181
{ 
# 182
T ret; 
# 183
tex3D(&ret, texObject, x, y, z); 
# 184
return ret; 
# 185
} 
#endif
# 188 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 189
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z, bool *
# 190
isResident) 
# 191
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)isResident;
# 195
::exit(___);}
#if 0
# 191
{ 
# 192
unsigned char res; 
# 193
__nv_tex_surf_handler("__itex3D_sparse", ptr, obj, x, y, z, &res); 
# 194
(*isResident) = (res != 0); 
# 195
} 
#endif
# 197 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 198
tex3D(cudaTextureObject_t texObject, float x, float y, float z, bool *isResident) 
# 199
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)isResident;
# 203
::exit(___);}
#if 0
# 199
{ 
# 200
T ret; 
# 201
tex3D(&ret, texObject, x, y, z, isResident); 
# 202
return ret; 
# 203
} 
#endif
# 207 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 208
tex1DLayered(T *ptr, cudaTextureObject_t obj, float x, int layer) 
# 209
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;
# 211
::exit(___);}
#if 0
# 209
{ 
# 210
__nv_tex_surf_handler("__itex1DLayered", ptr, obj, x, layer); 
# 211
} 
#endif
# 213 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 214
tex1DLayered(cudaTextureObject_t texObject, float x, int layer) 
# 215
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;
# 219
::exit(___);}
#if 0
# 215
{ 
# 216
T ret; 
# 217
tex1DLayered(&ret, texObject, x, layer); 
# 218
return ret; 
# 219
} 
#endif
# 221 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 222
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer) 
# 223
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;
# 225
::exit(___);}
#if 0
# 223
{ 
# 224
__nv_tex_surf_handler("__itex2DLayered", ptr, obj, x, y, layer); 
# 225
} 
#endif
# 227 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 228
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer) 
# 229
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;
# 233
::exit(___);}
#if 0
# 229
{ 
# 230
T ret; 
# 231
tex2DLayered(&ret, texObject, x, y, layer); 
# 232
return ret; 
# 233
} 
#endif
# 236 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 237
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, bool *isResident) 
# 238
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)isResident;
# 242
::exit(___);}
#if 0
# 238
{ 
# 239
unsigned char res; 
# 240
__nv_tex_surf_handler("__itex2DLayered_sparse", ptr, obj, x, y, layer, &res); 
# 241
(*isResident) = (res != 0); 
# 242
} 
#endif
# 244 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 245
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer, bool *isResident) 
# 246
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)isResident;
# 250
::exit(___);}
#if 0
# 246
{ 
# 247
T ret; 
# 248
tex2DLayered(&ret, texObject, x, y, layer, isResident); 
# 249
return ret; 
# 250
} 
#endif
# 254 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 255
texCubemap(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 256
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 258
::exit(___);}
#if 0
# 256
{ 
# 257
__nv_tex_surf_handler("__itexCubemap", ptr, obj, x, y, z); 
# 258
} 
#endif
# 261 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 262
texCubemap(cudaTextureObject_t texObject, float x, float y, float z) 
# 263
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 267
::exit(___);}
#if 0
# 263
{ 
# 264
T ret; 
# 265
texCubemap(&ret, texObject, x, y, z); 
# 266
return ret; 
# 267
} 
#endif
# 270 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 271
texCubemapLayered(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer) 
# 272
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;
# 274
::exit(___);}
#if 0
# 272
{ 
# 273
__nv_tex_surf_handler("__itexCubemapLayered", ptr, obj, x, y, z, layer); 
# 274
} 
#endif
# 276 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 277
texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer) 
# 278
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;
# 282
::exit(___);}
#if 0
# 278
{ 
# 279
T ret; 
# 280
texCubemapLayered(&ret, texObject, x, y, z, layer); 
# 281
return ret; 
# 282
} 
#endif
# 284 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 285
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, int comp = 0) 
# 286
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;
# 288
::exit(___);}
#if 0
# 286
{ 
# 287
__nv_tex_surf_handler("__itex2Dgather", ptr, obj, x, y, comp); 
# 288
} 
#endif
# 290 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 291
tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0) 
# 292
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;
# 296
::exit(___);}
#if 0
# 292
{ 
# 293
T ret; 
# 294
tex2Dgather(&ret, to, x, y, comp); 
# 295
return ret; 
# 296
} 
#endif
# 299 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 300
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, bool *isResident, int comp = 0) 
# 301
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;(void)comp;
# 305
::exit(___);}
#if 0
# 301
{ 
# 302
unsigned char res; 
# 303
__nv_tex_surf_handler("__itex2Dgather_sparse", ptr, obj, x, y, comp, &res); 
# 304
(*isResident) = (res != 0); 
# 305
} 
#endif
# 307 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 308
tex2Dgather(cudaTextureObject_t to, float x, float y, bool *isResident, int comp = 0) 
# 309
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)isResident;(void)comp;
# 313
::exit(___);}
#if 0
# 309
{ 
# 310
T ret; 
# 311
tex2Dgather(&ret, to, x, y, isResident, comp); 
# 312
return ret; 
# 313
} 
#endif
# 317 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 318
tex1DLod(T *ptr, cudaTextureObject_t obj, float x, float level) 
# 319
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;
# 321
::exit(___);}
#if 0
# 319
{ 
# 320
__nv_tex_surf_handler("__itex1DLod", ptr, obj, x, level); 
# 321
} 
#endif
# 323 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 324
tex1DLod(cudaTextureObject_t texObject, float x, float level) 
# 325
{int volatile ___ = 1;(void)texObject;(void)x;(void)level;
# 329
::exit(___);}
#if 0
# 325
{ 
# 326
T ret; 
# 327
tex1DLod(&ret, texObject, x, level); 
# 328
return ret; 
# 329
} 
#endif
# 332 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 333
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level) 
# 334
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;
# 336
::exit(___);}
#if 0
# 334
{ 
# 335
__nv_tex_surf_handler("__itex2DLod", ptr, obj, x, y, level); 
# 336
} 
#endif
# 338 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 339
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level) 
# 340
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;
# 344
::exit(___);}
#if 0
# 340
{ 
# 341
T ret; 
# 342
tex2DLod(&ret, texObject, x, y, level); 
# 343
return ret; 
# 344
} 
#endif
# 348 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 349
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level, bool *isResident) 
# 350
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;(void)isResident;
# 354
::exit(___);}
#if 0
# 350
{ 
# 351
unsigned char res; 
# 352
__nv_tex_surf_handler("__itex2DLod_sparse", ptr, obj, x, y, level, &res); 
# 353
(*isResident) = (res != 0); 
# 354
} 
#endif
# 356 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 357
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level, bool *isResident) 
# 358
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;(void)isResident;
# 362
::exit(___);}
#if 0
# 358
{ 
# 359
T ret; 
# 360
tex2DLod(&ret, texObject, x, y, level, isResident); 
# 361
return ret; 
# 362
} 
#endif
# 367 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 368
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 369
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 371
::exit(___);}
#if 0
# 369
{ 
# 370
__nv_tex_surf_handler("__itex3DLod", ptr, obj, x, y, z, level); 
# 371
} 
#endif
# 373 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 374
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 375
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 379
::exit(___);}
#if 0
# 375
{ 
# 376
T ret; 
# 377
tex3DLod(&ret, texObject, x, y, z, level); 
# 378
return ret; 
# 379
} 
#endif
# 382 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 383
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level, bool *isResident) 
# 384
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;(void)isResident;
# 388
::exit(___);}
#if 0
# 384
{ 
# 385
unsigned char res; 
# 386
__nv_tex_surf_handler("__itex3DLod_sparse", ptr, obj, x, y, z, level, &res); 
# 387
(*isResident) = (res != 0); 
# 388
} 
#endif
# 390 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 391
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level, bool *isResident) 
# 392
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;(void)isResident;
# 396
::exit(___);}
#if 0
# 392
{ 
# 393
T ret; 
# 394
tex3DLod(&ret, texObject, x, y, z, level, isResident); 
# 395
return ret; 
# 396
} 
#endif
# 401 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 402
tex1DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, int layer, float level) 
# 403
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;
# 405
::exit(___);}
#if 0
# 403
{ 
# 404
__nv_tex_surf_handler("__itex1DLayeredLod", ptr, obj, x, layer, level); 
# 405
} 
#endif
# 407 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 408
tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level) 
# 409
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;
# 413
::exit(___);}
#if 0
# 409
{ 
# 410
T ret; 
# 411
tex1DLayeredLod(&ret, texObject, x, layer, level); 
# 412
return ret; 
# 413
} 
#endif
# 416 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 417
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level) 
# 418
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;
# 420
::exit(___);}
#if 0
# 418
{ 
# 419
__nv_tex_surf_handler("__itex2DLayeredLod", ptr, obj, x, y, layer, level); 
# 420
} 
#endif
# 422 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 423
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level) 
# 424
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;
# 428
::exit(___);}
#if 0
# 424
{ 
# 425
T ret; 
# 426
tex2DLayeredLod(&ret, texObject, x, y, layer, level); 
# 427
return ret; 
# 428
} 
#endif
# 431 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 432
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level, bool *isResident) 
# 433
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;(void)isResident;
# 437
::exit(___);}
#if 0
# 433
{ 
# 434
unsigned char res; 
# 435
__nv_tex_surf_handler("__itex2DLayeredLod_sparse", ptr, obj, x, y, layer, level, &res); 
# 436
(*isResident) = (res != 0); 
# 437
} 
#endif
# 439 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 440
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level, bool *isResident) 
# 441
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;(void)isResident;
# 445
::exit(___);}
#if 0
# 441
{ 
# 442
T ret; 
# 443
tex2DLayeredLod(&ret, texObject, x, y, layer, level, isResident); 
# 444
return ret; 
# 445
} 
#endif
# 448 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 449
texCubemapLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 450
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 452
::exit(___);}
#if 0
# 450
{ 
# 451
__nv_tex_surf_handler("__itexCubemapLod", ptr, obj, x, y, z, level); 
# 452
} 
#endif
# 454 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 455
texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 456
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 460
::exit(___);}
#if 0
# 456
{ 
# 457
T ret; 
# 458
texCubemapLod(&ret, texObject, x, y, z, level); 
# 459
return ret; 
# 460
} 
#endif
# 463 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 464
texCubemapGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 465
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 467
::exit(___);}
#if 0
# 465
{ 
# 466
__nv_tex_surf_handler("__itexCubemapGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy); 
# 467
} 
#endif
# 469 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 470
texCubemapGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 471
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 475
::exit(___);}
#if 0
# 471
{ 
# 472
T ret; 
# 473
texCubemapGrad(&ret, texObject, x, y, z, dPdx, dPdy); 
# 474
return ret; 
# 475
} 
#endif
# 477 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 478
texCubemapLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float level) 
# 479
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 481
::exit(___);}
#if 0
# 479
{ 
# 480
__nv_tex_surf_handler("__itexCubemapLayeredLod", ptr, obj, x, y, z, layer, level); 
# 481
} 
#endif
# 483 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 484
texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) 
# 485
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 489
::exit(___);}
#if 0
# 485
{ 
# 486
T ret; 
# 487
texCubemapLayeredLod(&ret, texObject, x, y, z, layer, level); 
# 488
return ret; 
# 489
} 
#endif
# 491 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 492
tex1DGrad(T *ptr, cudaTextureObject_t obj, float x, float dPdx, float dPdy) 
# 493
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;
# 495
::exit(___);}
#if 0
# 493
{ 
# 494
__nv_tex_surf_handler("__itex1DGrad", ptr, obj, x, dPdx, dPdy); 
# 495
} 
#endif
# 497 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 498
tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy) 
# 499
{int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;
# 503
::exit(___);}
#if 0
# 499
{ 
# 500
T ret; 
# 501
tex1DGrad(&ret, texObject, x, dPdx, dPdy); 
# 502
return ret; 
# 503
} 
#endif
# 506 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 507
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy) 
# 508
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 510
::exit(___);}
#if 0
# 508
{ 
# 509
__nv_tex_surf_handler("__itex2DGrad_v2", ptr, obj, x, y, &dPdx, &dPdy); 
# 510
} 
#endif
# 512 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 513
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy) 
# 514
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 518
::exit(___);}
#if 0
# 514
{ 
# 515
T ret; 
# 516
tex2DGrad(&ret, texObject, x, y, dPdx, dPdy); 
# 517
return ret; 
# 518
} 
#endif
# 521 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 522
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy, bool *isResident) 
# 523
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;
# 527
::exit(___);}
#if 0
# 523
{ 
# 524
unsigned char res; 
# 525
__nv_tex_surf_handler("__itex2DGrad_sparse", ptr, obj, x, y, &dPdx, &dPdy, &res); 
# 526
(*isResident) = (res != 0); 
# 527
} 
#endif
# 529 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 530
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy, bool *isResident) 
# 531
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;
# 535
::exit(___);}
#if 0
# 531
{ 
# 532
T ret; 
# 533
tex2DGrad(&ret, texObject, x, y, dPdx, dPdy, isResident); 
# 534
return ret; 
# 535
} 
#endif
# 539 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 540
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 541
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 543
::exit(___);}
#if 0
# 541
{ 
# 542
__nv_tex_surf_handler("__itex3DGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy); 
# 543
} 
#endif
# 545 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 546
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 547
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 551
::exit(___);}
#if 0
# 547
{ 
# 548
T ret; 
# 549
tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy); 
# 550
return ret; 
# 551
} 
#endif
# 554 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 555
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy, bool *isResident) 
# 556
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;
# 560
::exit(___);}
#if 0
# 556
{ 
# 557
unsigned char res; 
# 558
__nv_tex_surf_handler("__itex3DGrad_sparse", ptr, obj, x, y, z, &dPdx, &dPdy, &res); 
# 559
(*isResident) = (res != 0); 
# 560
} 
#endif
# 562 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 563
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy, bool *isResident) 
# 564
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;
# 568
::exit(___);}
#if 0
# 564
{ 
# 565
T ret; 
# 566
tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy, isResident); 
# 567
return ret; 
# 568
} 
#endif
# 573 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 574
tex1DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) 
# 575
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 577
::exit(___);}
#if 0
# 575
{ 
# 576
__nv_tex_surf_handler("__itex1DLayeredGrad", ptr, obj, x, layer, dPdx, dPdy); 
# 577
} 
#endif
# 579 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 580
tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) 
# 581
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 585
::exit(___);}
#if 0
# 581
{ 
# 582
T ret; 
# 583
tex1DLayeredGrad(&ret, texObject, x, layer, dPdx, dPdy); 
# 584
return ret; 
# 585
} 
#endif
# 588 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 589
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 590
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 592
::exit(___);}
#if 0
# 590
{ 
# 591
__nv_tex_surf_handler("__itex2DLayeredGrad_v2", ptr, obj, x, y, layer, &dPdx, &dPdy); 
# 592
} 
#endif
# 594 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 595
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 596
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 600
::exit(___);}
#if 0
# 596
{ 
# 597
T ret; 
# 598
tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy); 
# 599
return ret; 
# 600
} 
#endif
# 603 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 604
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy, bool *isResident) 
# 605
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;
# 609
::exit(___);}
#if 0
# 605
{ 
# 606
unsigned char res; 
# 607
__nv_tex_surf_handler("__itex2DLayeredGrad_sparse", ptr, obj, x, y, layer, &dPdx, &dPdy, &res); 
# 608
(*isResident) = (res != 0); 
# 609
} 
#endif
# 611 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 612
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy, bool *isResident) 
# 613
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;
# 617
::exit(___);}
#if 0
# 613
{ 
# 614
T ret; 
# 615
tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy, isResident); 
# 616
return ret; 
# 617
} 
#endif
# 621 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 622
texCubemapLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 623
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 625
::exit(___);}
#if 0
# 623
{ 
# 624
__nv_tex_surf_handler("__itexCubemapLayeredGrad_v2", ptr, obj, x, y, z, layer, &dPdx, &dPdy); 
# 625
} 
#endif
# 627 "/usr/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 628
texCubemapLayeredGrad(cudaTextureObject_t texObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 629
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 633
::exit(___);}
#if 0
# 629
{ 
# 630
T ret; 
# 631
texCubemapLayeredGrad(&ret, texObject, x, y, z, layer, dPdx, dPdy); 
# 632
return ret; 
# 633
} 
#endif
# 58 "/usr/include/surface_indirect_functions.h" 3
template< class T> struct __nv_isurf_trait { }; 
# 59
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
# 60
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
# 61
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
# 62
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
# 63
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
# 64
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
# 65
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
# 66
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
# 67
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
# 68
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
# 69
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
# 70
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
# 71
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
# 72
template<> struct __nv_isurf_trait< long long>  { typedef void type; }; 
# 73
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
# 74
template<> struct __nv_isurf_trait< unsigned long long>  { typedef void type; }; 
# 75
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
# 76
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
# 77
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
# 79
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
# 80
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
# 81
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
# 82
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
# 83
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
# 84
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
# 85
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
# 86
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
# 87
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
# 89
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
# 90
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
# 91
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
# 92
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
# 93
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
# 94
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
# 95
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
# 98
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 99
surf1Dread(T *ptr, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 100
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;
# 102
::exit(___);}
#if 0
# 100
{ 
# 101
__nv_tex_surf_handler("__isurf1Dread", ptr, obj, x, mode); 
# 102
} 
#endif
# 104 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 105
surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 106
{int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;
# 110
::exit(___);}
#if 0
# 106
{ 
# 107
T ret; 
# 108
surf1Dread(&ret, surfObject, x, boundaryMode); 
# 109
return ret; 
# 110
} 
#endif
# 112 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 113
surf2Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 114
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;
# 116
::exit(___);}
#if 0
# 114
{ 
# 115
__nv_tex_surf_handler("__isurf2Dread", ptr, obj, x, y, mode); 
# 116
} 
#endif
# 118 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 119
surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 120
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;
# 124
::exit(___);}
#if 0
# 120
{ 
# 121
T ret; 
# 122
surf2Dread(&ret, surfObject, x, y, boundaryMode); 
# 123
return ret; 
# 124
} 
#endif
# 127 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 128
surf3Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 129
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 131
::exit(___);}
#if 0
# 129
{ 
# 130
__nv_tex_surf_handler("__isurf3Dread", ptr, obj, x, y, z, mode); 
# 131
} 
#endif
# 133 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 134
surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 135
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;
# 139
::exit(___);}
#if 0
# 135
{ 
# 136
T ret; 
# 137
surf3Dread(&ret, surfObject, x, y, z, boundaryMode); 
# 138
return ret; 
# 139
} 
#endif
# 141 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 142
surf1DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 143
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;
# 145
::exit(___);}
#if 0
# 143
{ 
# 144
__nv_tex_surf_handler("__isurf1DLayeredread", ptr, obj, x, layer, mode); 
# 145
} 
#endif
# 147 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 148
surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 149
{int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;
# 153
::exit(___);}
#if 0
# 149
{ 
# 150
T ret; 
# 151
surf1DLayeredread(&ret, surfObject, x, layer, boundaryMode); 
# 152
return ret; 
# 153
} 
#endif
# 155 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 156
surf2DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 157
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 159
::exit(___);}
#if 0
# 157
{ 
# 158
__nv_tex_surf_handler("__isurf2DLayeredread", ptr, obj, x, y, layer, mode); 
# 159
} 
#endif
# 161 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 162
surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 163
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;
# 167
::exit(___);}
#if 0
# 163
{ 
# 164
T ret; 
# 165
surf2DLayeredread(&ret, surfObject, x, y, layer, boundaryMode); 
# 166
return ret; 
# 167
} 
#endif
# 169 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 170
surfCubemapread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 171
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 173
::exit(___);}
#if 0
# 171
{ 
# 172
__nv_tex_surf_handler("__isurfCubemapread", ptr, obj, x, y, face, mode); 
# 173
} 
#endif
# 175 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 176
surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 177
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;
# 181
::exit(___);}
#if 0
# 177
{ 
# 178
T ret; 
# 179
surfCubemapread(&ret, surfObject, x, y, face, boundaryMode); 
# 180
return ret; 
# 181
} 
#endif
# 183 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 184
surfCubemapLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 185
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 187
::exit(___);}
#if 0
# 185
{ 
# 186
__nv_tex_surf_handler("__isurfCubemapLayeredread", ptr, obj, x, y, layerface, mode); 
# 187
} 
#endif
# 189 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 190
surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 191
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;
# 195
::exit(___);}
#if 0
# 191
{ 
# 192
T ret; 
# 193
surfCubemapLayeredread(&ret, surfObject, x, y, layerface, boundaryMode); 
# 194
return ret; 
# 195
} 
#endif
# 197 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 198
surf1Dwrite(T val, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 199
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;
# 201
::exit(___);}
#if 0
# 199
{ 
# 200
__nv_tex_surf_handler("__isurf1Dwrite_v2", &val, obj, x, mode); 
# 201
} 
#endif
# 203 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 204
surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 205
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;
# 207
::exit(___);}
#if 0
# 205
{ 
# 206
__nv_tex_surf_handler("__isurf2Dwrite_v2", &val, obj, x, y, mode); 
# 207
} 
#endif
# 209 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 210
surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 211
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 213
::exit(___);}
#if 0
# 211
{ 
# 212
__nv_tex_surf_handler("__isurf3Dwrite_v2", &val, obj, x, y, z, mode); 
# 213
} 
#endif
# 215 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 216
surf1DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 217
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;
# 219
::exit(___);}
#if 0
# 217
{ 
# 218
__nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, obj, x, layer, mode); 
# 219
} 
#endif
# 221 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 222
surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 223
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 225
::exit(___);}
#if 0
# 223
{ 
# 224
__nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, obj, x, y, layer, mode); 
# 225
} 
#endif
# 227 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 228
surfCubemapwrite(T val, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 229
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 231
::exit(___);}
#if 0
# 229
{ 
# 230
__nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, obj, x, y, face, mode); 
# 231
} 
#endif
# 233 "/usr/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 234
surfCubemapLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 235
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 237
::exit(___);}
#if 0
# 235
{ 
# 236
__nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, obj, x, y, layerface, mode); 
# 237
} 
#endif
# 3634 "/usr/include/crt/device_functions.h" 3
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, CUstream_st * stream = 0); 
# 68 "/usr/include/device_launch_parameters.h" 3
extern "C" {
# 71
extern const uint3 __device_builtin_variable_threadIdx; 
# 72
extern const uint3 __device_builtin_variable_blockIdx; 
# 73
extern const dim3 __device_builtin_variable_blockDim; 
# 74
extern const dim3 __device_builtin_variable_gridDim; 
# 75
extern const int __device_builtin_variable_warpSize; 
# 80
}
# 62 "/usr/include/c++/12/bits/stl_relops.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 66
namespace rel_ops { 
# 86 "/usr/include/c++/12/bits/stl_relops.h" 3
template< class _Tp> inline bool 
# 88
operator!=(const _Tp &__x, const _Tp &__y) 
# 89
{ return !(__x == __y); } 
# 99 "/usr/include/c++/12/bits/stl_relops.h" 3
template< class _Tp> inline bool 
# 101
operator>(const _Tp &__x, const _Tp &__y) 
# 102
{ return __y < __x; } 
# 112 "/usr/include/c++/12/bits/stl_relops.h" 3
template< class _Tp> inline bool 
# 114
operator<=(const _Tp &__x, const _Tp &__y) 
# 115
{ return !(__y < __x); } 
# 125 "/usr/include/c++/12/bits/stl_relops.h" 3
template< class _Tp> inline bool 
# 127
operator>=(const _Tp &__x, const _Tp &__y) 
# 128
{ return !(__x < __y); } 
# 129
}
# 132
}
# 39 "/usr/include/c++/12/initializer_list" 3
#pragma GCC visibility push ( default )
# 43
namespace std { 
# 46
template< class _E> 
# 47
class initializer_list { 
# 50
public: typedef _E value_type; 
# 51
typedef const _E &reference; 
# 52
typedef const _E &const_reference; 
# 53
typedef size_t size_type; 
# 54
typedef const _E *iterator; 
# 55
typedef const _E *const_iterator; 
# 58
private: iterator _M_array; 
# 59
size_type _M_len; 
# 62
constexpr initializer_list(const_iterator __a, size_type __l) : _M_array(__a), _M_len(__l) 
# 63
{ } 
# 66
public: constexpr initializer_list() noexcept : _M_array((0)), _M_len((0)) 
# 67
{ } 
# 71
constexpr size_type size() const noexcept { return _M_len; } 
# 75
constexpr const_iterator begin() const noexcept { return _M_array; } 
# 79
constexpr const_iterator end() const noexcept { return begin() + size(); } 
# 80
}; 
# 88
template< class _Tp> constexpr const _Tp *
# 90
begin(initializer_list< _Tp>  __ils) noexcept 
# 91
{ return __ils.begin(); } 
# 99
template< class _Tp> constexpr const _Tp *
# 101
end(initializer_list< _Tp>  __ils) noexcept 
# 102
{ return __ils.end(); } 
# 103
}
# 105
#pragma GCC visibility pop
# 82 "/usr/include/c++/12/utility" 3
namespace std __attribute((__visibility__("default"))) { 
# 90
template< class _Tp, class _Up = _Tp> inline _Tp 
# 93
exchange(_Tp &__obj, _Up &&__new_val) noexcept(__and_< is_nothrow_move_constructible< _Tp> , is_nothrow_assignable< _Tp &, _Up> > ::value) 
# 96
{ return std::__exchange(__obj, std::forward< _Up> (__new_val)); } 
# 101
template< class _Tp> 
# 102
[[nodiscard]] constexpr add_const_t< _Tp>  &
# 104
as_const(_Tp &__t) noexcept 
# 105
{ return __t; } 
# 107
template < typename _Tp >
    void as_const ( const _Tp && ) = delete;
# 221 "/usr/include/c++/12/utility" 3
}
# 206 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 207
cudaLaunchKernel(const T *
# 208
func, dim3 
# 209
gridDim, dim3 
# 210
blockDim, void **
# 211
args, size_t 
# 212
sharedMem = 0, cudaStream_t 
# 213
stream = 0) 
# 215
{ 
# 216
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 217
} 
# 277 "/usr/include/cuda_runtime.h" 3
template< class ...ExpTypes, class ...ActTypes> static inline cudaError_t 
# 278
cudaLaunchKernelEx(const cudaLaunchConfig_t *
# 279
config, void (*
# 280
kernel)(ExpTypes ...), ActTypes &&...
# 281
args) 
# 283
{ 
# 284
return [&](ExpTypes ...coercedArgs) { 
# 285
void *pArgs[] = {(&coercedArgs)...}; 
# 286
return ::cudaLaunchKernelExC(config, (const void *)(kernel), pArgs); 
# 287
} (std::forward< ActTypes> (args)...); 
# 288
} 
# 340 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 341
cudaLaunchCooperativeKernel(const T *
# 342
func, dim3 
# 343
gridDim, dim3 
# 344
blockDim, void **
# 345
args, size_t 
# 346
sharedMem = 0, cudaStream_t 
# 347
stream = 0) 
# 349
{ 
# 350
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 351
} 
# 384 "/usr/include/cuda_runtime.h" 3
static inline cudaError_t cudaEventCreate(cudaEvent_t *
# 385
event, unsigned 
# 386
flags) 
# 388
{ 
# 389
return ::cudaEventCreateWithFlags(event, flags); 
# 390
} 
# 428 "/usr/include/cuda_runtime.h" 3
static inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t *
# 429
pGraphExec, cudaGraph_t 
# 430
graph, cudaGraphNode_t *
# 431
pErrorNode, char *
# 432
pLogBuffer, size_t 
# 433
bufferSize) 
# 435
{ 
# 436
(void)pErrorNode; 
# 437
(void)pLogBuffer; 
# 438
(void)bufferSize; 
# 439
return ::cudaGraphInstantiate(pGraphExec, graph, 0); 
# 440
} 
# 499 "/usr/include/cuda_runtime.h" 3
static inline cudaError_t cudaMallocHost(void **
# 500
ptr, size_t 
# 501
size, unsigned 
# 502
flags) 
# 504
{ 
# 505
return ::cudaHostAlloc(ptr, size, flags); 
# 506
} 
# 508
template< class T> static inline cudaError_t 
# 509
cudaHostAlloc(T **
# 510
ptr, size_t 
# 511
size, unsigned 
# 512
flags) 
# 514
{ 
# 515
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
# 516
} 
# 518
template< class T> static inline cudaError_t 
# 519
cudaHostGetDevicePointer(T **
# 520
pDevice, void *
# 521
pHost, unsigned 
# 522
flags) 
# 524
{ 
# 525
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
# 526
} 
# 628 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 629
cudaMallocManaged(T **
# 630
devPtr, size_t 
# 631
size, unsigned 
# 632
flags = 1) 
# 634
{ 
# 635
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
# 636
} 
# 718 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 719
cudaStreamAttachMemAsync(cudaStream_t 
# 720
stream, T *
# 721
devPtr, size_t 
# 722
length = 0, unsigned 
# 723
flags = 4) 
# 725
{ 
# 726
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
# 727
} 
# 729
template< class T> inline cudaError_t 
# 730
cudaMalloc(T **
# 731
devPtr, size_t 
# 732
size) 
# 734
{ 
# 735
return ::cudaMalloc((void **)((void *)devPtr), size); 
# 736
} 
# 738
template< class T> static inline cudaError_t 
# 739
cudaMallocHost(T **
# 740
ptr, size_t 
# 741
size, unsigned 
# 742
flags = 0) 
# 744
{ 
# 745
return cudaMallocHost((void **)((void *)ptr), size, flags); 
# 746
} 
# 748
template< class T> static inline cudaError_t 
# 749
cudaMallocPitch(T **
# 750
devPtr, size_t *
# 751
pitch, size_t 
# 752
width, size_t 
# 753
height) 
# 755
{ 
# 756
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
# 757
} 
# 768 "/usr/include/cuda_runtime.h" 3
static inline cudaError_t cudaMallocAsync(void **
# 769
ptr, size_t 
# 770
size, cudaMemPool_t 
# 771
memPool, cudaStream_t 
# 772
stream) 
# 774
{ 
# 775
return ::cudaMallocFromPoolAsync(ptr, size, memPool, stream); 
# 776
} 
# 778
template< class T> static inline cudaError_t 
# 779
cudaMallocAsync(T **
# 780
ptr, size_t 
# 781
size, cudaMemPool_t 
# 782
memPool, cudaStream_t 
# 783
stream) 
# 785
{ 
# 786
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
# 787
} 
# 789
template< class T> static inline cudaError_t 
# 790
cudaMallocAsync(T **
# 791
ptr, size_t 
# 792
size, cudaStream_t 
# 793
stream) 
# 795
{ 
# 796
return ::cudaMallocAsync((void **)((void *)ptr), size, stream); 
# 797
} 
# 799
template< class T> static inline cudaError_t 
# 800
cudaMallocFromPoolAsync(T **
# 801
ptr, size_t 
# 802
size, cudaMemPool_t 
# 803
memPool, cudaStream_t 
# 804
stream) 
# 806
{ 
# 807
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
# 808
} 
# 847 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 848
cudaMemcpyToSymbol(const T &
# 849
symbol, const void *
# 850
src, size_t 
# 851
count, size_t 
# 852
offset = 0, cudaMemcpyKind 
# 853
kind = cudaMemcpyHostToDevice) 
# 855
{ 
# 856
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
# 857
} 
# 901 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 902
cudaMemcpyToSymbolAsync(const T &
# 903
symbol, const void *
# 904
src, size_t 
# 905
count, size_t 
# 906
offset = 0, cudaMemcpyKind 
# 907
kind = cudaMemcpyHostToDevice, cudaStream_t 
# 908
stream = 0) 
# 910
{ 
# 911
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
# 912
} 
# 949 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 950
cudaMemcpyFromSymbol(void *
# 951
dst, const T &
# 952
symbol, size_t 
# 953
count, size_t 
# 954
offset = 0, cudaMemcpyKind 
# 955
kind = cudaMemcpyDeviceToHost) 
# 957
{ 
# 958
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
# 959
} 
# 1003 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1004
cudaMemcpyFromSymbolAsync(void *
# 1005
dst, const T &
# 1006
symbol, size_t 
# 1007
count, size_t 
# 1008
offset = 0, cudaMemcpyKind 
# 1009
kind = cudaMemcpyDeviceToHost, cudaStream_t 
# 1010
stream = 0) 
# 1012
{ 
# 1013
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
# 1014
} 
# 1072 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1073
cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *
# 1074
pGraphNode, cudaGraph_t 
# 1075
graph, const cudaGraphNode_t *
# 1076
pDependencies, size_t 
# 1077
numDependencies, const T &
# 1078
symbol, const void *
# 1079
src, size_t 
# 1080
count, size_t 
# 1081
offset, cudaMemcpyKind 
# 1082
kind) 
# 1083
{ 
# 1084
return ::cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, (const void *)(&symbol), src, count, offset, kind); 
# 1085
} 
# 1143 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1144
cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *
# 1145
pGraphNode, cudaGraph_t 
# 1146
graph, const cudaGraphNode_t *
# 1147
pDependencies, size_t 
# 1148
numDependencies, void *
# 1149
dst, const T &
# 1150
symbol, size_t 
# 1151
count, size_t 
# 1152
offset, cudaMemcpyKind 
# 1153
kind) 
# 1154
{ 
# 1155
return ::cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, (const void *)(&symbol), count, offset, kind); 
# 1156
} 
# 1194 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1195
cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t 
# 1196
node, const T &
# 1197
symbol, const void *
# 1198
src, size_t 
# 1199
count, size_t 
# 1200
offset, cudaMemcpyKind 
# 1201
kind) 
# 1202
{ 
# 1203
return ::cudaGraphMemcpyNodeSetParamsToSymbol(node, (const void *)(&symbol), src, count, offset, kind); 
# 1204
} 
# 1242 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1243
cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t 
# 1244
node, void *
# 1245
dst, const T &
# 1246
symbol, size_t 
# 1247
count, size_t 
# 1248
offset, cudaMemcpyKind 
# 1249
kind) 
# 1250
{ 
# 1251
return ::cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, (const void *)(&symbol), count, offset, kind); 
# 1252
} 
# 1300 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1301
cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t 
# 1302
hGraphExec, cudaGraphNode_t 
# 1303
node, const T &
# 1304
symbol, const void *
# 1305
src, size_t 
# 1306
count, size_t 
# 1307
offset, cudaMemcpyKind 
# 1308
kind) 
# 1309
{ 
# 1310
return ::cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, (const void *)(&symbol), src, count, offset, kind); 
# 1311
} 
# 1359 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1360
cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t 
# 1361
hGraphExec, cudaGraphNode_t 
# 1362
node, void *
# 1363
dst, const T &
# 1364
symbol, size_t 
# 1365
count, size_t 
# 1366
offset, cudaMemcpyKind 
# 1367
kind) 
# 1368
{ 
# 1369
return ::cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, (const void *)(&symbol), count, offset, kind); 
# 1370
} 
# 1373
static inline cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out, cudaGraphExecUpdateResult *updateResult_out) 
# 1374
{ 
# 1375
cudaGraphExecUpdateResultInfo resultInfo; 
# 1376
cudaError_t status = cudaGraphExecUpdate(hGraphExec, hGraph, &resultInfo); 
# 1377
if (hErrorNode_out) { 
# 1378
(*hErrorNode_out) = (resultInfo.errorNode); 
# 1379
}  
# 1380
if (updateResult_out) { 
# 1381
(*updateResult_out) = (resultInfo.result); 
# 1382
}  
# 1383
return status; 
# 1384
} 
# 1412 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1413
cudaUserObjectCreate(cudaUserObject_t *
# 1414
object_out, T *
# 1415
objectToWrap, unsigned 
# 1416
initialRefcount, unsigned 
# 1417
flags) 
# 1418
{ 
# 1419
return ::cudaUserObjectCreate(object_out, objectToWrap, [](void *
# 1422
vpObj) { delete (reinterpret_cast< T *>(vpObj)); } , initialRefcount, flags); 
# 1425
} 
# 1427
template< class T> static inline cudaError_t 
# 1428
cudaUserObjectCreate(cudaUserObject_t *
# 1429
object_out, T *
# 1430
objectToWrap, unsigned 
# 1431
initialRefcount, cudaUserObjectFlags 
# 1432
flags) 
# 1433
{ 
# 1434
return cudaUserObjectCreate(object_out, objectToWrap, initialRefcount, (unsigned)flags); 
# 1435
} 
# 1462 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1463
cudaGetSymbolAddress(void **
# 1464
devPtr, const T &
# 1465
symbol) 
# 1467
{ 
# 1468
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
# 1469
} 
# 1494 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1495
cudaGetSymbolSize(size_t *
# 1496
size, const T &
# 1497
symbol) 
# 1499
{ 
# 1500
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
# 1501
} 
# 1546 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1547
cudaFuncSetCacheConfig(T *
# 1548
func, cudaFuncCache 
# 1549
cacheConfig) 
# 1551
{ 
# 1552
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
# 1553
} 
# 1555
template< class T> static inline cudaError_t 
# 1556
cudaFuncSetSharedMemConfig(T *
# 1557
func, cudaSharedMemConfig 
# 1558
config) 
# 1560
{ 
# 1561
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
# 1562
} 
# 1594 "/usr/include/cuda_runtime.h" 3
template< class T> inline cudaError_t 
# 1595
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
# 1596
numBlocks, T 
# 1597
func, int 
# 1598
blockSize, size_t 
# 1599
dynamicSMemSize) 
# 1600
{ 
# 1601
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
# 1602
} 
# 1646 "/usr/include/cuda_runtime.h" 3
template< class T> inline cudaError_t 
# 1647
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
# 1648
numBlocks, T 
# 1649
func, int 
# 1650
blockSize, size_t 
# 1651
dynamicSMemSize, unsigned 
# 1652
flags) 
# 1653
{ 
# 1654
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
# 1655
} 
# 1660
class __cudaOccupancyB2DHelper { 
# 1661
size_t n; 
# 1663
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
# 1664
size_t operator()(int) 
# 1665
{ 
# 1666
return n; 
# 1667
} 
# 1668
}; 
# 1716 "/usr/include/cuda_runtime.h" 3
template< class UnaryFunction, class T> static inline cudaError_t 
# 1717
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
# 1718
minGridSize, int *
# 1719
blockSize, T 
# 1720
func, UnaryFunction 
# 1721
blockSizeToDynamicSMemSize, int 
# 1722
blockSizeLimit = 0, unsigned 
# 1723
flags = 0) 
# 1724
{ 
# 1725
cudaError_t status; 
# 1728
int device; 
# 1729
cudaFuncAttributes attr; 
# 1732
int maxThreadsPerMultiProcessor; 
# 1733
int warpSize; 
# 1734
int devMaxThreadsPerBlock; 
# 1735
int multiProcessorCount; 
# 1736
int funcMaxThreadsPerBlock; 
# 1737
int occupancyLimit; 
# 1738
int granularity; 
# 1741
int maxBlockSize = 0; 
# 1742
int numBlocks = 0; 
# 1743
int maxOccupancy = 0; 
# 1746
int blockSizeToTryAligned; 
# 1747
int blockSizeToTry; 
# 1748
int blockSizeLimitAligned; 
# 1749
int occupancyInBlocks; 
# 1750
int occupancyInThreads; 
# 1751
size_t dynamicSMemSize; 
# 1757
if (((!minGridSize) || (!blockSize)) || (!func)) { 
# 1758
return cudaErrorInvalidValue; 
# 1759
}  
# 1765
status = ::cudaGetDevice(&device); 
# 1766
if (status != (cudaSuccess)) { 
# 1767
return status; 
# 1768
}  
# 1770
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
# 1774
if (status != (cudaSuccess)) { 
# 1775
return status; 
# 1776
}  
# 1778
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
# 1782
if (status != (cudaSuccess)) { 
# 1783
return status; 
# 1784
}  
# 1786
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
# 1790
if (status != (cudaSuccess)) { 
# 1791
return status; 
# 1792
}  
# 1794
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
# 1798
if (status != (cudaSuccess)) { 
# 1799
return status; 
# 1800
}  
# 1802
status = cudaFuncGetAttributes(&attr, func); 
# 1803
if (status != (cudaSuccess)) { 
# 1804
return status; 
# 1805
}  
# 1807
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
# 1813
occupancyLimit = maxThreadsPerMultiProcessor; 
# 1814
granularity = warpSize; 
# 1816
if (blockSizeLimit == 0) { 
# 1817
blockSizeLimit = devMaxThreadsPerBlock; 
# 1818
}  
# 1820
if (devMaxThreadsPerBlock < blockSizeLimit) { 
# 1821
blockSizeLimit = devMaxThreadsPerBlock; 
# 1822
}  
# 1824
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
# 1825
blockSizeLimit = funcMaxThreadsPerBlock; 
# 1826
}  
# 1828
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
# 1830
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
# 1834
if (blockSizeLimit < blockSizeToTryAligned) { 
# 1835
blockSizeToTry = blockSizeLimit; 
# 1836
} else { 
# 1837
blockSizeToTry = blockSizeToTryAligned; 
# 1838
}  
# 1840
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
# 1842
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
# 1849
if (status != (cudaSuccess)) { 
# 1850
return status; 
# 1851
}  
# 1853
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
# 1855
if (occupancyInThreads > maxOccupancy) { 
# 1856
maxBlockSize = blockSizeToTry; 
# 1857
numBlocks = occupancyInBlocks; 
# 1858
maxOccupancy = occupancyInThreads; 
# 1859
}  
# 1863
if (occupancyLimit == maxOccupancy) { 
# 1864
break; 
# 1865
}  
# 1866
}  
# 1874
(*minGridSize) = (numBlocks * multiProcessorCount); 
# 1875
(*blockSize) = maxBlockSize; 
# 1877
return status; 
# 1878
} 
# 1912 "/usr/include/cuda_runtime.h" 3
template< class UnaryFunction, class T> static inline cudaError_t 
# 1913
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
# 1914
minGridSize, int *
# 1915
blockSize, T 
# 1916
func, UnaryFunction 
# 1917
blockSizeToDynamicSMemSize, int 
# 1918
blockSizeLimit = 0) 
# 1919
{ 
# 1920
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
# 1921
} 
# 1958 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1959
cudaOccupancyMaxPotentialBlockSize(int *
# 1960
minGridSize, int *
# 1961
blockSize, T 
# 1962
func, size_t 
# 1963
dynamicSMemSize = 0, int 
# 1964
blockSizeLimit = 0) 
# 1965
{ 
# 1966
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
# 1967
} 
# 1996 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1997
cudaOccupancyAvailableDynamicSMemPerBlock(size_t *
# 1998
dynamicSmemSize, T 
# 1999
func, int 
# 2000
numBlocks, int 
# 2001
blockSize) 
# 2002
{ 
# 2003
return ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, (const void *)func, numBlocks, blockSize); 
# 2004
} 
# 2055 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 2056
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
# 2057
minGridSize, int *
# 2058
blockSize, T 
# 2059
func, size_t 
# 2060
dynamicSMemSize = 0, int 
# 2061
blockSizeLimit = 0, unsigned 
# 2062
flags = 0) 
# 2063
{ 
# 2064
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
# 2065
} 
# 2099 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 2100
cudaOccupancyMaxPotentialClusterSize(int *
# 2101
clusterSize, T *
# 2102
func, const cudaLaunchConfig_t *
# 2103
config) 
# 2104
{ 
# 2105
return ::cudaOccupancyMaxPotentialClusterSize(clusterSize, (const void *)func, config); 
# 2106
} 
# 2142 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 2143
cudaOccupancyMaxActiveClusters(int *
# 2144
numClusters, T *
# 2145
func, const cudaLaunchConfig_t *
# 2146
config) 
# 2147
{ 
# 2148
return ::cudaOccupancyMaxActiveClusters(numClusters, (const void *)func, config); 
# 2149
} 
# 2182 "/usr/include/cuda_runtime.h" 3
template< class T> inline cudaError_t 
# 2183
cudaFuncGetAttributes(cudaFuncAttributes *
# 2184
attr, T *
# 2185
entry) 
# 2187
{ 
# 2188
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
# 2189
} 
# 2244 "/usr/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 2245
cudaFuncSetAttribute(T *
# 2246
entry, cudaFuncAttribute 
# 2247
attr, int 
# 2248
value) 
# 2250
{ 
# 2251
return ::cudaFuncSetAttribute((const void *)entry, attr, value); 
# 2252
} 
# 2263 "/usr/include/cuda_runtime.h" 3
#pragma GCC diagnostic pop
# 64 "CMakeCUDACompilerId.cu"
const char *info_compiler = ("INFO:compiler[NVIDIA]"); 
# 66
const char *info_simulate = ("INFO:simulate[GNU]"); 
# 369 "CMakeCUDACompilerId.cu"
const char info_version[] = {'I', 'N', 'F', 'O', ':', 'c', 'o', 'm', 'p', 'i', 'l', 'e', 'r', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((12 / 10000000) % 10)), (('0') + ((12 / 1000000) % 10)), (('0') + ((12 / 100000) % 10)), (('0') + ((12 / 10000) % 10)), (('0') + ((12 / 1000) % 10)), (('0') + ((12 / 100) % 10)), (('0') + ((12 / 10) % 10)), (('0') + (12 % 10)), '.', (('0') + ((0 / 10000000) % 10)), (('0') + ((0 / 1000000) % 10)), (('0') + ((0 / 100000) % 10)), (('0') + ((0 / 10000) % 10)), (('0') + ((0 / 1000) % 10)), (('0') + ((0 / 100) % 10)), (('0') + ((0 / 10) % 10)), (('0') + (0 % 10)), '.', (('0') + ((140 / 10000000) % 10)), (('0') + ((140 / 1000000) % 10)), (('0') + ((140 / 100000) % 10)), (('0') + ((140 / 10000) % 10)), (('0') + ((140 / 1000) % 10)), (('0') + ((140 / 100) % 10)), (('0') + ((140 / 10) % 10)), (('0') + (140 % 10)), ']', '\000'}; 
# 398 "CMakeCUDACompilerId.cu"
const char info_simulate_version[] = {'I', 'N', 'F', 'O', ':', 's', 'i', 'm', 'u', 'l', 'a', 't', 'e', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((12 / 10000000) % 10)), (('0') + ((12 / 1000000) % 10)), (('0') + ((12 / 100000) % 10)), (('0') + ((12 / 10000) % 10)), (('0') + ((12 / 1000) % 10)), (('0') + ((12 / 100) % 10)), (('0') + ((12 / 10) % 10)), (('0') + (12 % 10)), '.', (('0') + ((3 / 10000000) % 10)), (('0') + ((3 / 1000000) % 10)), (('0') + ((3 / 100000) % 10)), (('0') + ((3 / 10000) % 10)), (('0') + ((3 / 1000) % 10)), (('0') + ((3 / 100) % 10)), (('0') + ((3 / 10) % 10)), (('0') + (3 % 10)), ']', '\000'}; 
# 418
const char *info_platform = ("INFO:platform[Linux]"); 
# 419
const char *info_arch = ("INFO:arch[]"); 
# 423
const char *info_language_standard_default = ("INFO:standard_default[17]"); 
# 439
const char *info_language_extensions_default = ("INFO:extensions_default[ON]"); 
# 450
int main(int argc, char *argv[]) 
# 451
{ 
# 452
int require = 0; 
# 453
require += (info_compiler[argc]); 
# 454
require += (info_platform[argc]); 
# 456
require += (info_version[argc]); 
# 459
require += (info_simulate[argc]); 
# 462
require += (info_simulate_version[argc]); 
# 464
require += (info_language_standard_default[argc]); 
# 465
require += (info_language_extensions_default[argc]); 
# 466
(void)argv; 
# 467
return require; 
# 468
} 

# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__7d1e1e9d_22_CMakeCUDACompilerId_cu_bd57c623
#ifdef _NV_ANON_NAMESPACE
#endif
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#include "CMakeCUDACompilerId.cudafe1.stub.c"
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
